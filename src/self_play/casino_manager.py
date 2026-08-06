import random
import threading
import time
import traceback

import ray
from ray.util.queue import Queue, Empty
import os
import torch

from src.global_settings import NUM_PLAYERS, NUM_TABLES, NUM_TRAINERS, RESOURCE_LIMITED, IS_RECURRENT, ALG, GAME_TYPE
from src.self_play.trainer_actor import TrainerActor
from src.self_play.leaderboard_actor import LeaderboardActor
from src.self_play.data_storage import DataStorage
from src.player_ai import PlayerAI, RNNPlayerAI
from torch.utils.tensorboard import SummaryWriter
from src.shared import SemanticTimer
from src.self_play.scheduler import JITTableScheduler, HistoricalSampling


class CasinoManager:
    def __init__(self, device: torch.device, save_folder: str = "./",
                 bc_pretrained_model_path: str = None, resume: bool = False):
        from src.game_registry import get_current_game_config
        try:
            self.player_ids = list(range(NUM_PLAYERS))
            self.device = device
            self.save_folder = save_folder
            os.makedirs(self.save_folder, exist_ok=True)
            self.player_save_folder = os.path.join(save_folder, "players")
            os.makedirs(self.player_save_folder, exist_ok=True)
            self.historical_save_folder = os.path.join(save_folder, "historical_checkpoints")
            os.makedirs(self.historical_save_folder, exist_ok=True)
            run_name = os.path.basename(save_folder)
            self.log_folder = os.path.join(os.path.dirname(save_folder), "tb_logs", run_name)
            os.makedirs(self.log_folder, exist_ok=True)
            self.mode = "categorical" if ALG == "NEURD" else "beta"
            self.discrete = True if ALG == "NEURD" else False
            manager_log_path = os.path.join(self.log_folder, "tensorboard_logs")
            self.writer = SummaryWriter(log_dir=manager_log_path)
            self.timer = SemanticTimer()
            self.loop_step = 0

            # player trackers
            self.is_playing = {player_id: False for player_id in self.player_ids}
            self.is_training = {player_id: False for player_id in self.player_ids}
            self.is_playing_against = {player_id: [] for player_id in self.player_ids}
            self.player_dispatch_times = {player_id: time.time() for player_id in self.player_ids}
            self.timeout_threshold = 3600  # 1 hour (adjust based on how long a normal game/training takes)
            self.last_timeout_check = time.time()
            self.alg_class = get_current_game_config()["alg"]
            self.inference_wrapper_class = get_current_game_config()["inference_wrapper"]
            # we spin up the player models
            if IS_RECURRENT:
                self.players = [ray.put(RNNPlayerAI(self.alg_class.init_networks(torch.device("cpu"), discrete=self.discrete, mode=self.mode))) for _ in
                                self.player_ids]
            else:
                self.players = [ray.put(PlayerAI(self.alg_class.init_networks(torch.device("cpu"), discrete=self.discrete, mode=self.mode))) for _ in
                                self.player_ids]
            self.player_training_counts = [0] * len(self.player_ids)
            
            if bc_pretrained_model_path is not None:
                print(f"Loading pretrained model from {bc_pretrained_model_path}")
                # load the model
                param_dicts = torch.load(bc_pretrained_model_path, map_location=torch.device("cpu"), weights_only=True)

                loaded_players = []
                for player in self.players:
                    player: PlayerAI = ray.get(player)
                    player.load_params(param_dicts)
                    loaded_players.append(ray.put(player))

                self.players = loaded_players
            elif resume:
                print(f"Resuming models from {self.player_save_folder}")
                loaded_players = []
                for player_id in self.player_ids:
                    player = ray.get(self.players[player_id])
                    model_path = os.path.join(self.player_save_folder, f"{player_id}.pt")
                    if os.path.exists(model_path):
                        loaded_data = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
                        if isinstance(loaded_data, tuple) and len(loaded_data) == 3:
                            new_weights, new_optimizer_params, count = loaded_data
                            player.load_params(new_weights)
                            player.load_optimizers(new_optimizer_params)
                            self.player_training_counts[player_id] = count
                        elif isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                            new_weights, new_optimizer_params = loaded_data
                            player.load_params(new_weights)
                            player.load_optimizers(new_optimizer_params)
                        else:
                            player.load_params(loaded_data)
                    loaded_players.append(ray.put(player))
                self.players = loaded_players

            if resume:
                import glob
                # Verify if any counts were NOT loaded from .pt files (older checkpoints fallback)
                for player_id in self.player_ids:
                    if self.player_training_counts[player_id] == 0:
                        hist_files = glob.glob(os.path.join(self.historical_save_folder, f"{player_id}_*.pt"))
                        if hist_files:
                            versions = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in hist_files]
                            self.player_training_counts[player_id] = max(versions)

            self.table_max_size = 2
            self.table_min_size = 2
            if IS_RECURRENT:
                # number of games
                # self.batch_size = 1_000 if RESOURCE_LIMITED else 8_000
                self.batch_size = 10 if RESOURCE_LIMITED else 80
            else:
                # number of transitions
                if GAME_TYPE == "KUHN":
                    self.batch_size = 500 if RESOURCE_LIMITED else 4_000
                else:
                    self.batch_size = 5_000 if RESOURCE_LIMITED else 40_000
            self.on_policy = True

            # self.table_scheduler = PlanTableScheduler(self.table_min_size, self.table_max_size, self.player_ids)

            self.historical_sampling_queue_len = 10
            self.historical_sampling_send_queue = Queue(maxsize=0)
            self.historical_sampling_receive_queue = Queue(maxsize=self.historical_sampling_queue_len)
            self.historical_sampler = HistoricalSampling.remote(self.player_ids, self.historical_sampling_send_queue,
                                                         self.historical_sampling_receive_queue, self.historical_save_folder,
                                                         self.discrete, self.mode)
            self.historical_sampler.start.remote()
            
            historical_players_used = 0
            if resume:
                import json
                state_path = os.path.join(self.save_folder, "leaderboard_state.json")
                if os.path.exists(state_path):
                    with open(state_path, "r") as f:
                        state = json.load(f)
                        historical_players_used = state.get("historical_players_used", 0)
                        
            self.table_scheduler = JITTableScheduler(self.table_min_size, self.table_max_size, self.player_ids, self.historical_sampling_receive_queue)
            self.table_scheduler.historical_players_used = historical_players_used

            self.table_send_queue = Queue(maxsize=0)
            self.table_receive_queue = Queue(maxsize=0)

            self.trainer_send_queue = Queue(maxsize=0)
            self.trainer_receive_queue = Queue(maxsize=0)

            # max_tables_needed = len(self.player_ids) // self.table_min_size
            print(f"Opening casino with {NUM_TABLES} permanent tables of size between {self.table_min_size} and "
                  f"{self.table_max_size}...")
            self.table_ids = [table_id for table_id in range(NUM_TABLES)]
            self.TableActor = get_current_game_config()['table_actor']
            self.tables = [self.TableActor.remote(table_id, device, self.table_send_queue, self.table_receive_queue,
                                             self.historical_sampling_receive_queue,
                                             self.table_max_size, self.discrete, self.mode,
                                             self.batch_size, self.log_folder) for table_id in self.table_ids]   # we spin up the tables at the beginning to avoid the churn
            for table in self.tables:
                table.start.remote()

            self.data_storage = DataStorage(self.player_ids, self.batch_size, self.log_folder)

            self.trainer_ids = [trainer_id for trainer_id in range(NUM_TRAINERS)]
            self.trainers = [TrainerActor.remote(i, self.trainer_send_queue, self.trainer_receive_queue,
                                                 self.historical_sampling_send_queue,
                                                 device, self.discrete,
                                                 self.log_folder, self.player_save_folder, self.mode)
                             for i in self.trainer_ids]
            for trainer in self.trainers:
                trainer.start.remote()

            self.leaderboard_queue = Queue(maxsize=0)
            # self.leaderboard = LeaderboardActor.remote(self.leaderboard_queue, self.player_ids, save_folder)
            self.leaderboard = LeaderboardActor.options(name="GlobalLeaderboard", namespace="casino").remote(
                self.leaderboard_queue, self.table_send_queue, self.table_receive_queue, self.trainer_send_queue,
                self.trainer_receive_queue, self.player_ids, save_folder)

            self.leaderboard.start.remote()
            # min and max stack params are defined in terms of # of big blinds
            config = get_current_game_config()
            self.min_stack = config["min_stack"]
            self.max_stack = config["max_stack"]
            self.min_bb_ratio = config["min_bb_ratio"]
            self.max_bb_ratio = config["max_bb_ratio"]
            self.min_allowed_start_bb = config["min_allowed_start_bb"]
            self.stop_event = threading.Event()
        except Exception as e:
            traceback.print_exc()
            raise e

    def rescue_ghost_players(self):
        """Periodically checks for players stuck in a playing or training state."""
        current_time = time.time()

        # Only run this check every 5 seconds to save CPU
        if current_time - self.last_timeout_check < 5.0:
            return

        self.last_timeout_check = current_time

        for p_id in self.player_ids:
            # If the player is currently out in the wild...
            if self.is_playing[p_id] or self.is_training[p_id]:
                # ...and they've been gone longer than our threshold...
                if current_time - self.player_dispatch_times[p_id] > self.timeout_threshold:
                    state = "TRAINING" if self.is_training[p_id] else "PLAYING"
                    print(f"👻 [TIMEOUT MONITOR] Rescuing ghost player {p_id} stuck in {state} state!")

                    # 1. Clear their flags
                    self.is_playing[p_id] = False
                    self.is_training[p_id] = False
                    self.is_playing_against[p_id] = []

                    # 2. Put them safely back into the table scheduler pool
                    self.table_scheduler.add(p_id)

                    # 3. Reset their stopwatch
                    self.player_dispatch_times[p_id] = current_time

    def receive_from_trainer_queue(self):
        queue_empty = False
        try:
            message = self.trainer_receive_queue.get_nowait()
        except Empty:
            # queue is empty, we continue with our loop
            queue_empty = True
            player_id, new_weights = None, None
            message = None

        if not queue_empty:
            if message["type"] == "player":
                player_id, new_weights, new_optimizer_params = message["player_id"], message["new_weights"], message["new_optimizer_params"]
                # update that player's model weights
                # TODO: change to get the playerAI directly from the Trainer rather than having to load the player and then put it back. Could also check how much time it actually takes to see if it's worth the implementation effort

                player: PlayerAI = ray.get(self.players[player_id])
                player.load_params(new_weights)
                player.load_optimizers(new_optimizer_params)
                self.players[player_id] = ray.put(player)
                self.player_training_counts[player_id] += 1

                if not self.data_storage.can_train(player_id):
                    # add the player to the table scheduler
                    if not self.is_playing[player_id]:
                        self.table_scheduler.add(player_id)
                    self.is_training[player_id] = False
                else:
                    # They STILL have enough data to train again!
                    # Send them straight back to the training queue.
                    # batch = self.data_storage.get_batch(player_id)
                    # batch_ref = ray.put(batch)
                    batch_ref, num_samples = self.data_storage.get_batch(player_id)
                    trainer_data = {
                        "type": "player",
                        "player_id": player_id,
                        "batch_ref": batch_ref,
                        "num_samples": num_samples,
                        "player_ref": self.players[player_id],
                        "player_training_count": self.player_training_counts[player_id]
                    }
                    self.trainer_send_queue.put_nowait(trainer_data)
                    self.is_training[player_id] = True
                    self.player_dispatch_times[player_id] = time.time()

            elif message["type"] == "termination":
                trainer_id = message["trainer_id"]
                # we get the table with that index
                print(f"Terminating Trainer {trainer_id}")
                trainer_idx = self.trainer_ids.index(trainer_id)
                trainer = self.trainers.pop(trainer_idx)
                self.trainer_ids.pop(trainer_idx)
                ray.kill(trainer)

            elif message["type"] == "creation":
                # we find a suitable table id
                trainer_id = 0
                existing_trainer_ids = set(self.trainer_ids)
                while trainer_id in existing_trainer_ids:
                    trainer_id += 1

                print(f"Creating Trainer {trainer_id}")
                self.trainer_ids.append(trainer_id)
                new_trainer = TrainerActor.remote(trainer_id, self.trainer_send_queue, self.trainer_receive_queue, self.device,
                                    self.discrete, self.log_folder, self.player_save_folder, self.mode)
                self.trainers.append(new_trainer)
                new_trainer.start.remote()

        return queue_empty

    def receive_from_table_queue(self):
        queue_empty = False
        try:
            data = self.table_receive_queue.get_nowait()
        except Empty:
            # queue is empty, we continue with our loop
            queue_empty = True
            data = None

        if not queue_empty:
            # add the data to the data storage
            if data["type"] == "data":
                player_id, table_id = data["player_id"], data["table_id"]

                hand_info, player_winnings = data["hand_info"], data["player_winnings"]
                data_version = data["version"]
                num_samples = data["num_samples"]
                if data_version == self.player_training_counts[player_id]:
                    # Only add data from the same model version as the current one

                    self.data_storage.add(player_id, hand_info, num_samples)

                # send the player_winnings to the leaderboard
                self.leaderboard_queue.put_nowait((player_id, player_winnings, len(self.table_ids), len(self.trainer_ids),
                                                   self.is_playing, self.is_training, self.is_playing_against,
                                                   self.player_dispatch_times, self.table_scheduler.historical_players_used,
                                                   self.historical_sampler.len.remote()))

            elif data["type"] == "player":
                player_id, other_players = data["player_id"], data["other_players"]

                self.table_scheduler.update_weights(player_id, other_players)

                self.is_playing[player_id] = False  # Mark them as free!
                self.is_playing_against[player_id] = []
                if self.data_storage.can_train(player_id):
                    # batch = self.data_storage.get_batch(player_id)
                    # batch_ref = ray.put(batch)

                    batch_ref, num_samples = self.data_storage.get_batch(player_id)

                    trainer_data = {
                        "type": "player",
                        "player_id": player_id,
                        "batch_ref": batch_ref,
                        "num_samples": num_samples,
                        "player_ref": self.players[player_id],
                        "player_training_count": self.player_training_counts[player_id]
                    }
                    self.trainer_send_queue.put_nowait(trainer_data)
                    self.is_training[player_id] = True
                    self.player_dispatch_times[player_id] = time.time()
                else:
                    # send them to play more games
                    self.table_scheduler.add(player_id)

            elif data["type"] == "termination":
                table_id = data["table_id"]
                print(f"Closing Table {table_id}")
                # we get the table with that index
                table_idx = self.table_ids.index(table_id)
                table = self.tables.pop(table_idx)
                self.table_ids.pop(table_idx)
                ray.kill(table)

            elif data["type"] == "creation":
                # we find a suitable table id
                table_id = 0
                existing_table_ids = set(self.table_ids)
                while table_id in existing_table_ids:
                    table_id += 1
                print(f"Creating Table {table_id}")
                self.table_ids.append(table_id)
                new_table = self.TableActor.remote(table_id, self.device, self.table_send_queue, self.table_receive_queue,
                                      self.table_max_size, self.discrete, self.mode, self.batch_size)
                self.tables.append(new_table)
                new_table.start.remote()
            else:
                raise ValueError(f"Unknown message type {data['type']}")

            return queue_empty
        else:
            return queue_empty

    def start_casino(self):
        from src.game_registry import get_current_game_config
        print(f"Casino Starting")
        game_config = get_current_game_config()

        while (not self.stop_event.is_set()):  # keep running the casino forever
            with self.timer.time("Manager_Total_Loop_Time"):

                with self.timer.time("1_Rescue_Ghost_Players"):
                    self.rescue_ghost_players()

                activity_this_loop = False

                with self.timer.time("2_Drain_Trainer_Queue"):
                    while True:
                        queue_empty_1 = self.receive_from_trainer_queue()
                        if queue_empty_1:
                            break
                        activity_this_loop = True

                with self.timer.time("3_Drain_Table_Queue"):
                    while True:
                        queue_empty_2 = self.receive_from_table_queue()
                        if queue_empty_2:
                            break
                        activity_this_loop = True

                with self.timer.time("4_Table_Scheduler"):
                    player_ids = self.table_scheduler.get_table()
                    while player_ids is not None:
                        activity_this_loop = True
                        table_size = len(list(player_ids))

                        table_param_generator = game_config['table_param_generator']
                        small_blind = 1
                        big_blind = random.randint(self.min_bb_ratio, self.max_bb_ratio) * small_blind
                        bb_starting_stacks = random.randint(self.min_stack, self.max_stack)
                        
                        table_params = table_param_generator(
                            table_size=table_size,
                            small_blind=small_blind,
                            big_blind=big_blind,
                            bb_starting_stacks=bb_starting_stacks
                        )

                        data = {
                            "type": "players",
                            "player_ids": player_ids,
                            "player_refs": [self.players[player_id] if player_id >= 0 else None for player_id in player_ids],
                            "player_versions": [self.player_training_counts[p_id] if p_id >= 0 else None for p_id in player_ids],
                            "table_params": table_params
                        }
                        self.table_send_queue.put_nowait(data)

                        for p_id in player_ids:
                            if p_id < 0: continue
                            self.is_playing[p_id] = True
                            self.is_playing_against[p_id] = [player_id for player_id in player_ids if player_id != p_id]
                            self.player_dispatch_times[p_id] = time.time()

                        player_ids = self.table_scheduler.get_table()

                with self.timer.time("5_Sleep_Backoff"):
                    if not activity_this_loop:
                        time.sleep(1e-6)

            self.loop_step += 1
            if self.loop_step % 10000 == 0:
                self.timer.log_to_tensorboard(self.writer, "Manager", self.loop_step)
                self.timer.reset()
                # self.loop_step = 1  # to prevent it from blowing up to the moon

        print("Casino cleaning up and shutting down...")

    def old_start_casino(self):
        from src.game_registry import get_current_game_config
        print(f"Casino Starting")
        game_config = get_current_game_config()

        while (not self.stop_event.is_set()):   # keep running the casino forever
            # casino main loop
            # Step 0: timeout check (safety measure)
            self.rescue_ghost_players()

            # Step 1: Receive from our trainer queue
            queue_empty_1 = self.receive_from_trainer_queue()

            # Step 2: Receive from our table queue
            queue_empty_2 = self.receive_from_table_queue()

            # Step 3: Receive from the scheduler to see if we can spin up new tables
            # player_ids, table_size = self.table_scheduler.get_full_waiting_room()
            player_ids = self.table_scheduler.get_table()
            while player_ids is not None:
                table_size = len(list(player_ids))
                # spin up a table
                table_param_generator = game_config['table_param_generator']
                small_blind = 1
                big_blind = random.randint(self.min_bb_ratio, self.max_bb_ratio) * small_blind
                bb_starting_stacks = random.randint(self.min_stack, self.max_stack)
                
                table_params = table_param_generator(
                    table_size=table_size,
                    small_blind=small_blind,
                    big_blind=big_blind,
                    bb_starting_stacks=bb_starting_stacks
                )
                # gather the player's parameters and send it all
                data = {
                    "type": "players",
                    "player_ids": player_ids,
                    "player_refs": [self.players[player_id] for player_id in player_ids],
                    "player_versions": [self.player_training_counts[p_id] for p_id in player_ids],
                    "table_params": table_params
                }
                self.table_send_queue.put_nowait(data)
                # update player status to playing
                for p_id in player_ids:
                    self.is_playing[p_id] = True
                    self.is_playing_against[p_id] = [player_id for player_id in player_ids if player_id != p_id]
                    self.player_dispatch_times[p_id] = time.time()
                player_ids = self.table_scheduler.get_table()

        print("Casino cleaning up and shutting down...")

    def start(self):
        try:
            self.start_casino()
        except (Exception, KeyboardInterrupt) as e:
            if isinstance(e, KeyboardInterrupt):
                print("Casino terminated")
            else:
                print(f"Casino error: {e}")
                import traceback
                traceback.print_exc()
            return
        finally:
            # tell the casino to shut down
            self.stop_event.set()

            # need to tell the tables to terminate cleanly
            print(f"Telling the tables to close")
            for _ in self.tables:
                self.table_send_queue.put_nowait({
                    "type": "message",
                    "terminate": True
                })

            for table in self.tables:
                ray.kill(table)

            # need to tell the trainers to terminate cleanly
            print(f"Telling the trainers to leave")
            for _ in self.trainers:
                self.trainer_send_queue.put_nowait({
                    "type": "message",
                    "terminate": True
                })

            for trainer in self.trainers:
                ray.kill(trainer)

            # need to tell the leaderboard gui to terminate
            print(f"Closing the leaderboard")
            self.leaderboard.set_done.remote()
            ray.kill(self.leaderboard)

            time.sleep(5)  # giving time for everyone to close
