import threading
import time
import traceback

import ray
from src.table_actor import get_sp_table_actor_class
from ray.util.queue import Queue, Empty
import os
import torch

from src.global_settings import NUM_TABLES, RESOURCE_LIMITED, IS_RECURRENT, ALG, GAME_TYPE, HISTORY_WIDTH
from src.self_play.trainer import Trainer
from src.utils.player_ai import PlayerAI, RNNPlayerAI
from torch.utils.tensorboard import SummaryWriter
from src.utils.shared import SemanticTimer
from src.utils.historical_sampling import HistoricalSampling
from src.utils.data_storage import DataStorage


PLAYER_ID = 0


class CasinoManager:
    def __init__(self, device: torch.device, save_folder: str = "./", resume: bool = False):
        from src.game_registry import get_current_game_config
        try:
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
            self.timeout_threshold = 3600  # 1 hour (adjust based on how long a normal game/training takes)
            self.last_timeout_check = time.time()
            self.alg_class = get_current_game_config()["alg"]
            self.inference_wrapper_class = get_current_game_config()["inference_wrapper"]
            # we spin up the player models
            if IS_RECURRENT:
                self.player = RNNPlayerAI(self.alg_class.init_networks(torch.device("cpu"), discrete=self.discrete, mode=self.mode))
            else:
                self.player = PlayerAI(self.alg_class.init_networks(torch.device("cpu"), discrete=self.discrete, mode=self.mode))
            self.player_training_count = 0

            if resume:
                print(f"Resuming model from {self.player_save_folder}")
                model_path = os.path.join(self.player_save_folder, f"{PLAYER_ID}.pt")
                if os.path.exists(model_path):
                    loaded_data = torch.load(model_path, map_location=torch.device("cpu"), weights_only=True)
                    if isinstance(loaded_data, tuple) and len(loaded_data) == 3:
                        new_weights, new_optimizer_params, count = loaded_data
                        self.player.load_params(new_weights)
                        self.player.load_optimizers(new_optimizer_params)
                        self.player_training_count = count
                    elif isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                        new_weights, new_optimizer_params = loaded_data
                        self.player.load_params(new_weights)
                        self.player.load_optimizers(new_optimizer_params)
                    else:
                        self.player.load_params(loaded_data)

            if resume:
                import glob
                # Verify if any counts were NOT loaded from .pt files (older checkpoints fallback)
                if self.player_training_count == 0:
                    hist_files = glob.glob(os.path.join(self.historical_save_folder, f"{PLAYER_ID}_*.pt"))
                    if hist_files:
                        versions = [int(os.path.basename(f).split('_')[1].split('.')[0]) for f in hist_files]
                        self.player_training_count = max(versions)

            self.inference_wrapper = self.inference_wrapper_class(self.player.models, discrete=self.discrete)

            self.table_max_size = 2
            self.table_min_size = 2
            if IS_RECURRENT:
                # number of games
                # self.batch_size = 1_000 if RESOURCE_LIMITED else 8_000
                self.batch_size = 250 if RESOURCE_LIMITED else 2_000   # kuhn poker has ~2 transitions per game
            else:
                # number of transitions
                if GAME_TYPE == "KUHN":
                    self.batch_size = 500 if RESOURCE_LIMITED else 4_000
                else:
                    self.batch_size = 5_000 if RESOURCE_LIMITED else 40_000
            self.on_policy = True

            self.data_storage = DataStorage([PLAYER_ID, ], self.batch_size, self.log_folder)

            self.historical_sampling_queue_len = 100
            self.historical_sampling_send_queue = Queue(maxsize=0)
            self.historical_sampling_receive_queue = Queue(maxsize=self.historical_sampling_queue_len)
            self.historical_sampler = HistoricalSampling.remote([PLAYER_ID,], self.historical_sampling_send_queue,
                                                         self.historical_sampling_receive_queue, self.historical_save_folder,
                                                         self.discrete, self.mode)
            self.historical_sampler.start.remote()

            self.table_receive_queue = Queue(maxsize=0)
            self.table_send_queue = Queue(maxsize=0)

            # max_tables_needed = len(self.player_ids) // self.table_min_size
            print(f"Opening casino with {NUM_TABLES} permanent tables of size between {self.table_min_size} and "
                  f"{self.table_max_size}...")
            self.table_ids = [table_id for table_id in range(NUM_TABLES)]
            self.TableActor = get_sp_table_actor_class()
            self.tables = [self.TableActor.remote(table_id, device, self.table_send_queue, self.table_receive_queue,
                                             self.historical_sampling_receive_queue,
                                             self.table_max_size, self.discrete, self.mode,
                                             self.batch_size, self.log_folder, self.player) for table_id in self.table_ids]   # we spin up the tables at the beginning to avoid the churn

            self.active_tasks = [table.start.remote() for table in self.tables]

            self.trainer = Trainer(self.historical_sampling_send_queue, device, self.discrete,
                                                 self.log_folder, self.player_save_folder, self.mode)  # in self_play we only have one trainer

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

                hand_info, player_winnings = data["hand_info"], data["player_winnings"]
                num_samples = data["num_samples"]
                self.data_storage.add(PLAYER_ID, hand_info, num_samples)

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
                                              self.historical_sampling_receive_queue,
                                              self.table_max_size, self.discrete, self.mode,
                                              self.batch_size, self.log_folder, self.player)
                self.tables.append(new_table)
                new_table.start.remote()
            else:
                raise ValueError(f"Unknown message type {data['type']}")

            return queue_empty
        else:
            return queue_empty


    def can_train(self):
        return self.data_storage.can_train(PLAYER_ID)

    def train(self):
        if not self.can_train():
            return

        # first we stop the tables
        stop_requests = [table.stop.remote() for table in self.tables]
        ray.get(stop_requests)  # make sure all the stop requests went through
        ray.get(self.active_tasks)

        # we perform the model update
        batch_ref, num_samples = self.data_storage.get_batch(PLAYER_ID)
        batch = ray.get(batch_ref)
        new_model_params, new_optim_params, self.player_training_count = self.trainer.start(PLAYER_ID, self.player, num_samples, batch, self.player_training_count)
        self.player.load_params(new_model_params)
        self.player.load_optimizers(new_optim_params)
        new_model_params_ref = ray.put(new_model_params)
        new_optim_params_ref = ray.put(new_optim_params)
        if self.player_training_count % HISTORY_WIDTH == 0:
            print(f"### Player trained {self.player_training_count} times ###")

        # send the new weights to the tables
        update_requests = [table.update_table.remote(new_model_params_ref, new_optim_params_ref) for table in self.tables]
        ray.get(update_requests)

        # clear the existing queue
        while not self.table_receive_queue.empty():
            self.table_receive_queue.get()

        # final, resume the tables with the new weights
        self.active_tasks = [table.start.remote() for table in self.tables]

    def start_casino(self):
        print(f"Casino Starting")

        while (not self.stop_event.is_set()):  # keep running the casino forever
            with self.timer.time("Manager_Total_Loop_Time"):

                activity_this_loop = False

                with self.timer.time("1_Drain_Table_Queue"):
                    while True:
                        queue_empty_2 = self.receive_from_table_queue()
                        if queue_empty_2:
                            break
                        activity_this_loop = True

                if self.can_train():
                    with self.timer.time("2_Train"):
                        self.train()

                with self.timer.time("5_Sleep_Backoff"):
                    if not activity_this_loop:
                        time.sleep(1e-6)


            self.loop_step += 1
            if self.loop_step % 10000 == 0:
                self.timer.log_to_tensorboard(self.writer, "Manager", self.loop_step)
                self.timer.reset()
                # self.loop_step = 1  # to prevent it from blowing up to the moon

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

            for table in self.tables:
                ray.kill(table)

            time.sleep(5)  # giving time for everyone to close
