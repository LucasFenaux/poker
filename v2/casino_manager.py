import random
import threading
import time
from typing import Any

import ray
from ray.util.queue import Queue, Empty
import os
import torch

from global_settings import NUM_PLAYERS, NUM_TABLES, NUM_TRAINERS
from alg import PPO
from trainer_actor import TrainerActor
from table_actor import TableActor
from leaderboard_actor import LeaderboardActor


class PlayerAI:
    def __init__(self, models):
        self.models = models

    def load_params(self, param_dicts):
        for model, param_dict in zip(self.models, param_dicts):
            model.load_state_dict(param_dict)

    def get_params(self):
        params = []
        for model in self.models:
            params.append(model.state_dict())
        return params


# @ray.remote(num_cpus=0)
class DataStorage:
    # def __init__(self, in_queue, out_queue, player_ids, batch_size: int, on_policy):
    def __init__(self, player_ids, batch_size: int, on_policy):
        # self.in_queue = in_queue
        # self.out_queue = out_queue
        self.player_ids = player_ids
        self.batch_size = batch_size
        self.states = {player_id: [] for player_id in self.player_ids}
        self.current_actors = {player_id: [] for player_id in self.player_ids}
        self.rewards = {player_id: [] for player_id in self.player_ids}
        self.actions = {player_id: [] for player_id in self.player_ids}
        self.sample_weights = {player_id: [] for player_id in self.player_ids}
        self.on_policy = on_policy

    def add(self, player_id, hand_info_ref):
        hand_info: dict[str, Any] = ray.get(hand_info_ref)
        self.states[player_id].extend(hand_info["states"])
        self.rewards[player_id].extend(hand_info["rewards"])
        self.current_actors[player_id].extend(hand_info["current_actors"])
        self.actions[player_id].extend(hand_info["actions"])
        self.sample_weights[player_id].extend(hand_info["sample_weights"])
        return self.can_train(player_id)

    def can_train(self, player_id):
        if len(self.states[player_id]) >= self.batch_size:
            return True  # can train

        return False

    def get_batch(self, player_id):
        assert len(self.states[player_id]) >= self.batch_size
        assert len(self.rewards[player_id]) >= self.batch_size
        assert len(self.actions[player_id]) >= self.batch_size
        assert len(self.current_actors[player_id]) >= self.batch_size
        assert len(self.sample_weights[player_id]) >= self.batch_size

        # since we are gonna train on the data, if we have a ONPolicy alg, we need to get rid of the extra

        states = self.states[player_id][:self.batch_size]
        current_actors = self.current_actors[player_id][:self.batch_size]
        rewards = self.rewards[player_id][:self.batch_size]
        actions = self.actions[player_id][:self.batch_size]
        sample_weights = self.sample_weights[player_id][:self.batch_size]

        if self.on_policy:
            self.states[player_id] = []
            self.current_actors[player_id] = []
            self.rewards[player_id] = []
            self.actions[player_id] = []
            self.sample_weights[player_id] = []
        else:
            self.states[player_id] = self.states[player_id][self.batch_size:]
            self.current_actors[player_id] = self.current_actors[player_id][self.batch_size:]
            self.rewards[player_id] = self.rewards[player_id][self.batch_size:]
            self.actions[player_id] = self.actions[player_id][self.batch_size:]
            self.sample_weights[player_id] = self.sample_weights[player_id][self.batch_size:]

        return {
            "states": (states, current_actors),
            "rewards": rewards,
            "actions": actions,
            "sample_weights": sample_weights,
        }

# @ray.remote(num_cpus=0)
class TableScheduler:
    # def __init__(self, in_queue, out_queue, table_min_size: int, table_max_size: int, player_ids):
    def __init__(self, table_min_size: int, table_max_size: int, player_ids):
        # self.in_queue = in_queue
        # self.out_queue = out_queue
        self.player_ids = player_ids
        self.table_max_size = table_max_size
        self.table_min_size = table_min_size
        self.waiting_rooms = [[] for _ in range(NUM_PLAYERS//table_min_size + int(NUM_PLAYERS%table_min_size > 0))]
        print(f"Created {len(self.waiting_rooms)} waiting rooms")
        self.next_table_size = [random.randint(table_min_size, table_max_size)] * len(self.waiting_rooms)
        self.last_table_played_at: dict[int, int] = {player_id: None for player_id in player_ids}
        # TODO: implement a new way to do the waiting room so I don't have only table max size ** 2 players playing

    def all_waiting_rooms_are_full(self):
        for waiting_room, next_table_size in zip(self.waiting_rooms, self.next_table_size):
            if len(waiting_room) < next_table_size:
                return False
        return True

    def add(self, player_id: int, table_id: int = None):
        # if table_id is None, came back from training so no need to update the last table played at
        if table_id is not None:
            self.last_table_played_at[player_id] = table_id

        if self.all_waiting_rooms_are_full():
            # tell the casino to start more tables before trying to add back players into the waiting rooms
            return False

        table_id = self.last_table_played_at[player_id]
        # find which waiting room to put them in
        added_to_a_table = False
        for i, waiting_room in enumerate(self.waiting_rooms):
            found_same_table_player = False
            for other_player_id in waiting_room:
                if self.last_table_played_at[other_player_id] == table_id:
                    found_same_table_player = True
                    break
            if not found_same_table_player and len(waiting_room) < self.next_table_size[i]:
                # we can add them to the waiting room
                waiting_room.append(player_id)
                added_to_a_table = True
                break

        if not added_to_a_table:
            # we just add them to a non-full table for now
            # TODO: do something better than that. Might require a scheduler overhaul
            for i, waiting_room in enumerate(self.waiting_rooms):
                if len(waiting_room) < self.next_table_size[i]:
                    waiting_room.append(player_id)
                    break
        return True

    def get_full_waiting_room(self):
        for i, (waiting_room, next_table_size) in enumerate(zip(self.waiting_rooms, self.next_table_size)):
            if len(waiting_room) > next_table_size:
                raise Exception("WTF HAPPENED HERE")
            elif len(waiting_room) == next_table_size:
                player_ids = self.waiting_rooms.pop(i)
                table_size = self.next_table_size.pop(i)

                self.waiting_rooms.append([])
                self.next_table_size.append(random.randint(self.table_min_size, self.table_max_size))
                return player_ids, table_size

        return None, None


class CasinoManager:
    def __init__(self, device: torch.device, save_folder: str = "./", discrete: bool = False):
        self.player_ids = list(range(NUM_PLAYERS))
        self.device = device
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.player_save_folder = os.path.join(save_folder, "players")
        os.makedirs(self.player_save_folder, exist_ok=True)
        self.log_folder = os.path.join(save_folder, "logs")
        self.mode = "beta"
        self.is_playing = {player_id: False for player_id in self.player_ids}
        # we spin up the player models
        self.players = [ray.put(PlayerAI(PPO.init_networks(torch.device("cpu"), discrete=discrete, mode=self.mode))) for _ in
                        self.player_ids]
        self.player_training_counts = [0] * len(self.player_ids)

        self.table_max_size = 2
        self.table_min_size = 2
        self.batch_size = 5000
        self.on_policy = True

        self.table_scheduler = TableScheduler(self.table_min_size, self.table_max_size, self.player_ids)

        self.table_send_queue = Queue(maxsize=0)
        self.table_receive_queue = Queue(maxsize=0)

        self.trainer_send_queue = Queue(maxsize=0)
        self.trainer_receive_queue = Queue(maxsize=0)

        # max_tables_needed = len(self.player_ids) // self.table_min_size
        print(f"Opening casino with {NUM_TABLES} permanent tables of size between {self.table_min_size} and "
              f"{self.table_max_size}...")
        self.table_ids = [table_id for table_id in range(NUM_TABLES)]
        self.tables = [TableActor.remote(table_id, device, self.table_send_queue, self.table_receive_queue,
                                         self.table_max_size, discrete, self.mode) for table_id in self.table_ids]   # we spin up the tables at the beginning to avoid the churn
        for table in self.tables:
            table.start.remote()

        self.data_storage = DataStorage(self.player_ids, self.batch_size, self.on_policy)

        self.trainer_ids = [trainer_id for trainer_id in range(NUM_TRAINERS)]
        self.trainers = [TrainerActor.remote(i, self.trainer_send_queue, self.trainer_receive_queue, device, discrete,
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

        self.discrete = discrete
        # min and max stack params are defined in terms of # of big blinds
        self.min_stack = 50
        self.max_stack = 500
        self.min_bb_ratio = 1
        self.max_bb_ratio = 5
        self.min_allowed_start_bb = 10
        self.stop_event = threading.Event()
        available = ray.available_resources()
        free_cpus = available.get('CPU', 0)
        assert free_cpus > 0, print(f"Only {free_cpus} CPUs are available whereas {NUM_TRAINERS} are "
                                                f"requested.")

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
                player_id, new_weights = message["player_id"], message["new_weights"]
                # update that player's model weights
                player: PlayerAI = ray.get(self.players[player_id])
                player.load_params(new_weights)
                self.players[player_id] = ray.put(player)
                self.player_training_counts[player_id] += 1

                if not self.data_storage.can_train(player_id):
                    # add the player to the table scheduler
                    if not self.is_playing[player_id]:
                        self.table_scheduler.add(player_id)

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

                if data_version == self.player_training_counts[player_id]:
                    # Only add data from the same model version as the current one

                    self.data_storage.add(player_id, hand_info)

                # send the player_winnings to the leaderboard
                self.leaderboard_queue.put_nowait((player_id, player_winnings, len(self.table_ids), len(self.trainer_ids)))

            elif data["type"] == "player":
                player_id, table_id = data["player_id"], data["table_id"]

                self.is_playing[player_id] = False  # Mark them as free!
                if self.data_storage.can_train(player_id):
                    batch = self.data_storage.get_batch(player_id)
                    batch_ref = ray.put(batch)

                    trainer_data = {
                        "type": "player",
                        "player_id": player_id,
                        "batch_ref": batch_ref,
                        "player_ref": self.players[player_id],
                        "player_training_count": self.player_training_counts[player_id]
                    }
                    self.trainer_send_queue.put_nowait(trainer_data)
                else:
                    # send them to play more games
                    self.table_scheduler.add(player_id, table_id)

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
                new_table = TableActor.remote(table_id, self.device, self.table_send_queue, self.table_receive_queue,
                                      self.table_max_size, self.discrete, self.mode)
                self.tables.append(new_table)
                new_table.start.remote()
            else:
                raise ValueError(f"Unknown message type {data['type']}")

            return queue_empty
        else:
            return queue_empty

    def start_casino(self):
        print(f"Casino Starting")
        # initialize the casino by putting all the players into the table queue
        players_left = [player_id for player_id in self.player_ids]
        num_players_left = len(players_left)
        while num_players_left > 0:
            if num_players_left <= self.table_max_size:
                # last table, we start it and move on
                table_size = num_players_left
                last_table = True
            else:
                table_size = random.randint(self.table_min_size, self.table_max_size)
                if num_players_left - table_size < self.table_min_size:
                    table_size = num_players_left - self.table_min_size
                last_table = False
            # print(num_players_left)
            # pick the players
            if last_table:
                player_ids = players_left
            else:
                player_ids = random.sample(players_left, table_size)

            # update the players left
            players_left = [player for player in players_left if player not in player_ids]
            num_players_left = len(players_left)

            small_blind = 1  # we only deal with relative values anyways
            big_blind = random.randint(self.min_bb_ratio, self.max_bb_ratio) * small_blind
            # starting_stacks = random.randint(max(self.min_stack, big_blind * 10), max(self.max_stack, big_blind * 10))
            bb_starting_stacks = random.randint(self.min_stack, self.max_stack)
            starting_stacks = bb_starting_stacks * big_blind
            table_params = {
                "raw_blinds_or_straddles": (small_blind, big_blind),
                "min_bet": big_blind,
                "raw_starting_stacks": starting_stacks,
                "player_count": table_size
            }

            # gather the player's parameters and send it all
            data = {
                "type": "players",
                "player_ids": player_ids,
                "player_refs": [self.players[player_id] for player_id in player_ids],
                "player_versions": [self.player_training_counts[p_id] for p_id in player_ids],
                "table_params": table_params
            }
            self.table_send_queue.put_nowait(data)
            # update the player statuses
            for p_id in player_ids:
                self.is_playing[p_id] = True

        while (not self.stop_event.is_set()):   # keep running the casino forever
            # casino main loop
            # Step 1: Receive from our trainer queue
            queue_empty_1 = self.receive_from_trainer_queue()

            # Step 2: Receive from our table queue
            queue_empty_2 = self.receive_from_table_queue()

            # Step 3: Receive from the scheduler to see if we can spin up new tables
            player_ids, table_size = self.table_scheduler.get_full_waiting_room()
            while player_ids is not None and table_size is not None:
                # spin up a table
                small_blind = 1
                big_blind = random.randint(self.min_bb_ratio, self.max_bb_ratio) * small_blind
                # starting_stacks = random.randint(max(self.min_stack, big_blind * 10), max(self.max_stack, big_blind * 10))
                bb_starting_stacks = random.randint(self.min_stack, self.max_stack)
                starting_stacks = bb_starting_stacks * big_blind

                table_params = {
                    "raw_blinds_or_straddles": (small_blind, big_blind),
                    "min_bet": big_blind,
                    "raw_starting_stacks": starting_stacks,
                    "player_count": table_size
                }
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

                player_ids, table_size = self.table_scheduler.get_full_waiting_room()

        print("Casino cleaning up and shutting down...")

    def start(self):
        try:
            self.start_casino()
        except KeyboardInterrupt as e:
            print("Casino terminated")
            raise e
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
