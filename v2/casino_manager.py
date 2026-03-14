import random
import threading
import time

import ray
from ray.util.queue import Queue, Empty
import os
import torch

from global_settings import NUM_PLAYERS, NUM_TABLES, NUM_TRAINERS
from models import load_model
from trainer_actor import TrainerActor
from table_actor import TableActor
from leaderboard_actor import LeaderboardActor
# from leaderboard_gui import LeaderboardGUI


class DataStorage:
    def __init__(self, player_ids, batch_size: int):
        self.player_ids = player_ids
        self.batch_size = batch_size
        self.states = {player_id: [] for player_id in self.player_ids}
        self.current_actors = {player_id: [] for player_id in self.player_ids}
        self.rewards = {player_id: [] for player_id in self.player_ids}
        self.actions = {player_id: [] for player_id in self.player_ids}

    def add(self, player_id, hand_info):
        self.states[player_id].extend(hand_info["states"])
        self.rewards[player_id].extend(hand_info["rewards"])
        self.current_actors[player_id].extend(hand_info["current_actors"])
        self.actions[player_id].extend(hand_info["actions"])
        if len(self.states[player_id]) >= self.batch_size:
            return True  # can train

        return False

    def get_batch(self, player_id):
        assert len(self.states[player_id]) >= self.batch_size
        assert len(self.rewards[player_id]) >= self.batch_size
        assert len(self.actions[player_id]) >= self.batch_size
        assert len(self.current_actors[player_id]) >= self.batch_size

        states = self.states[player_id][:self.batch_size]
        self.states[player_id] = self.states[player_id][self.batch_size:]

        current_actors = self.current_actors[player_id][:self.batch_size]
        self.current_actors[player_id] = self.current_actors[player_id][self.batch_size:]

        rewards = self.rewards[player_id][:self.batch_size]
        self.rewards[player_id] = self.rewards[player_id][self.batch_size:]

        actions = self.actions[player_id][:self.batch_size]
        self.actions[player_id] = self.actions[player_id][self.batch_size:]

        return {
            "states": (states, current_actors),
            "rewards": rewards,
            "actions": actions
        }


class TableScheduler:
    def __init__(self, table_min_size: int, table_max_size: int, player_ids):
        self.player_ids = player_ids
        self.table_max_size = table_max_size
        self.table_min_size = table_min_size
        # self.waiting_rooms = [[] for _ in range(table_max_size)]
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
        self.leaderboard_queue = Queue(maxsize=0)
        self.leaderboard = LeaderboardActor.remote(self.leaderboard_queue, self.player_ids, save_folder)
        self.leaderboard.start.remote()
        # we spin up the player models
        self.players = [load_model(player_id, device, discrete) for player_id in self.player_ids]

        self.table_max_size = 2
        self.table_min_size = 2
        self.batch_size = 5000
        self.table_scheduler = TableScheduler(self.table_min_size, self.table_max_size, self.player_ids)

        self.table_send_queue = Queue(maxsize=0)
        self.table_receive_queue = Queue(maxsize=0)

        self.trainer_send_queue = Queue(maxsize=0)
        self.trainer_receive_queue = Queue(maxsize=0)

        # max_tables_needed = len(self.player_ids) // self.table_min_size
        print(f"Opening casino with {NUM_TABLES} permanent tables of size between {self.table_min_size} and "
              f"{self.table_max_size}...")
        self.tables = [TableActor.remote(table_id, device, self.table_send_queue, self.table_receive_queue,
                                         self.table_max_size, discrete) for table_id in range(NUM_TABLES)]   # we spin up the tables at the beginning to avoid the churn
        for table in self.tables:
            table.start.remote()
        # check that we have enough cpus to allocate the number of trainers we want
        self.data_storage = DataStorage(self.player_ids, self.batch_size)
        # available = ray.available_resources()
        # free_cpus = available.get('CPU', 0)
        # assert free_cpus >= NUM_TRAINERS, print(f"Only {free_cpus} CPUs are available whereas {NUM_TRAINERS} are "
        #                                         f"requested.")

        self.trainers = [TrainerActor.remote(i, self.trainer_send_queue, self.trainer_receive_queue, device, discrete)
                         for i in range(NUM_TRAINERS)]
        for trainer in self.trainers:
            trainer.start.remote()
        self.min_stack = 50
        self.max_stack = 1000
        self.min_small_blind = 1
        self.max_small_blind = 3
        self.min_bb_ratio = 1
        self.max_bb_ratio = 5
        self.min_allowed_start_bb = 10
        self.leaderboard_gui = None
        self.stop_event = threading.Event()
        available = ray.available_resources()
        free_cpus = available.get('CPU', 0)
        assert free_cpus >= 0, print(f"Only {free_cpus} CPUs are available whereas {NUM_TRAINERS} are "
                                                f"requested.")

    def _start_gui(self):
        self.leaderboard_gui = LeaderboardGUI(self.leaderboard)

    def receive_from_trainer_queue(self):
        queue_empty = False
        try:
            player_id, new_weights = self.trainer_receive_queue.get_nowait()
        except Empty:
            # queue is empty, we continue with our loop
            queue_empty = True
            player_id, new_weights = None, None

        if not queue_empty:
            # update that player's model weights
            self.players[player_id].load_state_dict(new_weights)

            # add the player to the table scheduler
            self.table_scheduler.add(player_id, None)
        return queue_empty, player_id

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
            player_id, table_id = data["player_id"], data["table_id"]
            hand_info, player_winnings = data["hand_info"], data["player_winnings"]

            can_train = self.data_storage.add(player_id, hand_info)

            # send the player_winnings to the leaderboard
            self.leaderboard_queue.put((player_id, player_winnings))

            # add the player to the table scheduler
            if not can_train:
                self.table_scheduler.add(player_id, table_id)
            return queue_empty, can_train, player_id
        else:
            return queue_empty, False, None

    def save_player(self, player_id):
        model = self.players[player_id]
        torch.save(model.state_dict(), os.path.join(self.player_save_folder, f"{player_id}.pt"))

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
                    table_size = self.table_min_size
                last_table = False
            # print(num_players_left)
            # pick the players
            if last_table:
                players = players_left
            else:
                players = random.sample(players_left, table_size)

            # update the players left
            players_left = [player for player in players_left if player not in players]
            num_players_left = len(players_left)
            small_blind = random.randint(self.min_small_blind, self.max_small_blind)
            big_blind = random.randint(self.min_bb_ratio, self.max_bb_ratio) * small_blind
            starting_stacks = random.randint(max(self.min_stack, big_blind * 10), max(self.max_stack, big_blind * 10))

            table_params = {
                "raw_blinds_or_straddles": (small_blind, big_blind),
                "min_bet": big_blind,
                "raw_starting_stacks": starting_stacks,
                "player_count": table_size
            }

            # gather the player's parameters and send it all
            data = {
                "type": "players",
                "player_ids": players,
                "players_params_list": [self.players[player_id].state_dict() for player_id in players],
                "table_params": table_params
            }
            self.table_send_queue.put(data)
        # i = 0
        while (not self.stop_event.is_set()):   # keep running the casino forever
            # if i % 100 == 0:
            #     self.leaderboard.save.remote()
            # casino main loop
            # Step 1: Receive from our trainer queue
            trainer_queue_empty, player_id = self.receive_from_trainer_queue()

            if not trainer_queue_empty:
                assert player_id is not None
                # save the player's parameter whom we updated
                self.save_player(player_id)

            # Step 2: Receive from our table queue
            table_queue_empty, can_train, player_id = self.receive_from_table_queue()

            # Step 3: Send the player we received to the trainer if it has enough data to train
            if not table_queue_empty and can_train:
                data = {
                    "type": "player",
                    "player_id": player_id,
                    "state_dict": self.players[player_id].state_dict(),
                    "data_batch": self.data_storage.get_batch(player_id)
                }
                self.trainer_send_queue.put(data)

            # if trainer_queue_empty and table_queue_empty:
            #     time.sleep(0.01)
            # else:
            #     time.sleep(0.02)

            # Step 4: Check if any of the waiting rooms are full, if so, put them in the table queue
            player_ids, table_size = self.table_scheduler.get_full_waiting_room()
            if not (player_ids is None or table_size is None):
                # players are ready to play
                # data["players_params_list"], data["player_ids"], ** data["table_params"]
                # initialize the table parameters
                small_blind = random.randint(self.min_small_blind, self.max_small_blind)
                big_blind = random.randint(self.min_bb_ratio, self.max_bb_ratio) * small_blind
                starting_stacks = random.randint(max(self.min_stack, big_blind*10) , max(self.max_stack, big_blind*10))

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
                    "players_params_list": [self.players[player_id].state_dict() for player_id in player_ids],
                    "table_params": table_params
                }
                self.table_send_queue.put(data)
            # i += 1
        print("Casino cleaning up and shutting down...")

    def start(self):
        try:
            # we need the casino thread seperate so the gui doesn't block it
            # can't launch the gui inside the thread on Mac
            casino_thread = threading.Thread(target=self.start_casino, daemon=True)
            casino_thread.start()
            while True:
                time.sleep(1)
            # self._start_gui()
            # self.start_casino()
        except KeyboardInterrupt as e:
            print("Casino terminated")
            raise e
        finally:
            # tell the casino to shut down
            self.stop_event.set()

            # need to tell the tables to terminate cleanly
            print(f"Telling the tables to close")
            for _ in self.tables:
                self.table_send_queue.put({
                    "type": "message",
                    "terminate": True
                })

            for table in self.tables:
                ray.get(table)

            # need to tell the trainers to terminate cleanly
            print(f"Telling the trainers to leave")
            for _ in self.trainers:
                self.trainer_send_queue.put({
                    "type": "message",
                    "terminate": True
                })

            for trainer in self.trainers:
                ray.get(trainer)

            # need to tell the leaderboard gui to terminate
            print(f"Closing the leaderboard")
            self.leaderboard.set_done.remote()
            ray.get(self.leaderboard_gui)
