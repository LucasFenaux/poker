import random
import threading
import time
from typing import Any

import ray
from ray.util.queue import Queue, Empty
import os
import torch
import numpy as np

from src.global_settings import NUM_PLAYERS, NUM_TABLES, NUM_TRAINERS, MAX_TABLE_SIZE, RESOURCE_LIMITED
from src.alg import PPO
from src.trainer_actor import TrainerActor
from src.table_actor import TableActor
from src.leaderboard_actor import LeaderboardActor
import math


class PlayerAI:
    def __init__(self, models, optimizer_params=None):
        self.models = models
        self.optimizer_params = optimizer_params

    def load_params(self, param_dicts):
        for model, param_dict in zip(self.models, param_dicts):
            model.load_state_dict(param_dict)

    def get_params(self):
        params = []
        for model in self.models:
            params.append(model.state_dict())
        return params

    def load_optimizers(self, optimizer_params):
        self.optimizer_params = optimizer_params

    def get_optimizer_params(self):
        return self.optimizer_params


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
        # unless we are not resource limited in which case we send everything
        if RESOURCE_LIMITED:
            states = self.states[player_id][:self.batch_size]
            current_actors = self.current_actors[player_id][:self.batch_size]
            rewards = self.rewards[player_id][:self.batch_size]
            actions = self.actions[player_id][:self.batch_size]
            sample_weights = self.sample_weights[player_id][:self.batch_size]
        else:
            states = self.states[player_id]
            current_actors = self.current_actors[player_id]
            rewards = self.rewards[player_id]
            actions = self.actions[player_id]
            sample_weights = self.sample_weights[player_id]

        if self.on_policy or not RESOURCE_LIMITED:
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


class JITTableScheduler:
    def __init__(self, table_min_size: int, table_max_size: int, player_ids):
        self.player_ids = player_ids
        self.table_max_size = table_max_size
        self.table_min_size = table_min_size
        assert table_max_size <= MAX_TABLE_SIZE
        assert self.table_min_size <= self.table_max_size
        # self.max_plans = 10
        self.min_pool = max(table_max_size * 2, int(len(self.player_ids)/10))
        self.weights = {
            player_id: {
                other_player_id: 0 for other_player_id in player_ids if player_id != other_player_id
            } for player_id in player_ids
        }
        self.pool = set(self.player_ids[:])

    def update_weights(self, player_id: int, other_players: list[tuple[int, int]]):
        if other_players is not None:
            player_weights = self.weights[player_id]
            for other_player, other_player_version in other_players:
                if other_player != player_id:
                    player_weights[other_player] += other_player_version

    def add(self, player_id: int):
        """
        Add a player into the scheduler to be scheduled for a game
        :param player_id: Which player we are adding back in.
        :param other_players:  The other players the player was playing with if he was a table and their current version
        :return:
        """
        assert player_id not in self.pool, (f"Player duplicated - {player_id} is already in the pool")
        self.pool.add(player_id)

    def get_table(self):
        if len(self.pool) <= self.min_pool:
            return None

        table = []

        available_players = list(self.pool)
        assert len(available_players) >= self.table_max_size
        table_size = random.randint(self.table_min_size, self.table_max_size)

        starter_idx = random.randint(0, len(available_players) - 1)
        starter = available_players.pop(starter_idx)
        table.append(starter)

        starter_weights = self.weights[starter]
        available_player_weights = []
        # we then get the weights of the starter and pick from the remaining players based on those weights
        for remaining_player in available_players:
            available_player_weights.append(starter_weights[remaining_player])

        # we invert them
        max_weight = max(available_player_weights)
        inverted_weights = [max_weight - remaining_player_weight + 1 for remaining_player_weight in available_player_weights]

        # then we normalize them
        normalized_player_weights = [inverted_weight / sum(inverted_weights) for
                                    inverted_weight in inverted_weights]

        assert math.isclose(sum(normalized_player_weights), 1.0, rel_tol=1e-5)

        # select the followers based on their weight
        followers = np.random.choice(available_players, p=normalized_player_weights, replace=False, size=table_size - 1)

        for follower in followers:
            follower = follower.item()
            available_players.remove(follower)
            table.append(follower)

        for player in table:
            self.pool.remove(player)

        return table


class PlanTableScheduler:
    def __init__(self, table_min_size: int, table_max_size: int, player_ids):
        self.player_ids = player_ids
        self.table_max_size = table_max_size
        self.table_min_size = table_min_size
        assert table_max_size <= MAX_TABLE_SIZE
        assert self.table_min_size <= self.table_max_size
        # self.max_plans = 10
        self.max_plans = np.inf
        self.min_pool = max(table_max_size * 2, int(len(self.player_ids)/10))
        self.plan_count = 0
        self.weights = {
            player_id: {
                other_player_id: 0 for other_player_id in player_ids if player_id != other_player_id
            } for player_id in player_ids
        }
        self.pool = set(self.player_ids[:])
        self.plans: list[list[set]] = []
        # self.was_updated = False
        # self.already_returned_none = False

    def _generate_plan(self):
        """
        We generate a partition of the all the players into tables based on their mutual weights at time of generation.
        :return: a plan, which is a list of disjoint sets such that the union of all those sets is self.player_ids
        """
        available_players = self.player_ids[:]
        plan = []
        while len(available_players) >= self.table_min_size:  # we accept that some players might not get to play a plan
            table = []
            # we pick a random table size
            if len(available_players) < self.table_max_size:
                table_size = len(available_players)
            else:
                table_size = random.randint(self.table_min_size, self.table_max_size)
                if len(available_players) - table_size < self.table_min_size:
                    table_size = self.table_min_size

            # we then pick a random player as the starter of the table
            starter_idx = random.randint(0, len(available_players)-1)
            starter = available_players.pop(starter_idx)
            table.append(starter)

            starter_weights = self.weights[starter]
            available_player_weights = []
            # we then get the weights of the starter and pick from the remaining players based on those weights
            for remaining_player in available_players:
                available_player_weights.append(starter_weights[remaining_player])

            # we invert them
            max_weight = max(available_player_weights)
            inverted_weights = [max_weight - remaining_player_weight + 1 for remaining_player_weight in available_player_weights]

            # then we normalize them
            normalized_player_weights = [inverted_weight / sum(inverted_weights) for
                                        inverted_weight in inverted_weights]

            assert math.isclose(sum(normalized_player_weights), 1.0, rel_tol=1e-5)

            # select the followers based on their weight
            followers = np.random.choice(available_players, p=normalized_player_weights, replace=False, size=table_size-1)

            for follower in followers:
                follower = follower.item()
                available_players.remove(follower)
                table.append(follower)

            plan.append(set(table))
        return plan

    def update_weights(self, player_id: int, other_players: list[tuple[int, int]]):
        if other_players is not None:
            player_weights = self.weights[player_id]
            for other_player, other_player_version in other_players:
                if other_player != player_id:
                    player_weights[other_player] += other_player_version

    def add(self, player_id: int):
        """
        Add a player into the scheduler to be scheduled for a game
        :param player_id: Which player we are adding back in.
        :param other_players:  The other players the player was playing with if he was a table and their current version
        :return:
        """
        assert player_id not in self.pool, (f"Player duplicated - {player_id} is already in the pool")
        self.pool.add(player_id)
        # self.was_updated = True

    def _find_table(self):
        # we get the current plan and check if a table is available with the players in the pool
        for i, plan in enumerate(self.plans):
            for j, potential_table in enumerate(plan):
                potential_table: set
                # we check if it is a possible table
                if potential_table.issubset(self.pool):
                    # we found a suitable table
                    table = plan.pop(j)
                    if len(plan) == 0:
                        self.plans.pop(i)  # we remove the empty list
                    # we update the pool
                    for player in table:
                        self.pool.remove(player)

                    return list(table)
        return None

    def get_table(self):
        if len(self.pool) < self.min_pool:
        # if not self.was_updated and self.already_returned_none:
            # we have not changed since last query and we already verified that no table is available, we lazily return
            return None

        # we get the current plan and check if a table is available with the players in the pool
        table = self._find_table()

        if table is not None:
            return table

        # if we got here, it means no suitable table was found
        # we first check if we already have too many plans
        if len(self.plans) >= self.max_plans:
            # too many plans, can't generate a new one, we wait until players catch up to move forward
            # self.already_returned_none = True
            return None

        # we still have room to create a new plan
        new_plan = self._generate_plan()
        print(f"Newest plan ({self.plan_count}): {new_plan} | {len(self.plans)}")
        self.plan_count += 1
        self.plans.append(new_plan)
        # self.was_updated = True

        # we try to find a table again
        table = self._find_table()
        return table


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
        self.batch_size = 5000 if not RESOURCE_LIMITED else 20000
        self.on_policy = True

        # self.table_scheduler = PlanTableScheduler(self.table_min_size, self.table_max_size, self.player_ids)
        self.table_scheduler = JITTableScheduler(self.table_min_size, self.table_max_size, self.player_ids)

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
        assert free_cpus > 0, (f"Only {free_cpus} CPUs are available whereas {NUM_TRAINERS} are "
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
                player_id, new_weights, new_optimizer_params = message["player_id"], message["new_weights"], message["new_optimizer_params"]
                # update that player's model weights
                player: PlayerAI = ray.get(self.players[player_id])
                player.load_params(new_weights)
                player.load_optimizers(new_optimizer_params)
                self.players[player_id] = ray.put(player)
                self.player_training_counts[player_id] += 1

                if not self.data_storage.can_train(player_id):
                    # add the player to the table scheduler
                    if not self.is_playing[player_id]:
                        self.table_scheduler.add(player_id)
                else:
                    # They STILL have enough data to train again!
                    # Send them straight back to the training queue.
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
                player_id, other_players = data["player_id"], data["other_players"]

                self.table_scheduler.update_weights(player_id, other_players)

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
        # no need to put players in the table queue, the new table scheduler will spin up games as we query it

        # players_left = [player_id for player_id in self.player_ids]
        # num_players_left = len(players_left)
        # while num_players_left > 0:
        #     if num_players_left <= self.table_max_size:
        #         # last table, we start it and move on
        #         table_size = num_players_left
        #         last_table = True
        #     else:
        #         table_size = random.randint(self.table_min_size, self.table_max_size)
        #         if num_players_left - table_size < self.table_min_size:
        #             table_size = num_players_left - self.table_min_size
        #         last_table = False
        #     # print(num_players_left)
        #     # pick the players
        #     if last_table:
        #         player_ids = players_left
        #     else:
        #         player_ids = random.sample(players_left, table_size)
        #
        #     # update the players left
        #     players_left = [player for player in players_left if player not in player_ids]
        #     num_players_left = len(players_left)
        #
        #     small_blind = 1  # we only deal with relative values anyways
        #     big_blind = random.randint(self.min_bb_ratio, self.max_bb_ratio) * small_blind
        #     # starting_stacks = random.randint(max(self.min_stack, big_blind * 10), max(self.max_stack, big_blind * 10))
        #     bb_starting_stacks = random.randint(self.min_stack, self.max_stack)
        #     starting_stacks = bb_starting_stacks * big_blind
        #     table_params = {
        #         "raw_blinds_or_straddles": (small_blind, big_blind),
        #         "min_bet": big_blind,
        #         "raw_starting_stacks": starting_stacks,
        #         "player_count": table_size
        #     }
        #
        #     # gather the player's parameters and send it all
        #     data = {
        #         "type": "players",
        #         "player_ids": player_ids,
        #         "player_refs": [self.players[player_id] for player_id in player_ids],
        #         "player_versions": [self.player_training_counts[p_id] for p_id in player_ids],
        #         "table_params": table_params
        #     }
        #     self.table_send_queue.put_nowait(data)
        #     # update the player statuses
        #     for p_id in player_ids:
        #         self.is_playing[p_id] = True

        while (not self.stop_event.is_set()):   # keep running the casino forever
            # casino main loop
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

                player_ids = self.table_scheduler.get_table()

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
