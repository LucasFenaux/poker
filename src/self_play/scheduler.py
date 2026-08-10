import random
import os
import math
import numpy as np
import ray
from ray.util.queue import Empty
import torch
import asyncio

from src.global_settings import (MAX_TABLE_SIZE, HISTORY_LOG_WIDTH, USE_HISTORICAL_SAMPLING,
                             HISTORICAL_SAMPLING_RATE, HISTORY_BURN_IN, IS_RECURRENT)
from src.player_ai import PlayerAI, RNNPlayerAI
import glob
import os
import torch
from src.global_settings import IS_RECURRENT
from src.game_registry import get_current_game_config
import json


class JITTableScheduler:
    def __init__(self, table_min_size: int, table_max_size: int, player_ids, historical_sampling_receive_queue):
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
        self.historical_sampling_receive_queue = historical_sampling_receive_queue
        self.historical_players_used = 0

    def update_weights(self, player_id: int, other_players: list[tuple[int, int]]):
        if other_players is not None:
            player_weights = self.weights[player_id]
            for other_player, other_player_version in other_players:
                if other_player != player_id and other_player >= 0:
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

    def _get_history_table(self):
        # we first check if there are any historical checkpoints to assign
        if self.historical_sampling_receive_queue.qsize() < self.table_max_size - 1:
            # default back to the default get table
            return self._get_table()


        table = []

        available_players = list(self.pool)
        assert len(available_players) >= 1
        table_size = random.randint(self.table_min_size, self.table_max_size)

        starter_idx = random.randint(0, len(available_players) - 1)
        starter = available_players.pop(starter_idx)
        table.append(starter)

        for _ in range(table_size - 1):
            self.historical_players_used += 1
            table.append(-self.historical_players_used)  # -id indicates that the table should sample a historical player

        for player in table:
            if player >= 0:
                self.pool.remove(player)

        return table

    def _get_table(self):

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

    def get_table(self):
        if len(self.pool) <= self.min_pool:
            return None
        if USE_HISTORICAL_SAMPLING:
            if random.random() < HISTORICAL_SAMPLING_RATE:
                return self._get_history_table()

        return self._get_table()


@ray.remote(num_cpus=0)
class HistoricalSampling:
    sampling_types = ["uniform", "loguniform"]
    def __init__(self, player_ids, in_queue, out_queue, player_save_folder, discrete, mode):
        self.sampling_mode = "uniform"
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.player_ids = player_ids
        self.player_save_folder = player_save_folder
        self.discrete = discrete
        self.mode = mode
        self.checkpoints = {}
        self.num_checkpoints = 0
        self.alg_class = get_current_game_config()["alg"]
        self.inference_wrapper_class = get_current_game_config()["inference_wrapper"]
        
        counts_path = os.path.join(self.player_save_folder, "sampling_counts.json")
        if os.path.exists(counts_path):
            with open(counts_path, "r") as f:
                self.sampling_counts = json.load(f)
        else:
            self.sampling_counts = {}
        self.total_samples = 0

        for player_id in self.player_ids:
            hist_files = glob.glob(os.path.join(self.player_save_folder, f"{player_id}_*.pt"))
            if hist_files:
                print(f"HistoricalSampling: Resuming {len(hist_files)} checkpoints for player {player_id}")
            # sort files by version to add them in the correct order
            hist_files.sort(key=lambda x: int(os.path.basename(x).split('_')[1].split('.')[0]))
            
            for f in hist_files:
                loaded_data = torch.load(f, map_location=torch.device("cpu"), weights_only=True)
                version = int(os.path.basename(f).split('_')[1].split('.')[0])
                
                if IS_RECURRENT:
                    player = RNNPlayerAI(self.alg_class.init_networks(device=torch.device("cpu"), discrete=self.discrete, mode=self.mode))
                else:
                    player = PlayerAI(self.alg_class.init_networks(device=torch.device("cpu"), discrete=self.discrete, mode=self.mode))
                
                if isinstance(loaded_data, tuple) and len(loaded_data) == 2:
                    player.load_params(loaded_data[0])
                else:
                    player.load_params(loaded_data)
                
                psd_ref = ray.put(player)
                
                if self.sampling_mode == "uniform":
                    self._add_uniform(player_id, version, psd_ref)
                elif self.sampling_mode == "loguniform":
                    self._add_loguniform(player_id, version, psd_ref)
                
                self.num_checkpoints += 1

    @staticmethod
    def should_add(player_version):
        if player_version < HISTORY_BURN_IN:
            return False

        return player_version % (HISTORY_LOG_WIDTH**(int(math.log(player_version, HISTORY_LOG_WIDTH)))) == 0

    def can_sample(self):
        return self.num_checkpoints > 10

    def sample(self):
        if len(list(self.checkpoints.values())) == 0:
            raise AttributeError

        if self.sampling_mode == "uniform":
            player_id, version, psd_ref = self._sample_uniform()
        elif self.sampling_mode == "loguniform":
            player_id, version, psd_ref = self._sample_loguniform()
        else:
            raise NotImplementedError

        # Track sampling counts
        str_p_id = str(player_id)
        str_ver = str(version)
        if str_p_id not in getattr(self, 'sampling_counts', {}):
            if not hasattr(self, 'sampling_counts'):
                self.sampling_counts = {}
            self.sampling_counts[str_p_id] = {}
        
        self.sampling_counts[str_p_id][str_ver] = self.sampling_counts[str_p_id].get(str_ver, 0) + 1
        
        self.total_samples = getattr(self, 'total_samples', 0) + 1
        if self.total_samples % 100 == 0:
            import json
            with open(os.path.join(self.player_save_folder, "sampling_counts.json"), "w") as f:
                json.dump(self.sampling_counts, f, indent=4)

        return {"ref": psd_ref}

    def _sample_loguniform(self):
        cur_keys = list(self.checkpoints.keys())
        player_id = random.choice(cur_keys)
        player_bins = self.checkpoints[player_id]

        selected_bin = random.choice(player_bins)
        version, psd_ref = random.choice(selected_bin)
        return player_id, version, psd_ref

    def _sample_uniform(self):
        cur_keys = list(self.checkpoints.keys())
        player_id = random.choice(cur_keys)
        player_lists = self.checkpoints[player_id]
        sample = random.choice(player_lists)
        while isinstance(sample, list):
            sample = random.choice(sample)

        version, psd_ref = sample
        return player_id, version, psd_ref

    def save(self, player_id, player_state_dicts, player_version):
        torch.save(player_state_dicts, os.path.join(self.player_save_folder, f"{player_id}_{player_version}.pt"))

    def _add_loguniform(self, player_id, version, psd_ref):
        item = (version, psd_ref)
        if player_id not in self.checkpoints:
            self.checkpoints[player_id] = [[item, ], ]
        else:
            last_bin = self.checkpoints[player_id][-1]
            if len(last_bin) >= HISTORY_LOG_WIDTH:
                self.checkpoints[player_id].append([item, ])
            else:
                self.checkpoints[player_id][-1].append(item)

    def _add_uniform(self, player_id, version, psd_ref):
        item = (version, psd_ref)
        # in uniform (exponential decay), we have recursive lists that can contain either items or lists.
        if player_id not in self.checkpoints:
            self.checkpoints[player_id] = [item, ]
        else:
            if ((not isinstance(self.checkpoints[player_id][0], list) and len(self.checkpoints[player_id]) >= HISTORY_LOG_WIDTH - 1)
                    or len(self.checkpoints[player_id]) >= HISTORY_LOG_WIDTH):
                # we are either at the lowest level where there are no lists yet or we take into account the list containing the recursion
                self.checkpoints[player_id] = [self.checkpoints[player_id], item]
            else:
                self.checkpoints[player_id].append(item)

    def add(self, player_id, player_state_dicts, player_version):
        # double check that the player version is valid
        assert self.should_add(player_version)
        # convert the player weights to player AIs
        # first initialize a blank player AI

        if IS_RECURRENT:
            player = RNNPlayerAI(self.alg_class.init_networks(device=torch.device("cpu"), discrete=self.discrete, mode=self.mode))
        else:
            player = PlayerAI(self.alg_class.init_networks(device=torch.device("cpu"), discrete=self.discrete, mode=self.mode))

        player.load_params(player_state_dicts[0])

        psd_ref = ray.put(player)

        if self.sampling_mode == "uniform":
            self._add_uniform(player_id, player_version, psd_ref)
        elif self.sampling_mode == "loguniform":
            self._add_loguniform(player_id, player_version, psd_ref)
        else:
            raise NotImplementedError

        self.save(player_id, player_state_dicts, player_version)
        self.num_checkpoints += 1


    async def start(self):
        while True:
            while self.out_queue.qsize() < self.out_queue.maxsize and self.can_sample():
                self.out_queue.put_nowait(self.sample())

            try:
                data = self.in_queue.get_nowait()
                await asyncio.sleep(0)

            except Empty:
                await asyncio.sleep(0.05)
                continue

            player_id, player_state_dicts, player_version = (data["player_id"], data["player_state_dicts"],
                                                             data["player_version"])

            self.add(player_id, player_state_dicts, player_version)

    def len(self):
        return self.num_checkpoints


# obsolete
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