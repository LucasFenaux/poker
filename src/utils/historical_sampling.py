import random
import math
import ray
from ray.util.queue import Empty
import asyncio

from src.global_settings import (HISTORY_LOG_WIDTH, HISTORY_BURN_IN)
from src.utils.player_ai import PlayerAI, RNNPlayerAI
import glob
import os
import torch
from src.global_settings import IS_RECURRENT
import json


@ray.remote(num_cpus=0)
class HistoricalSampling:
    sampling_types = ["uniform", "loguniform"]

    def __init__(self, player_ids, in_queue, out_queue, player_save_folder, discrete, mode):
        from src.game_registry import get_current_game_config
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
                    player = RNNPlayerAI(
                        self.alg_class.init_networks(device=torch.device("cpu"), discrete=self.discrete,
                                                     mode=self.mode))
                else:
                    player = PlayerAI(self.alg_class.init_networks(device=torch.device("cpu"), discrete=self.discrete,
                                                                   mode=self.mode))

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

        return player_version % (HISTORY_LOG_WIDTH ** (int(math.log(player_version, HISTORY_LOG_WIDTH)))) == 0

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
            if ((not isinstance(self.checkpoints[player_id][0], list) and len(
                    self.checkpoints[player_id]) >= HISTORY_LOG_WIDTH - 1)
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
            player = RNNPlayerAI(
                self.alg_class.init_networks(device=torch.device("cpu"), discrete=self.discrete, mode=self.mode))
        else:
            player = PlayerAI(
                self.alg_class.init_networks(device=torch.device("cpu"), discrete=self.discrete, mode=self.mode))

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