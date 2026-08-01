from typing import Union
from ray.util.queue import Queue

from src.player_ai import PlayerAI, RNNPlayerAI
from abc import ABC, abstractmethod


class BaseTable(ABC):
    default_params = {
        "ante_trimming_status": True,
        "raw_antes": 0,
        "raw_blinds_or_straddles": (1, 2),
        "min_bet": 2,
        "raw_starting_stacks": 200,  # 200 for 100 BB
        "player_count": 2,
        "mode": "tree"
    }
    @abstractmethod
    def __init__(self, table_id, device, in_queue: Queue, out_queue: Queue,
                 historical_sampling_receive_queue: Queue,
                 max_table_size: int, discrete: bool,
                 model_mode: str, batch_size: int, log_folder: str):
        pass

    @abstractmethod
    def reset(self, players: list[Union[PlayerAI, RNNPlayerAI]], player_ids, **table_params):
       pass

    @abstractmethod
    def play_game(self):
        pass

    @abstractmethod
    def start(self):
        pass
