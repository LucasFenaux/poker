import torch
import torch.nn as nn
from dataclasses import dataclass
from typing import Optional, Union, Dict
import math
from src.global_settings import MAX_TABLE_SIZE

def sign_fn(x):
    return 1 if x >= 0 else -1

def safe_div(a, b, default=0.0, max_val=10000.0):
    if b is None or abs(float(b)) < 1e-5:
        return float(default)
    ratio = float(a) / float(b)
    return max(min(ratio, max_val), -max_val)

def safe_log(x):
    if isinstance(x, torch.Tensor):
        sign = torch.sign(x)
        x = sign * torch.log(x.abs() + 1)
    else:
        sign = sign_fn(x)
        x = sign * math.log(math.fabs(x) + 1)
    return x

def safe_lin_sqrt(x):
    if isinstance(x, torch.Tensor):
        sign = torch.sign(x)
        x = torch.where(x.abs() <= 1, x, sign * torch.sqrt(x.abs()))
    else:
        sign = sign_fn(x)
        if math.fabs(x) > 1:
            x = sign * math.sqrt(math.fabs(x))
    return x

@dataclass(slots=True)
class StateSnapshot:
    hole_cards: str
    board_cards: str
    player_count: int
    blinds_or_straddles: tuple[int, ...]
    bets: list[float]
    stacks: list[float]
    in_hand: list[bool]
    pots: list[float]
    min_bet: Optional[float]
    max_bet: Optional[float]

def extract_state_snapshot(state, current_actor) -> StateSnapshot:
    cards = state.hole_cards[current_actor]
    cards = "".join([repr(card) for card in cards])
    if len(state.board_cards) > 0 and isinstance(state.board_cards[0], list):
        board_cards = state.board_cards[0][:]
    elif len(state.board_cards) > 0:
        raise TypeError(f"Board is single depth all of the time: {state.board_cards}")
    else:
        board_cards = []

    to_add = 5 - len(board_cards)
    board_cards = "".join([repr(card) for card in board_cards] + ["??"] * to_add)

    return StateSnapshot(
        hole_cards=cards,
        board_cards=board_cards,
        player_count=state.player_count,
        blinds_or_straddles=tuple(state.blinds_or_straddles),
        bets=list(state.bets),
        stacks=list(state.stacks),
        in_hand=list(state.statuses),
        pots=[p.amount for p in state.pots] if state.pots else [],
        min_bet=state.min_completion_betting_or_raising_to_amount,
        max_bet=state.max_completion_betting_or_raising_to_amount
    )

class StatePreprocessor:
    rank_mapping = {"2": 0, "3": 1, "4": 2, "5": 3, "6": 4, "7": 5, "8": 6, "9": 7, "T": 8, "J": 9, "Q": 10, "K": 11, "A": 12, "?": 13}
    suit_mapping = {"c": 0, "d": 1, "h": 2, "s": 3, "?": 4}

    def __init__(self, max_num_players=MAX_TABLE_SIZE):
        self.max_num_players = max_num_players

    @classmethod
    def parse_cards(cls, cards: str):
        if not cards:
            return [13], [4]
        assert len(cards) % 2 == 0
        ranks = [cls.rank_mapping[cards[i]] for i in range(len(cards)) if i % 2 == 0]
        suits = [cls.suit_mapping[cards[i]] for i in range(len(cards)) if i % 2 == 1]
        return ranks, suits

class CardEmbedding(nn.Module):
    def __init__(self, rank_dim, suit_dim):
        super().__init__()
        self.rank_embedding = nn.Embedding(14, rank_dim)
        self.suit_embedding = nn.Embedding(5, suit_dim)

    def forward(self, ranks, suits):
        r = self.rank_embedding(ranks)
        s = self.suit_embedding(suits)
        return torch.cat([r, s], dim=-1).flatten(start_dim=-2)

class StateInterpreter(nn.Module):
    def __init__(self, device, rank_dim: int = 16, suit_dim: int = 4):
        super().__init__()
        self.device = device
        self.rank_dim = rank_dim
        self.suit_dim = suit_dim
        self.card_embedding = CardEmbedding(rank_dim, suit_dim)
