from enum import Enum
from fractions import Fraction
from abc import ABC, abstractmethod
import torch.nn as nn


class Action(Enum):
    CHECK_OR_FOLD = 0
    CHECK_OR_CALL = 1
    RAISE = 2

    @classmethod
    def decide_action(cls, action):
        if hasattr(action, "item"):
            val = action.item()
        else:
            val = action
        return cls(int(val))

def to_exact_fraction(amount: float) -> Fraction:
    return Fraction(str(amount))


class ActionInterpreter(ABC, nn.Module):
    def __init__(self, mode="beta"):
        super().__init__()
        self.mode = mode

    @abstractmethod
    def forward(self, x, min_bet, max_bet, pot_size):
        pass