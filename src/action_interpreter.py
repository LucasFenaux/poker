from enum import Enum
import torch.nn as nn
from fractions import Fraction
import math

class Action(Enum):
    CHECK_OR_FOLD = 0
    CHECK_OR_CALL = 1
    RAISE = 2
    # ALL_IN = 3

    @classmethod
    def decide_action(cls, action):
        # we assume the action has been passed through a Sigmoid or some other 0,1 bounding function
        # if action < 0.3:
        if action < cls.get_call_threshold():
            return cls.CHECK_OR_FOLD
        # elif action < 0.6:
        elif action < cls.get_raise_threshold():
            return cls.CHECK_OR_CALL
        else:
            return cls.RAISE

    @staticmethod
    def get_raise_threshold():
        return 2/3

    @staticmethod
    def get_call_threshold():
        return 1/3

        # elif action < 0.9:
        #     return cls.RAISE
        # else:
        #     return cls.ALL_IN

def to_exact_fraction(amount: float) -> Fraction:
    return Fraction(str(amount))


class ActionInterpreter(nn.Module):

    def __init__(self, mode="beta"):
        super(ActionInterpreter, self).__init__()
        if mode == "beta":
            self.squashing_fn = nn.Identity()
        elif mode == "normal":
            self.squashing_fn = nn.Sigmoid()

    def forward(self, x, min_bet, max_bet):
        # assume x is a potentially batched tensor whose last dimension is 2
        assert x.shape[-1] == 2
        assert len(x.shape) <= 2

        # def bet_size_scaling(bet):
        #     return to_exact_fraction(bet * (max_bet - min_bet) + min_bet)

        def bet_size_scaling(bet):
            # exponential scaling rather than linear scaling
            safe_min = max(float(min_bet), 1e-5)
            safe_max = max(float(max_bet), safe_min)

            if safe_max <= safe_min:
                return to_exact_fraction(safe_min)

            log_min = math.log(safe_min)
            log_max = math.log(safe_max)

            # Interpolate in log-space
            scaled_log = log_min + bet * (log_max - log_min)
            scaled_bet = math.exp(scaled_log)

            scaled_bet = min(max(scaled_bet, safe_min), safe_max)  # fix floating point issues

            return to_exact_fraction(scaled_bet)
        # we squash both the action and the bet sizing and use the bet sizing as the slider between min and max bet

        if len(x.shape) == 1:
            action = Action.decide_action(self.squashing_fn(x[0]).item())
            bet_sizing = self.squashing_fn(x[1]).item()
            bet_sizing = bet_size_scaling(bet_sizing)

        else:
            action = self.squashing_fn(x[:, 0])
            action = [Action.decide_action(x.item()) for x in action]
            bet_sizing = self.squashing_fn(x[:, 1])
            bet_sizing = [bet_size_scaling(x.item()) for x in bet_sizing]

        return action, bet_sizing

