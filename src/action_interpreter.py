from enum import Enum
import torch.nn as nn
from fractions import Fraction
import math

# class Action(Enum):
#     CHECK_OR_FOLD = 0
#     CHECK_OR_CALL = 1
#     RAISE = 2
#
#     @classmethod
#     def decide_action(cls, action):
#         # we assume the action has been passed through a Sigmoid or some other 0,1 bounding function
#         if action < cls.get_call_threshold():
#             return cls.CHECK_OR_FOLD
#         elif action < cls.get_raise_threshold():
#             return cls.CHECK_OR_CALL
#         else:
#             return cls.RAISE
#
#     @staticmethod
#     def get_raise_threshold():
#         return 2/3
#
#     @staticmethod
#     def get_call_threshold():
#         return 1/3

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


class HoldemActionInterpreter(nn.Module):

    def __init__(self, mode="beta"):
        super(HoldemActionInterpreter, self).__init__()
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
            action = Action.decide_action(x[0])
            bet_sizing = self.squashing_fn(x[1]).item()
            bet_sizing = bet_size_scaling(bet_sizing)

        else:
            action = [Action.decide_action(v) for v in x[:, 0]]
            bet_sizing = self.squashing_fn(x[:, 1])
            bet_sizing = [bet_size_scaling(v.item()) for v in bet_sizing]

        return action, bet_sizing


class KuhnActionInterpreter(nn.Module):
    def __init__(self, mode="beta"):
        super(KuhnActionInterpreter, self).__init__()
        if mode == "beta":
            self.squashing_fn = nn.Identity()
        elif mode == "normal":
            self.squashing_fn = nn.Sigmoid()

    def forward(self, x, min_bet, max_bet):
        from src.ppo_self_play.global_settings import GAME_TYPE
        expected_size = 1 if GAME_TYPE == "KUHN" else 2
        
        if x.dim() == 0:
            x = x.unsqueeze(0)
            
        assert x.shape[-1] == expected_size
        assert len(x.shape) <= 2
        
        if len(x.shape) == 1:
            val = int(x[0].item())
            action = Action.CHECK_OR_FOLD if val == 0 else Action.RAISE
            bet_sizing = to_exact_fraction(1.0)
        else:
            action = [Action.CHECK_OR_FOLD if int(v.item()) == 0 else Action.RAISE for v in x[:, 0]]
            bet_sizing = [to_exact_fraction(1.0) for _ in x[:, 0]]

        return action, bet_sizing

