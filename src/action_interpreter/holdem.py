import math
import torch.nn as nn
from src.action_interpreter.action_interpreter import Action, to_exact_fraction, ActionInterpreter

class ContinuousHoldemActionInterpreter(ActionInterpreter):

    def __init__(self, mode="beta"):
        super(ContinuousHoldemActionInterpreter, self).__init__(mode)
        if mode == "beta":
            self.squashing_fn = nn.Identity()
        elif mode == "normal":
            self.squashing_fn = nn.Sigmoid()

    def forward(self, x, min_bet, max_bet, pot_size=None):
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


class DiscreteHoldemBetActions:
    MIN_BET = 0
    THIRD_POT = 1
    HALF_POT = 2
    TWO_THIRD_POT = 3
    POT = 4
    ALL_IN = 5
    NUM_BETTING_SIZES = 6

    @classmethod
    def decide_action(cls, action):
        if hasattr(action, "item"):
            val = action.item()
        else:
            val = action
        return cls(int(val))

    @staticmethod
    def compute_bet(action, min_bet, max_bet, pot_size):
        action = Action.decide_action(action)
        if action == DiscreteHoldemBetActions.MIN_BET:
            return min_bet
        elif action == DiscreteHoldemBetActions.THIRD_POT:
            return to_exact_fraction(min(max_bet, max(min_bet, pot_size/3)))
        elif action == DiscreteHoldemBetActions.HALF_POT:
            return to_exact_fraction(min(max_bet, max(min_bet, pot_size/2)))
        elif action == DiscreteHoldemBetActions.TWO_THIRD_POT:
            return to_exact_fraction(min(max_bet, max(min_bet, 2*pot_size/3)))
        elif action == DiscreteHoldemBetActions.TWO_THIRD_POT:
            return to_exact_fraction(min(max_bet, max(min_bet, pot_size)))
        elif action == DiscreteHoldemBetActions.ALL_IN:
            return max_bet
        else:
            raise NotImplementedError


class DiscreteHoldemActionInterpreter(ActionInterpreter):

    def __init__(self, mode="beta"):
        super(DiscreteHoldemActionInterpreter, self).__init__(mode)
        if mode == "beta":
            self.squashing_fn = nn.Identity()
        elif mode == "normal":
            self.squashing_fn = nn.Sigmoid()

    def forward(self, x, min_bet, max_bet, pot_size):
        # assume x is a potentially batched tensor whose last dimension is 2
        assert x.shape[-1] == 2
        assert len(x.shape) <= 2

        if len(x.shape) == 1:
            action = Action.decide_action(x[0])
            bet_sizing = DiscreteHoldemBetActions.compute_bet(x[1], min_bet, max_bet, pot_size)
        else:
            action = [Action.decide_action(v) for v in x[:, 0]]
            bet_sizing = [DiscreteHoldemBetActions.compute_bet(v, min_bet, max_bet, pot_size) for v in x[:, 1]]

        return action, bet_sizing