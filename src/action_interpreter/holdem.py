import math
import torch.nn as nn
from src.action_interpreter.action_interpreter import Action, to_exact_fraction, ActionInterpreter

class HoldemActionInterpreter(ActionInterpreter):

    def __init__(self, mode="beta"):
        super(HoldemActionInterpreter, self).__init__(mode)
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