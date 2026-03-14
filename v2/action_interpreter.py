from enum import Enum
import torch.nn as nn
from fractions import Fraction


class Action(Enum):
    CHECK_OR_FOLD = 0
    CHECK_OR_CALL = 1
    RAISE = 2
    ALL_IN = 3

    @classmethod
    def decide_action(cls, action):
        # we assume the action has been passed through a Sigmoid or some other 0,1 bounding function
        if action < -1/2:
            return cls.CHECK_OR_FOLD
        elif action < 0:
            return cls.CHECK_OR_CALL
        elif action < 1/2:
            return cls.RAISE
        else:
            return cls.ALL_IN

def to_exact_fraction(amount: float) -> Fraction:
    return Fraction(str(amount))


class ActionInterpreter(nn.Module):

    def __init__(self):
        super(ActionInterpreter, self).__init__()
        self.squashing_fn = nn.Tanh()

    def forward(self, x, min_bet, max_bet):
        # assume x is a potentially batched tensor whose last dimension is 2
        assert x.shape[-1] == 2
        assert len(x.shape) <= 2

        def bet_size_scaling(bet):
            if bet < 0:
                # we assume it's a min_bet
                return min_bet

            return to_exact_fraction(bet * (max_bet - min_bet) + min_bet)

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

