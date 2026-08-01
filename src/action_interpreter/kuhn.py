from src.action_interpreter.action_interpreter import Action, to_exact_fraction, ActionInterpreter
from src.global_settings import GAME_TYPE


class KuhnActionInterpreter(ActionInterpreter):
    def forward(self, x, min_bet, max_bet):
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