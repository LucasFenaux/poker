from .action_interpreter import Action, to_exact_fraction
from .holdem import HoldemActionInterpreter
from .kuhn import KuhnActionInterpreter
from src.global_settings import GAME_TYPE


def get_action_interpreter_class():
    if GAME_TYPE == "KUHN":
        return KuhnActionInterpreter
    elif GAME_TYPE == "HOLDEM":
        return HoldemActionInterpreter
    else:
        raise NotImplementedError