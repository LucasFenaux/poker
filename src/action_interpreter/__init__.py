from .action_interpreter import Action, to_exact_fraction
from .holdem import ContinuousHoldemActionInterpreter, DiscreteHoldemActionInterpreter, DiscreteHoldemBetActions
from .kuhn import KuhnActionInterpreter
from src.global_settings import GAME_TYPE, ALG


def get_bet_action_map_class():
    if GAME_TYPE == "HOLDEM" and ALG == "NEURD":
        return DiscreteHoldemActionInterpreter
    else:
        return None


def get_action_interpreter_class():
    if ALG == "PPO":
        if GAME_TYPE == "KUHN":
            return KuhnActionInterpreter
        elif GAME_TYPE == "HOLDEM":
            return ContinuousHoldemActionInterpreter
        else:
            raise NotImplementedError
    elif ALG == "NEURD":
        if GAME_TYPE == "KUHN":
            return KuhnActionInterpreter
        elif GAME_TYPE == "HOLDEM":
            return DiscreteHoldemActionInterpreter
        else:
            raise NotImplementedError
    else:
        raise NotImplementedError