from .state_interpreter import *
from .holdem import HoldemStatePreprocessor, HoldemStateInterpreter
from .kuhn import KuhnStatePreprocessor, KuhnStateInterpreter
from src.global_settings import GAME_TYPE

def get_state_interpreter_class():
    if GAME_TYPE == "KUHN":
        return KuhnStateInterpreter
    elif GAME_TYPE == "HOLDEM":
        return HoldemStateInterpreter
    else:
        raise NotImplementedError


def get_state_preprocessor_class():
    if GAME_TYPE == "KUHN":
        return KuhnStatePreprocessor
    elif GAME_TYPE == "HOLDEM":
        return HoldemStatePreprocessor
    else:
        raise NotImplementedError