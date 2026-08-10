from .holdem import HoldemTableActor
from .kuhn import KuhnTableActor
from src.global_settings import GAME_TYPE


def get_table_actor_class():
    if GAME_TYPE == "KUHN":
        return KuhnTableActor
    elif GAME_TYPE == "HOLDEM":
        return HoldemTableActor
    else:
        raise NotImplementedError