from .holdem import MARLHoldemTableActor
from .kuhn import MARLKuhnTableActor
from src.global_settings import GAME_TYPE


def get_table_actor_class():
    if GAME_TYPE == "KUHN":
        return MARLKuhnTableActor
    elif GAME_TYPE == "HOLDEM":
        return MARLHoldemTableActor
    else:
        raise NotImplementedError