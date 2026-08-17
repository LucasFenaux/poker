from .holdem import MARLHoldemTableActor, SPHoldemTableActor
from .kuhn import MARLKuhnTableActor, SPKuhnTableActor
from src.global_settings import GAME_TYPE


def get_table_actor_class():
    if GAME_TYPE == "KUHN":
        return MARLKuhnTableActor
    elif GAME_TYPE == "HOLDEM":
        return MARLHoldemTableActor
    else:
        raise NotImplementedError

def get_sp_table_actor_class():
    if GAME_TYPE == "KUHN":
        return SPKuhnTableActor
    elif GAME_TYPE == "HOLDEM":
        return SPHoldemTableActor
    else:
        raise NotImplementedError