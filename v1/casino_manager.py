from global_settings import NUM_PLAYERS, NUM_GAMES
from player_actor import PlayerActor
from table_actor import TableActor

class CasinoManager:
    def __init__(self):
        self.players = [PlayerActor(i) for i in range(NUM_PLAYERS)]
        self.tables = []

    def _init_table(self, players: list[PlayerActor], params: dict):
        return TableActor(players, params)

    def start(self):
        for i in range(NUM_GAMES):
            # TODO: randomly initialize the size of the tables as well as the number of blinds and the small/big blind ratio
            pass



