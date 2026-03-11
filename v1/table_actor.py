import ray
from pokerkit import NoLimitTexasHoldem, Automation
from player_actor import PlayerActor


# def position_name(player_index, button_index, n_players):
#     offset = (player_index - button_index) % n_players
#
#     table_positions = {
#         2: ["BTN/SB", "BB"],
#         3: ["BTN", "SB", "BB"],
#         4: ["BTN", "SB", "BB", "UTG"],
#         5: ["BTN", "SB", "BB", "UTG", "CO"],
#         6: ["BTN", "SB", "BB", "UTG", "HJ", "CO"],
#         7: ["BTN", "SB", "BB", "UTG", "UTG+1", "HJ", "CO"],
#         8: ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "HJ", "CO"],
#         9: ["BTN", "SB", "BB", "UTG", "UTG+1", "MP", "MP+1", "HJ", "CO"],
#     }
#
#     return table_positions[n_players][offset]


@ray.remote(num_cpus=0.5)
class TableActor:
    default_params = {
        "ante_trimming_status": True,
        "raw_antes": 0,
        "raw_blinds_or_straddles": (1, 2),
        "min_bet": 2,
        "raw_starting_stacks": 100,
        "player_count": 2
    }

    def __init__(self, players: list[PlayerActor], params: dict):
        self.players: list[PlayerActor] = players
        self.params = params
        for param in self.default_params:
            if param not in self.params:
                self.params[param] = self.default_params[param]

        assert len(self.players) == self.params["player_count"]

        starting_stacks = self.params.pop("raw_starting_stacks")
        if isinstance(starting_stacks, int):
            self.stacks = [starting_stacks] * len(self.players)
        else:
            self.stacks = starting_stacks

    def _play_round(self):
        state = NoLimitTexasHoldem.create_state(
            (
                Automation.ANTE_POSTING,
                Automation.BET_COLLECTION,
                Automation.BLIND_OR_STRADDLE_POSTING,
                Automation.CARD_BURNING,
                Automation.HOLE_DEALING,
                Automation.BOARD_DEALING,
                Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                Automation.HAND_KILLING,
                Automation.CHIPS_PUSHING,
                Automation.CHIPS_PULLING,
            ),
            raw_starting_stacks=self.stacks,
            **self.params
        )

        while state.status:
            # find the current player and gather the necessary information
            current_actor = state.actor_index
            player = self.players[current_actor]







        # TODO: need to handle players busting out -> just remove them from player list and delete their stack entry?

        return False

    def play_game(self):
        done = False
        while not done:
            done = self._play_round()

            for player in self.players:
                player.update()

