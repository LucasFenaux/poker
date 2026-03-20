from pokerkit import NoLimitTexasHoldem, Automation
import copy
import time
repeats = 100000

start_time = time.time()
for i in range(repeats):
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
        **{
            "ante_trimming_status": True,
            "raw_antes": 0,
            "raw_blinds_or_straddles": (1, 2),
            "min_bet": 2,
            "raw_starting_stacks": 100,
            "player_count": 2,
            "mode": "linear"
        }
    )
print(f"Create: {time.time()-start_time:.2f} seconds")

start_time = time.time()
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
    **{
        "ante_trimming_status": True,
        "raw_antes": 0,
        "raw_blinds_or_straddles": (1, 2),
        "min_bet": 2,
        "raw_starting_stacks": 100,
        "player_count": 2,
        "mode": "linear"
    }
)
for i in range(repeats):
    state = copy.deepcopy(state)
print(f"Copy: {time.time()-start_time:.2f} seconds")