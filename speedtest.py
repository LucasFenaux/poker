import torch.distributions
from pokerkit import NoLimitTexasHoldem, Automation
import copy
import time
import random
from collections import deque

def a():
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
        cards_list = list(state.deck_cards)
        random.shuffle(cards_list)
        state.deck_cards = deque(cards_list)
    print(f"Copy & Shuffle: {time.time()-start_time:.2f} seconds")


def b():
    start_time = time.time()
    dist = torch.distributions.Normal(torch.rand((5000, 2)), torch.rand(5000, 2))
    raw_samples = dist.sample((10,))
    print(raw_samples.shape)
    print(f"Sampling 10 x 5000 x 2: {time.time()-start_time:.2f} seconds")

    start_time = time.time()
    dist = torch.distributions.Normal(torch.rand((5000, 2)), torch.rand(5000, 2))
    raw_samples = dist.sample((1000,))
    print(raw_samples.shape)
    print(f"Sampling 1000 x 5000 x 2: {time.time()-start_time:.2f} seconds")



if __name__ == '__main__':
    b()
