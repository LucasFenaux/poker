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


def c():
    import copy
    import pickle
    import timeit

    # 1. Create a moderately complex data structure to test
    sample_data = [
        {
            'id': i,
            'name': f'Item_number_{i}',
            'values': list(range(50))
        }
        for i in range(1000)
    ]

    # 2. Define our target methods
    def use_deepcopy():
        return copy.deepcopy(sample_data)

    def use_pickle():
        return pickle.loads(pickle.dumps(sample_data, protocol=-1))

    def only_dump_pickle():
        return pickle.dumps(sample_data, protocol=-1)

    # Pre-compute a dumped sample so only_load_pickle doesn't include dump time
    sample = only_dump_pickle()

    def only_load_pickle():
        return pickle.loads(sample)

    # 3. Benchmarking
    runs = 1000

    print("Benchmarking copy.deepcopy()...")
    deepcopy_time = timeit.timeit(use_deepcopy, number=runs)
    print(f"Deepcopy time: {deepcopy_time:.4f} seconds")

    print("\nBenchmarking pickle.loads(pickle.dumps())...")
    pickle_time = timeit.timeit(use_pickle, number=runs)
    print(f"Pickle time:   {pickle_time:.4f} seconds")

    print("\nBenchmarking ONLY pickle.dumps()...")
    dump_time = timeit.timeit(only_dump_pickle, number=runs)
    print(f"Dump time:     {dump_time:.4f} seconds")

    print("\nBenchmarking ONLY pickle.loads()...")
    load_time = timeit.timeit(only_load_pickle, number=runs)
    print(f"Load time:     {load_time:.4f} seconds")

    # 4. Comparisons requested
    print("\n--- Speedup Comparisons ---")

    # Compare Dump vs Load
    if load_time < dump_time:
        print(f"Load is {dump_time / load_time:.2f}x faster than Dump")
    else:
        print(f"Dump is {load_time / dump_time:.2f}x faster than Load")

    # Compare Only Load vs Full Pickle
    print(f"Only Load is {pickle_time / load_time:.2f}x faster than Full Pickle (Dump+Load)")

    # Compare Only Load vs Deepcopy
    print(f"Only Load is {deepcopy_time / load_time:.2f}x faster than Deepcopy")




if __name__ == '__main__':
    c()
