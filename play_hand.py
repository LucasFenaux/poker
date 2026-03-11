from pokerkit import Automation, NoLimitTexasHoldem

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
    True,            # uniform antes?
    0,               # antes
    (1, 2),          # blinds
    2,               # min bet
    (100, 100, 0),      # starting stacks
    3,               # player count
)
print(state.street_index, state.street_count)
# Example progression
state.complete_bet_or_raise_to(6)

# print(state)
state.check_or_call()
print(state.street_index, state.street_count)
# flop
# print(state)

# state.deal_board("2c7hTd")
state.check_or_call()
print(state.street_index, state.street_count)

state.complete_bet_or_raise_to(10)
print(state.street_index, state.street_count)

state.check_or_call()
print(state.street_index, state.street_count)
# turn
state.check_or_call()
print(state.street_index, state.street_count)

state.check_or_call()
print(state.street_index, state.street_count)

# river
state.check_or_call()
print(state.street_index, state.street_count)

state.check_or_call()
print(state.street_index, state.street_count)

print(state.stacks)
