import random
from pokerkit import State, calculate_equities, parse_range, Deck, StandardHighHand
from src.action_interpreter import Action


def get_valid_actions_dict(state: State) -> dict:
    valid_actions = {}
    if state.can_fold():
        valid_actions["fold"] = 0
    if state.can_check_or_call():
        valid_actions["check_or_call"] = state.checking_or_calling_amount
    if state.can_complete_bet_or_raise_to():
        valid_actions["complete_bet_or_raise_to"] = (
            state.min_completion_betting_or_raising_to_amount,
            state.max_completion_betting_or_raising_to_amount
        )
    return valid_actions


class FastBaselineBot:
    def __init__(self, player_index: int):
        self.player_index = player_index

    def update_index(self, new_index: int):
        self.player_index = new_index

    def get_action(self, state: State, valid_actions: dict) -> tuple[str, float]:
        active_opponents = sum(1 for i, active in enumerate(state.statuses) if i != self.player_index and active)

        # 1. Calculate Equity (Safeguard against 0 opponents just in case)
        equity = self._calculate_equity(state, max(1, active_opponents))

        # 2. Calculate Pot Odds
        # Total pot = previously completed pots + all current active bets on the table
        current_pot = sum(pot.amount for pot in state.pots) + sum(state.bets)

        call_amount = valid_actions.get("check_or_call", 0)

        # If checking is free, our pot odds are 0 (we don't need any equity to continue)
        if call_amount == 0:
            pot_odds = 0.0
        else:
            pot_odds = call_amount / (current_pot + call_amount)

        # 3. Probabilistic Action Selection based on Hand Strength
        rng = random.random()
        can_raise = "complete_bet_or_raise_to" in valid_actions

        # --- STRONG HAND (Solidly beating pot odds) ---
        if equity > pot_odds + 0.15:
            if can_raise and rng < 0.80:  # 80% Value Raise
                return self._try_raise(valid_actions, current_pot, call_amount)
            else:  # 20% Trap (Flat Call)
                return self._try_call(valid_actions)

        # --- MARGINAL / DRAWING HAND (Breaking even or slightly better) ---
        elif equity >= pot_odds:
            if can_raise and rng < 0.15:  # 15% Semi-bluff raise
                return self._try_raise(valid_actions, current_pot, call_amount)
            elif rng < 0.95:  # 80% Call / Continue
                return self._try_call(valid_actions)
            else:  # 5% Fold (occasional tight laydown)
                return self._try_fold(valid_actions)

        # --- WEAK HAND (Equity is worse than Pot Odds) ---
        else:
            # If we can check for free, ALWAYS check. Never fold when it costs nothing!
            if call_amount == 0:
                if can_raise and rng < 0.10:  # 10% stab/bluff when checked to
                    return self._try_raise(valid_actions, current_pot, call_amount)
                return self._try_call(valid_actions)
            else:
                if can_raise and rng < 0.05:  # 5% pure bluff raise facing a bet
                    return self._try_raise(valid_actions, current_pot, call_amount)
                else:  # 95% Fold
                    return self._try_fold(valid_actions)

    def _calculate_equity(self, state: State, active_opponents: int) -> float:
        my_cards_str = "".join([repr(c) for c in state.hole_cards[self.player_index]])
        my_range = parse_range(my_cards_str)
        # Assuming a somewhat tight-aggressive villain range for the baseline
        villain_range = parse_range('22+,A2+,K2+,Q2+,J2+,T2+,92+,82+,72+,62+,52+,42+,32+')
        ranges = (my_range,) + (villain_range,) * active_opponents

        flat_board = []
        if len(state.board_cards) > 0:
            if isinstance(state.board_cards[0], (list, tuple)):
                flat_board = list(state.board_cards[0])
            else:
                flat_board = list(state.board_cards)

        # Evaluating 100 samples is usually enough for a fast heuristic
        equities = calculate_equities(
            ranges, flat_board, 2, 5, Deck.STANDARD, (StandardHighHand,), sample_count=100
        )
        return equities[0]

    def _try_raise(self, valid_actions: dict, current_pot: float, call_amount: float) -> tuple[str, float]:
        if "complete_bet_or_raise_to" in valid_actions:
            min_raise, max_raise = valid_actions["complete_bet_or_raise_to"]

            # Dynamic sizing: pick a random amount between half-pot and 1.5x pot
            # pot_scalar = random.uniform(0.5, 1.5)
            pot_scalar = random.uniform(0.25, 1.5)

            # A standard raise is usually the amount to call + your desired pot fraction
            ideal_raise = call_amount + (current_pot * pot_scalar)

            # Ensure our ideal raise is clamped within legal PokerKit boundaries
            clamped_raise = max(min_raise, min(ideal_raise, max_raise))

            return Action.RAISE, float(clamped_raise)

        # Fallback to call if raising isn't legally allowed
        return self._try_call(valid_actions)

    def _try_call(self, valid_actions: dict) -> tuple[str, float]:
        if "check_or_call" in valid_actions:
            return Action.CHECK_OR_CALL, float(valid_actions["check_or_call"])
        return self._try_fold(valid_actions)

    def _try_fold(self, valid_actions: dict) -> tuple[str, float]:
        if "check_or_call" in valid_actions and valid_actions["check_or_call"] == 0:
            return Action.CHECK_OR_CALL, 0.0
        return Action.CHECK_OR_FOLD, 0.0