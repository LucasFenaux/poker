import torch
from pokerkit import NoLimitTexasHoldem, Automation
from src.state_interpreter import extract_state_snapshot
from src.action_interpreter import ActionInterpreter, Action


class HumanAIPokerManager:
    """
    Manages a heads-up poker game between a human player and an AI model.
    """

    def __init__(self, ai_player_model):
        self.ai_player = ai_player_model
        self.human_seat = 0
        self.ai_seat = 1
        self.action_interpreter = ActionInterpreter()
        self.state = None

        # Game Settings
        self.initial_stack = 200.0
        self.sb_amount = 1.0
        self.bb_amount = 2.0
        self.always_show_ai_cards = True
        self.display_in_bb = True

        # Track persistent stacks and history
        self.human_stack = 200.0
        self.ai_stack = 200.0
        self.start_of_hand_human_stack = 200.0
        self.start_of_hand_ai_stack = 200.0

        # NEW: Cache the dealt cards so PokerKit can't erase them when mucking
        self.dealt_human_cards = []
        self.dealt_ai_cards = []

        self.last_pot = 0.0
        self.hand_count = 0
        self.last_message = "Game Started"

    def set_game_params(self, starting_chips, sb, bb, disable_mucking, display_in_bb):
        self.initial_stack = float(starting_chips)
        self.human_stack = float(starting_chips)
        self.ai_stack = float(starting_chips)
        self.sb_amount = float(sb)
        self.bb_amount = float(bb)
        self.always_show_ai_cards = disable_mucking
        self.display_in_bb = display_in_bb
        self.hand_count = 0

    def start_new_hand(self):
        self.hand_count += 1

        if self.hand_count % 2 == 1:
            self.human_seat = 0
            self.ai_seat = 1
            dealer_str = "You are the Dealer (SB)"
        else:
            self.human_seat = 1
            self.ai_seat = 0
            dealer_str = "AI is the Dealer (SB)"

        if self.human_stack <= 0: self.human_stack = self.initial_stack
        if self.ai_stack <= 0: self.ai_stack = self.initial_stack

        self.start_of_hand_human_stack = self.human_stack
        self.start_of_hand_ai_stack = self.ai_stack
        self.last_pot = 0.0

        stacks = [0.0, 0.0]
        stacks[self.human_seat] = self.human_stack
        stacks[self.ai_seat] = self.ai_stack

        self.state = NoLimitTexasHoldem.create_state(
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
            ante_trimming_status=True,
            raw_antes=0,
            raw_blinds_or_straddles=(self.sb_amount, self.bb_amount),
            min_bet=self.bb_amount,
            raw_starting_stacks=stacks,
            player_count=2
        )

        # NEW: Snapshot the dealt cards immediately!
        if self.state.hole_cards:
            self.dealt_human_cards = [repr(c) for c in self.state.hole_cards[self.human_seat]]
            self.dealt_ai_cards = [repr(c) for c in self.state.hole_cards[self.ai_seat]]
        else:
            self.dealt_human_cards = []
            self.dealt_ai_cards = []

        self.last_message = f"Hand {self.hand_count} Started. {dealer_str}"
        self._check_and_play_ai_turn()

    def _apply_action(self, action_name, amount=None):
        pre_pot = float(sum(p.amount for p in self.state.pots) + sum(self.state.bets))
        amt_to_call = float(self.state.checking_or_calling_amount) if self.state.can_check_or_call() else 0.0
        actor = self.state.actor_index
        my_prev_bet = float(self.state.bets[actor]) if actor < len(self.state.bets) else 0.0

        if action_name == 'FOLD':
            self.state.fold()
            self.last_pot = pre_pot
        elif action_name == 'CHECK_CALL':
            self.state.check_or_call()
            self.last_pot = pre_pot + amt_to_call
        elif action_name == 'RAISE':
            self.state.complete_bet_or_raise_to(amount)
            self.last_pot = pre_pot + (float(amount) - my_prev_bet)

    def _fmt(self, raw_amount):
        """Helper to format text logs into BBs or Chips based on user preference."""
        if self.display_in_bb:
            return f"{raw_amount / self.bb_amount:g} BB"
        return f"{raw_amount:g}"

    def get_ui_state(self):
        if not self.state:
            return None

        flat_board = []
        for c in self.state.board_cards:
            if isinstance(c, (list, tuple)):
                flat_board.extend(c)
            else:
                flat_board.append(c)

        # Draw from our snapshot so human cards don't disappear if you fold
        human_cards = self.dealt_human_cards
        ai_cards = []

        is_over = not self.state.status
        winner_message = ""
        current_pot = float(sum(p.amount for p in self.state.pots) + sum(self.state.bets))

        if is_over:
            self.human_stack = float(self.state.stacks[self.human_seat])
            self.ai_stack = float(self.state.stacks[self.ai_seat])

            human_diff = self.human_stack - self.start_of_hand_human_stack
            ai_diff = self.ai_stack - self.start_of_hand_ai_stack

            if human_diff > 0.01:
                winner_message = f"🏆 You won {self._fmt(human_diff)}!"
            elif ai_diff > 0.01:
                winner_message = f"🤖 AI won {self._fmt(ai_diff)}!"
            else:
                winner_message = "🤝 Chopped Pot!"

            went_to_showdown = self.state.statuses[self.human_seat] and self.state.statuses[self.ai_seat]

            # Use our snapshot of the AI's cards instead of asking PokerKit
            if self.always_show_ai_cards or went_to_showdown:
                ai_cards = self.dealt_ai_cards

        current_actor_str = "Hand Over" if is_over else ("Human" if self.state.actor_index == self.human_seat else "AI")
        min_bet = float(
            self.state.min_completion_betting_or_raising_to_amount) if self.state.min_completion_betting_or_raising_to_amount else 0.0
        max_bet = float(
            self.state.max_completion_betting_or_raising_to_amount) if self.state.max_completion_betting_or_raising_to_amount else 0.0
        call_amt = float(self.state.checking_or_calling_amount) if self.state.can_check_or_call() else 0.0

        h_bet = float(self.state.bets[self.human_seat]) if len(self.state.bets) > self.human_seat else 0.0
        a_bet = float(self.state.bets[self.ai_seat]) if len(self.state.bets) > self.ai_seat else 0.0

        return {
            "is_hand_over": is_over,
            "board": [repr(c) for c in flat_board],
            "human_cards": human_cards,
            "ai_cards": ai_cards,
            "human_stack": float(self.state.stacks[self.human_seat]),
            "ai_stack": float(self.state.stacks[self.ai_seat]),
            "human_street_bet": h_bet,
            "ai_street_bet": a_bet,
            "pot": self.last_pot if is_over else current_pot,
            "current_actor": current_actor_str,
            "min_bet": min_bet,
            "max_bet": max_bet,
            "can_check": self.state.can_check_or_call() and self.state.checking_or_calling_amount == 0,
            "amount_to_call": call_amt,
            "last_message": self.last_message,
            "winner_message": winner_message,
            "bb_amount": self.bb_amount,
            "display_in_bb": self.display_in_bb
        }

    def process_human_action(self, action_type: str, bet_amount: float = 0.0):
        if self.state.actor_index != self.human_seat or not self.state.status:
            return False

        if action_type == 'FOLD' and self.state.can_fold():
            self._apply_action('FOLD')
            self.last_message = "You Folded."
        elif action_type == 'CHECK_CALL' and self.state.can_check_or_call():
            amt = float(self.state.checking_or_calling_amount)
            self._apply_action('CHECK_CALL')
            self.last_message = "You Checked." if amt == 0 else "You Called."
        elif action_type == 'RAISE':
            rounded_bet = round(float(bet_amount), 1)
            self._apply_action('RAISE', rounded_bet)
            self.last_message = f"You Raised to {self._fmt(rounded_bet)}."
        else:
            return False

        self._check_and_play_ai_turn()
        return True

    def _check_and_play_ai_turn(self):
        while self.state.status and self.state.actor_index == self.ai_seat:
            actor_idx = self.state.actor_index
            snapshot = extract_state_snapshot(self.state, actor_idx)

            with torch.no_grad():
                action_tensor = self.ai_player.get_action((snapshot, actor_idx))

            s_min_bet = self.state.min_completion_betting_or_raising_to_amount or max(self.state.bets)
            s_max_bet = self.state.max_completion_betting_or_raising_to_amount or s_min_bet

            interpreted_action, bet_sizing = self.action_interpreter(action_tensor, s_min_bet, s_max_bet)

            executed_action_str = ""

            if interpreted_action == Action.CHECK_OR_FOLD:
                if self.state.can_check_or_call() and self.state.checking_or_calling_amount == 0:
                    self._apply_action('CHECK_CALL')
                    executed_action_str = "AI Checked."
                elif self.state.can_fold():
                    self._apply_action('FOLD')
                    executed_action_str = "AI Folded."

            elif interpreted_action == Action.CHECK_OR_CALL:
                if self.state.can_check_or_call():
                    amt = float(self.state.checking_or_calling_amount)
                    self._apply_action('CHECK_CALL')
                    executed_action_str = "AI Checked." if amt == 0 else "AI Called."
                elif self.state.can_fold():
                    self._apply_action('FOLD')
                    executed_action_str = "AI Folded."

            elif interpreted_action == Action.RAISE:
                legal_min = self.state.min_completion_betting_or_raising_to_amount
                legal_max = self.state.max_completion_betting_or_raising_to_amount
                if legal_min is not None and legal_max is not None:
                    clamped_bet = max(float(legal_min), min(float(bet_sizing), float(legal_max)))
                    rounded_bet = round(clamped_bet, 1)
                    self._apply_action('RAISE', rounded_bet)
                    executed_action_str = f"AI Raised to {self._fmt(rounded_bet)}."
                elif self.state.can_check_or_call():
                    amt = float(self.state.checking_or_calling_amount)
                    self._apply_action('CHECK_CALL')
                    executed_action_str = "AI Checked." if amt == 0 else "AI Called."
                elif self.state.can_fold():
                    self._apply_action('FOLD')
                    executed_action_str = "AI Folded."

            self.last_message = executed_action_str