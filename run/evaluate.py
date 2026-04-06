import os
import glob
import argparse
import random
import torch
import numpy as np
from scipy import stats
from pokerkit import NoLimitTexasHoldem, Automation, State, calculate_equities, parse_range, Deck, StandardHighHand

# --- Local Project Imports ---
from src.state_interpreter import extract_state_snapshot
from src.action_interpreter import ActionInterpreter, Action
from src.alg import PPO, PPOInferenceWrapper


# --- 1. Baseline Bot Logic ---

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
        if active_opponents == 0:
            return self._try_raise(valid_actions)

        equity = self._calculate_equity(state, active_opponents)
        fair_share = 1.0 / (1 + active_opponents)

        if equity > (fair_share * 1.5):
            return self._try_raise(valid_actions)
        elif equity > (fair_share * 0.85):
            return self._try_call(valid_actions)
        else:
            return self._try_fold(valid_actions)

    def _calculate_equity(self, state: State, active_opponents: int) -> float:
        my_cards_str = "".join([repr(c) for c in state.hole_cards[self.player_index]])
        my_range = parse_range(my_cards_str)
        villain_range = parse_range('22+,A2+,K2+,Q2+,J2+,T2+,92+,82+,72+,62+,52+,42+,32+')
        ranges = (my_range,) + (villain_range,) * active_opponents

        flat_board = []
        if len(state.board_cards) > 0:
            if isinstance(state.board_cards[0], (list, tuple)):
                flat_board = list(state.board_cards[0])
            else:
                flat_board = list(state.board_cards)

        equities = calculate_equities(
            ranges, flat_board, 2, 5, Deck.STANDARD, (StandardHighHand,), sample_count=100
        )
        return equities[0]

    def _try_raise(self, valid_actions: dict) -> tuple[str, float]:
        if "complete_bet_or_raise_to" in valid_actions:
            return "complete_bet_or_raise_to", valid_actions["complete_bet_or_raise_to"][0]
        return self._try_call(valid_actions)

    def _try_call(self, valid_actions: dict) -> tuple[str, float]:
        if "check_or_call" in valid_actions:
            return "check_or_call", valid_actions["check_or_call"]
        return self._try_fold(valid_actions)

    def _try_fold(self, valid_actions: dict) -> tuple[str, float]:
        if "check_or_call" in valid_actions and valid_actions["check_or_call"] == 0:
            return "check_or_call", 0
        return "fold", 0


# --- 2. Evaluation Engine Helpers ---

def get_latest_run_folder(base_path="results"):
    runs = glob.glob(os.path.join(base_path, "run_*"))
    if not runs:
        raise ValueError(f"No run folders found in {base_path}")
    return max(runs, key=os.path.getmtime)


def load_ai_player(model_path, device):
    models = PPO.init_networks(device, mode="beta", discrete=False)
    ai_player = PPOInferenceWrapper(models, discrete=False)
    checkpoint, _ = torch.load(model_path, map_location=device, weights_only=True)
    ai_player.load_params(checkpoint)
    ai_player.to(device)
    return ai_player


# --- 3. REUSABLE GAME LOOP ---

def simulate_eval_games(ai_player, num_games, max_table_size, action_interpreter, baseline_bot, verbose=False):
    """Runs N games for a specific AI and returns the array of profit per hand (in bb)."""
    min_bb_ratio = 1
    max_bb_ratio = 5
    min_stack = 50
    max_stack = 500

    ai_winnings_per_hand_bb = []

    for game in range(num_games):
        if verbose and game % 50 == 0 and game > 0:
            print(f"Played {game}/{num_games} hands...")

        current_table_size = random.randint(2, max_table_size)
        ai_seat = random.randint(0, current_table_size - 1)

        small_blind = 1
        big_blind = random.randint(min_bb_ratio, max_bb_ratio) * small_blind
        bb_starting_stacks = random.randint(min_stack, max_stack)
        starting_chips = bb_starting_stacks * big_blind

        blinds = (small_blind, big_blind)
        min_bet = big_blind
        starting_stacks = [starting_chips] * current_table_size

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
            ante_trimming_status=True,
            raw_antes=0,
            raw_blinds_or_straddles=blinds,
            min_bet=min_bet,
            raw_starting_stacks=starting_stacks,
            player_count=current_table_size
        )

        while state.status:
            actor = state.actor_index

            if actor == ai_seat:
                # --- AI TURN ---
                snapshot = extract_state_snapshot(state, actor)
                with torch.no_grad():
                    action_tensor = ai_player.get_action((snapshot, actor))

                s_min_bet = state.min_completion_betting_or_raising_to_amount or max(state.bets)
                s_max_bet = state.max_completion_betting_or_raising_to_amount or s_min_bet
                interpreted_action, bet_sizing = action_interpreter(action_tensor, s_min_bet, s_max_bet)

                if interpreted_action == Action.CHECK_OR_FOLD:
                    if state.can_check_or_call() and state.checking_or_calling_amount == 0:
                        state.check_or_call()
                    elif state.can_fold():
                        state.fold()
                elif interpreted_action == Action.CHECK_OR_CALL:
                    if state.can_check_or_call():
                        state.check_or_call()
                    elif state.can_fold():
                        state.fold()
                elif interpreted_action == Action.RAISE:
                    if state.can_complete_bet_or_raise_to(bet_sizing):
                        state.complete_bet_or_raise_to(bet_sizing)
                    elif state.can_check_or_call():
                        state.check_or_call()
                    elif state.can_fold():
                        state.fold()
                else:  # ALL_IN
                    all_in_size = state.max_completion_betting_or_raising_to_amount
                    if state.can_complete_bet_or_raise_to(all_in_size):
                        state.complete_bet_or_raise_to(all_in_size)
                    elif state.can_check_or_call():
                        state.check_or_call()
                    elif state.can_fold():
                        state.fold()

            else:
                # --- HEURISTIC BOT TURN ---
                baseline_bot.update_index(actor)
                valid_actions = get_valid_actions_dict(state)
                action_str, amt = baseline_bot.get_action(state, valid_actions)

                if action_str == "fold":
                    state.fold()
                elif action_str == "check_or_call":
                    state.check_or_call()
                elif action_str == "complete_bet_or_raise_to":
                    state.complete_bet_or_raise_to(amt)

        # Record AI Profit safely using float() to avoid Fractional object errors
        ai_profit_chips = float(state.stacks[ai_seat]) - float(starting_stacks[ai_seat])
        ai_profit_bb = ai_profit_chips / float(big_blind)
        ai_winnings_per_hand_bb.append(ai_profit_bb)

    return np.array(ai_winnings_per_hand_bb, dtype=float)


def evaluate_agent(num_games, run_folder, player_id, max_table_size):
    device = torch.device("cpu")
    action_interpreter = ActionInterpreter()
    baseline_bot = FastBaselineBot(player_index=0)

    model_path = os.path.join(run_folder, "players", f"{player_id}.pt")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model weights not found at {model_path}")
    print(f"Loading AI Model from: {model_path}")
    ai_player = load_ai_player(model_path, device)

    print(f"\nStarting {num_games} Evaluation Games (Dynamic Tables: 2 to {max_table_size}-Max)...")

    # Delegate to the shared loop!
    ai_winnings_bb = simulate_eval_games(
        ai_player, num_games, max_table_size, action_interpreter, baseline_bot, verbose=True
    )

    # --- 4. Statistical Analysis ---
    total_profit_bb = np.sum(ai_winnings_bb)
    avg_profit_per_hand_bb = np.mean(ai_winnings_bb)
    win_rate_bb_100 = avg_profit_per_hand_bb * 100

    if np.std(ai_winnings_bb) == 0:
        t_stat, p_value = 0.0, 1.0
    else:
        t_stat, p_value = stats.ttest_1samp(ai_winnings_bb, 0.0)

    print("\n" + "=" * 50)
    print(f"🏆 EVALUATION RESULTS (2 to {max_table_size}-MAX RANDOM TABLES) 🏆")
    print("=" * 50)
    print(f"Total Hands Played:  {num_games}")
    print(f"AI Total Profit:     {total_profit_bb:.2f} Big Blinds")
    print(f"AI Win Rate:         {win_rate_bb_100:.2f} bb/100")
    print("-" * 50)
    print("Statistical Confidence (p-test):")
    print(f"T-Statistic:         {t_stat:.4f}")
    print(f"P-Value:             {p_value:.6f}")

    if p_value < 0.05 and t_stat > 0:
        print("\n✅ RESULT: The AI is statically SIGNIFICANTLY BETTER than the heuristic bots! (Likely not luck)")
    elif p_value < 0.05 and t_stat < 0:
        print("\n❌ RESULT: The AI is statically SIGNIFICANTLY WORSE than the heuristic bots.")
    else:
        print(
            "\n⚖️ RESULT: The difference is NOT statistically significant (p >= 0.05). More training or a larger sample size is needed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description="Evaluate a trained Poker AI against heuristic baselines on dynamic tables.")
    parser.add_argument("--games", type=int, default=200, help="Number of hands to play (default: 200)")
    parser.add_argument("--run_folder", type=str, default=None, help="Path to the specific run folder")
    parser.add_argument("--player_id", type=int, default=0, help="ID of the trained player to evaluate (default: 0)")
    parser.add_argument("--max_table_size", type=int, default=2,
                        help="Maximum number of players at the table (default: 2)")

    args = parser.parse_args()
    target_folder = args.run_folder if args.run_folder else get_latest_run_folder()

    evaluate_agent(args.games, target_folder, args.player_id, args.max_table_size)