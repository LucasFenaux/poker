import os
import argparse
import random
import torch
import numpy as np
from scipy import stats
from pokerkit import NoLimitTexasHoldem, Automation

# --- Local Project Imports ---
from src.state_interpreter import extract_state_snapshot
from src.action_interpreter import ActionInterpreter
from src.ppo_self_play.alg import RNNPPO, RNNPPOInferenceWrapper
from src.ppo_self_play.global_settings import IS_RECURRENT
from evaluate import FastBaselineBot, get_latest_run_folder, get_valid_actions_dict


def load_rnn_player(model_path, device):
    """Ensure we strictly load the recurrent models for this test."""
    if not IS_RECURRENT:
        raise ValueError("IS_RECURRENT must be True to test memory impact.")

    models = RNNPPO.init_networks(device, mode="beta", discrete=False)
    ai_player = RNNPPOInferenceWrapper(models, discrete=False)

    loaded_data = torch.load(model_path, map_location=device, weights_only=True)
    checkpoint = loaded_data[0] if isinstance(loaded_data, tuple) and len(loaded_data) == 2 else loaded_data

    ai_player.load_params(checkpoint)
    ai_player.to(device)
    return ai_player


def simulate_memory_test_games(ai_player, num_matches, hands_per_match, action_interpreter, baseline_bot,
                               amnesia_mode=False, track_divergence=False):
    """
    Plays M matches of N hands.
    GAME memory persists across the match. HAND memory resets every hand.
    """
    min_stack, max_stack = 100, 100
    ai_winnings_per_hand_bb = []

    # Divergence Tracking Metrics
    total_decisions = 0
    diverged_actions = 0
    l2_distances = []

    for match in range(num_matches):
        table_size = 2

        # --- 1. INITIALIZE GAME MEMORY (Once per match) ---
        if not amnesia_mode:
            game_hidden = ai_player.init_game_memory(batch_size=1)
        else:
            game_hidden = None

        for hand in range(hands_per_match):
            # Keep seats consistent across the match so game memory tracks the right opponent
            ai_seat = 0
            baseline_bot.update_index(1)

            # big_blind = random.randint(1, 5)
            big_blind = 2
            starting_chips = random.randint(min_stack, max_stack) * big_blind
            starting_stacks = [starting_chips] * table_size

            state = NoLimitTexasHoldem.create_state(
                (
                    Automation.ANTE_POSTING, Automation.BET_COLLECTION, Automation.BLIND_OR_STRADDLE_POSTING,
                    Automation.CARD_BURNING, Automation.HOLE_DEALING, Automation.BOARD_DEALING,
                    Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
                    Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
                ),
                ante_trimming_status=True, raw_antes=0, raw_blinds_or_straddles=(1, big_blind),
                min_bet=big_blind, raw_starting_stacks=starting_stacks, player_count=table_size
            )

            # --- 2. INITIALIZE HAND MEMORY (Once per hand) ---
            if not amnesia_mode:
                hand_hidden = ai_player.init_hand_memory(batch_size=1)
            else:
                hand_hidden = None

            while state.status:
                actor = state.actor_index

                if actor == ai_seat:
                    snapshot = extract_state_snapshot(state, actor)
                    s_min = state.min_completion_betting_or_raising_to_amount or max(state.bets)
                    s_max = state.max_completion_betting_or_raising_to_amount or s_min

                    with torch.no_grad():
                        # If Amnesia Mode, wipe everything at every step
                        if amnesia_mode:
                            hand_hidden = ai_player.init_hand_memory(batch_size=1)
                            game_hidden = ai_player.init_game_memory(batch_size=1)

                        # Shadow Pass for Divergence Tracking
                        if track_divergence and not amnesia_mode:
                            total_decisions += 1
                            zero_h = ai_player.init_hand_memory(batch_size=1)
                            zero_g = ai_player.init_game_memory(batch_size=1)

                            shadow_action_tensor, _ = ai_player.get_action((snapshot, actor), hand_hidden=zero_h,
                                                                           game_hidden=zero_g)
                            shadow_action_tensor = shadow_action_tensor.squeeze(0)
                            shadow_interp, _ = action_interpreter(shadow_action_tensor, s_min, s_max)

                        # True Forward Pass
                        action_tensor, hand_hidden = ai_player.get_action(
                            (snapshot, actor), hand_hidden=hand_hidden, game_hidden=game_hidden
                        )
                        action_tensor = action_tensor.squeeze(0)
                        interpreted_action, bet_sizing = action_interpreter(action_tensor, s_min, s_max)

                        # Compare True vs Shadow
                        if track_divergence and not amnesia_mode:
                            l2_dist = torch.nn.functional.mse_loss(action_tensor, shadow_action_tensor).item()
                            l2_distances.append(l2_dist)
                            if interpreted_action != shadow_interp:
                                diverged_actions += 1

                    # Execute action
                    if interpreted_action.name in ["CHECK_OR_FOLD", "CHECK_OR_CALL"]:
                        if state.can_check_or_call():
                            state.check_or_call()
                        else:
                            state.fold()
                    else:
                        target_size = bet_sizing if interpreted_action.name == "RAISE" else s_max
                        if state.can_complete_bet_or_raise_to(target_size):
                            state.complete_bet_or_raise_to(target_size)
                        elif state.can_check_or_call():
                            state.check_or_call()
                        else:
                            state.fold()
                else:
                    baseline_bot.update_index(actor)
                    valid_actions = get_valid_actions_dict(state)
                    action_str, amt = baseline_bot.get_action(state, valid_actions)
                    if action_str == "fold":
                        state.fold()
                    elif action_str == "check_or_call":
                        state.check_or_call()
                    else:
                        state.complete_bet_or_raise_to(amt)

            profit_bb = (float(state.stacks[ai_seat]) - float(starting_stacks[ai_seat])) / float(big_blind)
            ai_winnings_per_hand_bb.append(profit_bb)

            # --- 3. UPDATE GAME MEMORY (At the end of the hand) ---
            if not amnesia_mode:
                with torch.no_grad():
                    game_hidden = ai_player.network.update_game_memory(hand_hidden, game_hidden)

    metrics = {
        "winnings_bb": np.array(ai_winnings_per_hand_bb),
        "divergence_rate": (diverged_actions / total_decisions * 100) if total_decisions > 0 else 0.0,
        "avg_mse": np.mean(l2_distances) if l2_distances else 0.0
    }
    return metrics


def evaluate_memory_impact(num_matches, hands_per_match, model_path):
    device = torch.device("cpu")
    action_interpreter = ActionInterpreter("beta")
    baseline_bot = FastBaselineBot(player_index=0)

    print(f"\n🧠 Loading RNN Model for Memory Diagnostics: {model_path}")
    ai_player = load_rnn_player(model_path, device)

    total_hands = num_matches * hands_per_match

    # --- Phase 1: Normal Play + Shadow Execution ---
    print(
        f"\n▶️ Phase 1: Normal Play & Divergence Tracking ({num_matches} matches, {hands_per_match} hands each = {total_hands} total hands)...")
    normal_metrics = simulate_memory_test_games(
        ai_player, num_matches, hands_per_match, action_interpreter, baseline_bot, amnesia_mode=False,
        track_divergence=True
    )

    # --- Phase 2: Amnesia Play ---
    print(f"▶️ Phase 2: Full Amnesia Play (Memory Wiped at Every Step) ({total_hands} total hands)...")
    amnesia_metrics = simulate_memory_test_games(
        ai_player, num_matches, hands_per_match, action_interpreter, baseline_bot, amnesia_mode=True,
        track_divergence=False
    )

    # --- Statistical Breakdown ---
    normal_wr = np.mean(normal_metrics['winnings_bb']) * 100
    amnesia_wr = np.mean(amnesia_metrics['winnings_bb']) * 100

    t_stat, p_value = stats.ttest_ind(normal_metrics['winnings_bb'], amnesia_metrics['winnings_bb'], equal_var=False)

    print("\n" + "=" * 60)
    print("🧠 RECURRENT MEMORY DIAGNOSTIC REPORT 🧠".center(60))
    print("=" * 60)

    print("\n📊 1. POLICY DIVERGENCE (Impact of Memory on Raw Actions)")
    print(f"   Avg Tensor Output Difference (MSE): {normal_metrics['avg_mse']:.5f}")
    print(f"   Discrete Action Changed:            {normal_metrics['divergence_rate']:.2f}% of decisions")

    if normal_metrics['divergence_rate'] < 1.0:
        print("   ⚠️ WARNING: Memory is barely changing decisions. The agent may have 'Learned Amnesia'.")
    else:
        print("   ✅ Memory is actively altering decisions on a fundamental level.")

    print("\n📈 2. EMPIRICAL WIN RATE (Normal vs Amnesia)")
    print(f"   Normal Memory Win Rate:  {normal_wr:>+6.2f} bb/100")
    print(f"   Amnesia Mode Win Rate:   {amnesia_wr:>+6.2f} bb/100")
    print(f"   Win Rate Difference:     {normal_wr - amnesia_wr:>+6.2f} bb/100")

    print("\n📉 3. STATISTICAL SIGNIFICANCE (Welch's t-test)")
    print(f"   P-Value: {p_value:.5f}")
    if p_value < 0.05 and normal_wr > amnesia_wr:
        print("   ✅ RESULT: Memory provides a STATISTICALLY SIGNIFICANT advantage.")
    elif p_value < 0.05 and amnesia_wr > normal_wr:
        print("   ❌ RESULT: Memory makes the bot significantly WORSE. It is over-fitting or hallucinating patterns.")
    else:
        print("   ⚖️ RESULT: The win rate difference is not statistically significant. More hands needed.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Test the impact of RNN memory on Poker AI behavior.")
    parser.add_argument("--matches", type=int, default=20, help="Number of multi-hand sessions")
    parser.add_argument("--hands_per_match", type=int, default=500, help="Hands played consecutively per match")
    parser.add_argument("--run_folder", type=str, default=None, help="Path to the run folder")
    parser.add_argument("--player_id", type=int, default=0, help="ID of the trained player")

    args = parser.parse_args()
    folder = args.run_folder if args.run_folder else get_latest_run_folder()
    path = os.path.join(folder, "players", f"{args.player_id}.pt")

    evaluate_memory_impact(args.matches, args.hands_per_match, path)