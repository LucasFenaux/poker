import os
import glob
import argparse
import torch
import numpy as np
from scipy import stats
import time  # <-- Imported the time module

# --- Local Project Imports ---


# --- Import Shared Logic from evaluate.py ---
from evaluate import (
    FastBaselineBot,
    get_latest_run_folder,
    load_ai_player,
    simulate_eval_games
)


def evaluate_population(num_games, run_folder, max_table_size):
    device = torch.device("cpu")
    from src.game_registry import get_current_game_config
    ActionInterpreter = get_current_game_config()['action_interpreter']
    action_interpreter = ActionInterpreter()
    baseline_bot = FastBaselineBot(player_index=0)

    players_dir = os.path.join(run_folder, "players")
    if not os.path.exists(players_dir):
        raise FileNotFoundError(f"Players directory not found: {players_dir}")

    # Find all model files and sort them to evaluate in order (0.pt, 1.pt, etc.)
    model_files = glob.glob(os.path.join(players_dir, "*.pt"))
    model_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

    if not model_files:
        print("No .pt files found in the players directory!")
        return

    print(f"\n==================================================")
    print(f"🌟 STARTING POPULATION EVALUATION 🌟")
    print(f"Run Folder: {run_folder}")
    print(f"Population Size: {len(model_files)} models")
    print(f"Format: Dynamic (2 to {max_table_size}-Max) | {num_games} Hands per Model")
    print(f"==================================================\n")

    population_metrics = []
    counts = {"sig_win": 0, "sig_loss": 0, "lean_win": 0, "lean_loss": 0}
    # --- Start Overall Timer ---
    overall_start_time = time.time()

    # Evaluate each player individually
    for model_path in model_files:
        # --- Start Player Timer ---
        player_start_time = time.time()

        player_id = os.path.splitext(os.path.basename(model_path))[0]
        print(f"Evaluating Player {player_id:>3}...", end=" ", flush=True)

        try:
            ai_player = load_ai_player(model_path, device)

            # Delegate to the shared loop!
            ai_winnings_bb = simulate_eval_games(
                ai_player, num_games, max_table_size, action_interpreter, baseline_bot, verbose=False
            )

            avg_profit_per_hand_bb = np.mean(ai_winnings_bb)
            win_rate_bb_100 = avg_profit_per_hand_bb * 100

            # T-Test. Catch edge cases where variance is exactly 0
            if np.std(ai_winnings_bb) == 0:
                t_stat, p_value = 0.0, 1.0
            else:
                test_result = stats.ttest_1samp(ai_winnings_bb, 0.0)
                t_stat = test_result.statistic
                p_value = test_result.pvalue

            # --- FOUR-WAY MARKER LOGIC ---
            is_significant = bool(p_value < 0.05)
            is_positive = bool(t_stat > 0)

            # ANSI Color Codes for terminal
            GREEN_UP = "\033[92m▲\033[0m"
            RED_DOWN = "\033[91m▼\033[0m"

            if is_significant and is_positive:
                marker, cat_key = "✅ ", "sig_win"
            elif is_significant and not is_positive:
                marker, cat_key = "❌ ", "sig_loss"
            elif not is_significant and is_positive:
                marker, cat_key = f"{GREEN_UP} ", "lean_win"
            else:
                marker, cat_key = f"{RED_DOWN} ", "lean_loss"

            counts[cat_key] += 1
            # Stop Player Timer
            player_elapsed = time.time() - player_start_time

            # Save metrics for aggregate report
            population_metrics.append({
                "id": player_id, "win_rate": win_rate_bb_100, "p_value": p_value,
                "marker": marker, "is_winner": (cat_key == "sig_win")
            })

            print(
                f"Done! ({player_elapsed:.1f}s) | {marker} | Win Rate: {win_rate_bb_100:+8.2f} bb/100 | p-value: {p_value:.4f}")

        except Exception as e:
            player_elapsed = time.time() - player_start_time
            print(f"FAILED ❌ ({player_elapsed:.1f}s) Error: {e}")

    # --- Stop Overall Timer ---
    overall_elapsed = time.time() - overall_start_time

    # --- Aggregate Population Report ---
    if not population_metrics:
        print("\nNo metrics collected. Evaluation aborted.")
        return

    win_rates = [m["win_rate"] for m in population_metrics]
    significant_winners = [m for m in population_metrics if m["is_winner"]]

    best_player = max(population_metrics, key=lambda x: x["win_rate"])
    worst_player = min(population_metrics, key=lambda x: x["win_rate"])

    avg_population_win_rate = np.mean(win_rates)
    std_population_win_rate = np.std(win_rates)
    winner_percentage = (len(significant_winners) / len(population_metrics)) * 100

    print("\n" + "=" * 60)
    print("🏆 AGGREGATE POPULATION REPORT 🏆".center(60))
    print("=" * 60)
    print(f"Total Evaluation Time:       {overall_elapsed:.1f} seconds ({overall_elapsed / 60:.2f} mins)")
    print(f"Average Population Win Rate: {avg_population_win_rate:+.2f} bb/100")
    print(f"Population Std Deviation:    {std_population_win_rate:.2f} bb/100")
    print("-" * 60)
    print(
        f"Statistically Significant Winners: {len(significant_winners)} out of {len(population_metrics)} ({winner_percentage:.1f}%)")
    print("-" * 60)

    print("📊 CATEGORY TALLY:")
    print(f"  ✅ Significant Winners: {counts['sig_win']:>4}")
    print(f"  \033[92m▲\033[0m Leaning Winners:     {counts['lean_win']:>4}")
    print(f"  \033[91m▼\033[0m Leaning Losers:      {counts['lean_loss']:>4}")
    print(f"  ❌ Significant Losers:  {counts['sig_loss']:>4}")
    print("-" * 60)

    # Print a quick glance leaderboard with the legend
    print("📈 POPULATION LEADERBOARD:")
    print("  Legend: ✅ Sig. Winner | ❌ Sig. Loser | \033[92m▲\033[0m Lean Winner | \033[91m▼\033[0m Lean Loser")
    print("-" * 60)

    for m in sorted(population_metrics, key=lambda x: int(x["id"])):
        print(f"  Player {m['id']:>3}: {m['marker']} ({m['win_rate']:+7.2f} bb/100)")

    print("-" * 60)
    print(
        f"🌟 Best Player:  ID {best_player['id']:>3} | {best_player['win_rate']:+8.2f} bb/100 (p={best_player['p_value']:.4f})")
    print(
        f"📉 Worst Player: ID {worst_player['id']:>3} | {worst_player['win_rate']:+8.2f} bb/100 (p={worst_player['p_value']:.4f})")
    print("=" * 60)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate an entire population of trained Poker AIs.")
    parser.add_argument("--games", type=int, default=200, help="Number of hands to play PER MODEL (default: 200)")
    parser.add_argument("--run_folder", type=str, default=None, help="Path to the specific run folder")
    parser.add_argument("--max_table_size", type=int, default=2,
                        help="Maximum number of players at the table (default: 2)")

    args = parser.parse_args()
    target_folder = args.run_folder if args.run_folder else get_latest_run_folder()

    evaluate_population(args.games, target_folder, args.max_table_size)