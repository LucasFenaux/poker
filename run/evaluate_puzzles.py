import os
import glob
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass

# --- PokerKit Imports ---
from pokerkit import calculate_equities, parse_range, Deck, StandardHighHand, Card

# --- Local Project Imports ---
from src.state_interpreter import StateSnapshot
from src.action_interpreter import ActionInterpreter, Action
from evaluate import get_latest_run_folder
from src.alg import PPO


@dataclass
class PokerPuzzle:
    name: str
    description: str
    snapshot: StateSnapshot
    actor_index: int
    acceptable_actions: list[Action]
    baseline_ev: float = 0.0  # NEW: Holds the True Mathematical EV


def load_eval_models(model_path, device):
    """Custom loader that extracts BOTH the Policy and Value networks."""
    policy_net, value_net = PPO.init_networks(device, mode="beta", discrete=False)
    checkpoint, _ = torch.load(model_path, map_location=device, weights_only=True)

    # Checkpoint contains: [network_dict, value_dict, opt_dict, value_opt_dict]
    policy_net.load_state_dict(checkpoint[0])
    value_net.load_state_dict(checkpoint[1])

    policy_net.to(device)
    value_net.to(device)
    policy_net.eval()
    value_net.eval()

    return policy_net, value_net


def build_puzzle_suite() -> list[PokerPuzzle]:
    """
    Constructs a comprehensive suite of exact poker scenarios.
    All scenarios assume Heads Up (2-Max), Blinds 1/2.
    """
    puzzles = []

    puzzles.append(PokerPuzzle(
        name="AA Facing Shove", description="Preflop: Holding Pocket Aces facing a 100BB All-In.", actor_index=1,
        acceptable_actions=[Action.CHECK_OR_CALL, Action.RAISE],
        snapshot=StateSnapshot(hole_cards="AsAc", board_cards="??????????", player_count=2, blinds_or_straddles=(1, 2),
                               bets=[100.0, 2.0], stacks=[0.0, 98.0], in_hand=[True, True], pots=[0.0],
                               min_bet=98.0, max_bet=98.0)
    ))

    puzzles.append(PokerPuzzle(
        name="72o Facing 3-Bet", description="Preflop: Holding 72 offsuit facing a heavy 3-bet.", actor_index=0,
        acceptable_actions=[Action.CHECK_OR_FOLD],
        snapshot=StateSnapshot(hole_cards="7s2c", board_cards="??????????", player_count=2, blinds_or_straddles=(1, 2),
                               bets=[3.0, 10.0], stacks=[97.0, 90.0], in_hand=[True, True], pots=[0.0],
                               min_bet=7.0, max_bet=90.0)
    ))

    puzzles.append(PokerPuzzle(
        name="Nut Flush River", description="River: Holding the Nut Flush facing a half-pot bet.", actor_index=0,
        acceptable_actions=[Action.CHECK_OR_CALL, Action.RAISE],
        snapshot=StateSnapshot(hole_cards="AhKh", board_cards="2h7hThQc4h", player_count=2, blinds_or_straddles=(1, 2),
                               bets=[0.0, 20.0], stacks=[30.0, 10.0], in_hand=[True, True], pots=[40.0],
                               min_bet=20.0, max_bet=30.0)
    ))

    puzzles.append(PokerPuzzle(
        name="Busted Draw River", description="River: Missed completely. Facing a massive over-bet shove.",
        actor_index=0,
        acceptable_actions=[Action.CHECK_OR_FOLD],
        snapshot=StateSnapshot(hole_cards="AhJh", board_cards="2h5h8cTsKd", player_count=2, blinds_or_straddles=(1, 2),
                               bets=[0.0, 50.0], stacks=[50.0, 0.0], in_hand=[True, True], pots=[30.0],
                               min_bet=50.0, max_bet=50.0)
    ))

    puzzles.append(PokerPuzzle(
        name="Short Stack AKo", description="Preflop: Holding AKo with only 5 Big Blinds. Must shove.", actor_index=0,
        acceptable_actions=[Action.RAISE],
        snapshot=StateSnapshot(hole_cards="AsKd", board_cards="??????????", player_count=2, blinds_or_straddles=(1, 2),
                               bets=[1.0, 2.0], stacks=[4.0, 98.0], in_hand=[True, True], pots=[0.0],
                               min_bet=1.0, max_bet=4.0)
    ))

    puzzles.append(PokerPuzzle(
        name="Flopped Quads", description="Flop: Holding Quads facing a small continuation bet.", actor_index=1,
        acceptable_actions=[Action.CHECK_OR_CALL, Action.RAISE],
        snapshot=StateSnapshot(hole_cards="Td9s", board_cards="ThTsTc????", player_count=2, blinds_or_straddles=(1, 2),
                               bets=[5.0, 0.0], stacks=[85.0, 90.0], in_hand=[True, True], pots=[10.0],
                               min_bet=5.0, max_bet=85.0)
    ))

    puzzles.append(PokerPuzzle(
        name="KK Facing Open", description="Preflop: Holding Pocket Kings facing a standard 3BB open.", actor_index=1,
        acceptable_actions=[Action.CHECK_OR_CALL, Action.RAISE],
        snapshot=StateSnapshot(hole_cards="KsKc", board_cards="??????????", player_count=2, blinds_or_straddles=(1, 2),
                               bets=[3.0, 2.0], stacks=[97.0, 98.0], in_hand=[True, True], pots=[0.0],
                               min_bet=1.0, max_bet=97.0)
    ))

    puzzles.append(PokerPuzzle(
        name="23o Air River", description="River: Holding 23 offsuit on a high board. Facing a bet.", actor_index=0,
        acceptable_actions=[Action.CHECK_OR_FOLD],
        snapshot=StateSnapshot(hole_cards="2s3c", board_cards="AcKc5d9sJh", player_count=2, blinds_or_straddles=(1, 2),
                               bets=[0.0, 10.0], stacks=[40.0, 30.0], in_hand=[True, True], pots=[20.0],
                               min_bet=10.0, max_bet=30.0)
    ))

    return puzzles


def calculate_math_evs(puzzles: list[PokerPuzzle]):
    """Uses PokerKit to calculate the true mathematical EV for the optimal action."""
    print("🧮 Calculating True Mathematical EV for puzzles using PokerKit...")

    # We test against the identical villain range from the evaluate.py Baseline Bot
    villain_range = parse_range('22+,A2+,K2+,Q2+,J2+,T2+,92+,82+,72+,62+,52+,42+,32+')

    for p in puzzles:
        # 1. Fold Scenarios have exactly 0.0 EV (No future profit expected)
        if Action.CHECK_OR_FOLD in p.acceptable_actions and len(p.acceptable_actions) == 1:
            p.baseline_ev = 0.0
            continue

        # 2. Extract and format the board (THE FIX: Pass rank and suit as two separate arguments)
        board_str = p.snapshot.board_cards.replace("?", "")
        flat_board = [Card(board_str[i], board_str[i + 1]) for i in range(0, len(board_str), 2)]

        # 3. Calculate True Equity
        my_range = parse_range(p.snapshot.hole_cards)
        equities = calculate_equities(
            (my_range, villain_range), flat_board, 2, 5, Deck.STANDARD, (StandardHighHand,), sample_count=1500
        )
        equity = equities[0]

        # 4. Calculate Showdown EV = (Equity * Final_Pot) - Amount_to_Call
        my_bet = p.snapshot.bets[p.actor_index]
        villain_bet = p.snapshot.bets[1 - p.actor_index]
        current_pot = sum(p.snapshot.pots) + my_bet + villain_bet

        if p.name == "Short Stack AKo":
            # Custom sizing for the Shove scenario
            amount_to_call = 4.0  # We shove 4BB total
            final_pot = 10.0  # 5BB from us, 5BB from them
        else:
            # Standard Call EV
            amount_to_call = max(0.0, villain_bet - my_bet)
            final_pot = current_pot + amount_to_call

        p.baseline_ev = (equity * final_pot) - amount_to_call

    return puzzles


def plot_population_parameters(puzzle_params: dict, run_folder: str):
    """Generates a scatter plot of Alpha vs Beta for all puzzles."""
    fig, axes = plt.subplots(2, 4, figsize=(20, 10))
    fig.suptitle("Alpha vs Beta Parameters per Puzzle (Action Choice)", fontsize=18, fontweight='bold')

    for ax, (puzzle_name, params) in zip(axes.flatten(), puzzle_params.items()):
        alphas = params['alphas']
        betas = params['betas']

        ax.scatter(alphas, betas, alpha=0.7, color='dodgerblue', edgecolor='k')
        ax.set_title(puzzle_name, fontsize=12, fontweight='bold')
        ax.set_xlabel("Alpha (Pushing towards Raise)", fontsize=10)
        ax.set_ylabel("Beta (Pushing towards Fold)", fontsize=10)

        ax.set_xlim(-2, 52)
        ax.set_ylim(-2, 52)

        ax.plot([-2, 52], [-2, 52], 'r--', alpha=0.5, label="Confusion (Mean 0.5)")
        ax.legend(loc='upper left', fontsize=8)
        ax.grid(True, linestyle='--', alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_path = os.path.join(run_folder, "alpha_beta_analysis.png")
    plt.savefig(plot_path, dpi=200)
    print(f"\n📊 Saved Alpha/Beta scatter plot to: {plot_path}")


def evaluate_puzzles(run_folder):
    device = torch.device("cpu")
    action_interpreter = ActionInterpreter()

    # Build puzzles and calculate their mathematical EV!
    puzzles = build_puzzle_suite()
    puzzles = calculate_math_evs(puzzles)

    players_dir = os.path.join(run_folder, "players")
    if not os.path.exists(players_dir):
        raise FileNotFoundError(f"Players directory not found: {players_dir}")

    model_files = glob.glob(os.path.join(players_dir, "*.pt"))
    model_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

    if not model_files:
        print("No .pt files found!")
        return

    print(f"\n==================================================")
    print(f"🧠 POKER IQ TEST: FIXED SCENARIO EVALUATION 🧠")
    print(f"Run Folder: {run_folder}")
    print(f"Population: {len(model_files)} models | Puzzles: {len(puzzles)}")
    print(f"==================================================\n")

    puzzle_pass_counts = {p.name: 0 for p in puzzles}
    player_scores = []
    puzzle_params = {p.name: {'alphas': [], 'betas': [], 'values': []} for p in puzzles}

    start_time = time.time()

    for model_path in model_files:
        player_id = os.path.splitext(os.path.basename(model_path))[0]
        print(f"\nEvaluating Player {player_id:>3}...")

        try:
            policy_net, value_net = load_eval_models(model_path, device)
            score = 0

            for puzzle in puzzles:
                with torch.no_grad():
                    policy = policy_net(puzzle.snapshot, puzzle.actor_index)
                    raw_value_pred = value_net(puzzle.snapshot, puzzle.actor_index).item()

                    # Reverse the log scaling to get True BB EV
                    sign = 1.0 if raw_value_pred >= 0 else -1.0
                    true_value_pred = sign * (np.exp(abs(raw_value_pred)) - 1.0)

                    alpha_val = policy.concentration1.squeeze()[0].item()
                    beta_val = policy.concentration0.squeeze()[0].item()
                    action_tensor = policy.mean.cpu()

                interpreted_action, _ = action_interpreter(
                    action_tensor, puzzle.snapshot.min_bet, puzzle.snapshot.max_bet
                )

                passed = interpreted_action in puzzle.acceptable_actions
                if passed:
                    score += 1
                    puzzle_pass_counts[puzzle.name] += 1

                puzzle_params[puzzle.name]['alphas'].append(alpha_val)
                puzzle_params[puzzle.name]['betas'].append(beta_val)
                puzzle_params[puzzle.name]['values'].append(true_value_pred)

                # Print trace with Math EV comparison
                mark = "✅" if passed else "❌"
                print(
                    f"  {mark} [{puzzle.name:<18}] α/β: {alpha_val:>5.2f}/{beta_val:>5.2f} | Pred EV: {true_value_pred:>+6.2f} BB | Math EV: {puzzle.baseline_ev:>+6.2f} BB -> {interpreted_action.name}"
                )

            player_scores.append({"id": player_id, "score": score, "total": len(puzzles)})
            print(f"  --> Score: {score}/{len(puzzles)}")

        except Exception as e:
            print(f"Player {player_id:>3}: FAILED TO RUN - {e}")

    elapsed = time.time() - start_time

    # --- Print Aggregate Report ---
    print("\n" + "=" * 80)
    print("🏆 POPULATION IQ REPORT 🏆".center(80))
    print("=" * 80)
    print(f"Evaluation Time: {elapsed:.1f}s")

    avg_score = np.mean([p["score"] for p in player_scores])
    print(f"Average Population Score: {avg_score:.1f} / {len(puzzles)} ({(avg_score / len(puzzles)) * 100:.1f}%)")
    print("-" * 80)

    print("🧩 SCENARIO PASS RATES:")
    for puzzle in puzzles:
        pass_rate = (puzzle_pass_counts[puzzle.name] / len(model_files)) * 100
        print(f"  {puzzle.name:<22}: {pass_rate:>5.1f}% Passed")

    print("-" * 80)

    # ---------------------------------------------------------
    # AVERAGE PARAMETERS REPORT (With True Expected Value)
    # ---------------------------------------------------------
    print("🧠 AVERAGE POPULATION PARAMETERS (AI vs MATH EV):")
    for puzzle in puzzles:
        alphas = puzzle_params[puzzle.name]['alphas']
        betas = puzzle_params[puzzle.name]['betas']
        values = puzzle_params[puzzle.name]['values']

        if len(alphas) > 0:
            avg_alpha = np.mean(alphas)
            avg_beta = np.mean(betas)
            avg_action_mean = avg_alpha / (avg_alpha + avg_beta)
            avg_val = np.mean(values)
        else:
            avg_alpha, avg_beta, avg_action_mean, avg_val = 0.0, 0.0, 0.0, 0.0

        diff = abs(avg_val - puzzle.baseline_ev)

        print(
            f"  {puzzle.name:<18} | Action Mean: {avg_action_mean:.3f} | Pred EV: {avg_val:>+6.2f} BB | Math EV: {puzzle.baseline_ev:>+6.2f} BB | Diff: {diff:>6.2f}"
        )

    print("-" * 80)
    # ---------------------------------------------------------

    best_player = max(player_scores, key=lambda x: x["score"])
    print(f"🌟 Smartest Model: ID {best_player['id']} with {best_player['score']}/{len(puzzles)}")
    print("=" * 80)

    plot_population_parameters(puzzle_params, run_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Poker AIs against fixed logic puzzles.")
    parser.add_argument("--run_folder", type=str, default=None, help="Path to the run folder")

    args = parser.parse_args()
    target_folder = args.run_folder if args.run_folder else get_latest_run_folder()

    evaluate_puzzles(target_folder)