import os
import glob
import argparse
import time
import torch
import numpy as np
import matplotlib.pyplot as plt  # <-- Added for plotting
from dataclasses import dataclass

# --- Local Project Imports ---
from src.state_interpreter import StateSnapshot
from src.action_interpreter import ActionInterpreter, Action
from evaluate import get_latest_run_folder, load_ai_player


@dataclass
class PokerPuzzle:
    name: str
    description: str
    snapshot: StateSnapshot
    actor_index: int
    acceptable_actions: list[Action]


def build_puzzle_suite() -> list[PokerPuzzle]:
    """
    Constructs a comprehensive suite of exact poker scenarios.
    All scenarios assume Heads Up (2-Max), Blinds 1/2.
    """
    puzzles = []

    # 1. PRE-FLOP MONSTER (AA)
    puzzles.append(PokerPuzzle(
        name="AA Facing Shove",
        description="Preflop: Holding Pocket Aces facing a 100BB All-In.",
        actor_index=1,
        acceptable_actions=[Action.CHECK_OR_CALL, Action.RAISE],
        snapshot=StateSnapshot(
            hole_cards="AsAc", board_cards="??????????", player_count=2, blinds_or_straddles=(1, 2),
            bets=[100.0, 2.0], stacks=[0.0, 98.0], in_hand=[True, True], pots=[0.0],
            min_bet=98.0, max_bet=98.0
        )
    ))

    # 2. PRE-FLOP TRASH (72o)
    puzzles.append(PokerPuzzle(
        name="72o Facing 3-Bet",
        description="Preflop: Holding 72 offsuit facing a heavy 3-bet.",
        actor_index=0,
        acceptable_actions=[Action.CHECK_OR_FOLD],
        snapshot=StateSnapshot(
            hole_cards="7s2c", board_cards="??????????", player_count=2, blinds_or_straddles=(1, 2),
            bets=[3.0, 10.0], stacks=[97.0, 90.0], in_hand=[True, True], pots=[0.0],
            min_bet=7.0, max_bet=90.0
        )
    ))

    # 3. POST-FLOP NUTS (Flush)
    puzzles.append(PokerPuzzle(
        name="Nut Flush River",
        description="River: Holding the Nut Flush facing a half-pot bet.",
        actor_index=0,
        acceptable_actions=[Action.CHECK_OR_CALL, Action.RAISE],
        snapshot=StateSnapshot(
            hole_cards="AhKh", board_cards="2h7hThQc4h", player_count=2, blinds_or_straddles=(1, 2),
            bets=[0.0, 20.0], stacks=[30.0, 10.0], in_hand=[True, True], pots=[40.0],
            min_bet=20.0, max_bet=30.0
        )
    ))

    # 4. POST-FLOP AIR (Busted Draw)
    puzzles.append(PokerPuzzle(
        name="Busted Draw River",
        description="River: Missed completely. Facing a massive over-bet shove.",
        actor_index=0,
        acceptable_actions=[Action.CHECK_OR_FOLD],
        snapshot=StateSnapshot(
            hole_cards="AhJh", board_cards="2h5h8cTsKd", player_count=2, blinds_or_straddles=(1, 2),
            bets=[0.0, 50.0], stacks=[50.0, 0.0], in_hand=[True, True], pots=[30.0],
            min_bet=50.0, max_bet=50.0
        )
    ))

    # 5. SHORT STACK PUSH (AKo)
    puzzles.append(PokerPuzzle(
        name="Short Stack AKo",
        description="Preflop: Holding AKo with only 5 Big Blinds. Must shove.",
        actor_index=0,
        acceptable_actions=[Action.RAISE],
        snapshot=StateSnapshot(
            hole_cards="AsKd", board_cards="??????????", player_count=2, blinds_or_straddles=(1, 2),
            bets=[1.0, 2.0], stacks=[4.0, 98.0], in_hand=[True, True], pots=[0.0],
            min_bet=1.0, max_bet=4.0
        )
    ))

    # 6. UNBEATABLE FLOP (Quads)
    puzzles.append(PokerPuzzle(
        name="Flopped Quads",
        description="Flop: Holding Quads facing a small continuation bet.",
        actor_index=1,
        acceptable_actions=[Action.CHECK_OR_CALL, Action.RAISE],
        snapshot=StateSnapshot(
            hole_cards="Td9s", board_cards="ThTsTc????", player_count=2, blinds_or_straddles=(1, 2),
            bets=[5.0, 0.0], stacks=[85.0, 90.0], in_hand=[True, True], pots=[10.0],
            min_bet=5.0, max_bet=85.0
        )
    ))

    # 7. VALUE HAND PREFLOP (KK)
    puzzles.append(PokerPuzzle(
        name="KK Facing Open",
        description="Preflop: Holding Pocket Kings facing a standard 3BB open.",
        actor_index=1,
        acceptable_actions=[Action.CHECK_OR_CALL, Action.RAISE],
        snapshot=StateSnapshot(
            hole_cards="KsKc", board_cards="??????????", player_count=2, blinds_or_straddles=(1, 2),
            bets=[3.0, 2.0], stacks=[97.0, 98.0], in_hand=[True, True], pots=[0.0],
            min_bet=1.0, max_bet=97.0
        )
    ))

    # 8. ABSOLUTE AIR FACING BET (23o)
    puzzles.append(PokerPuzzle(
        name="23o Air River",
        description="River: Holding 23 offsuit on a high board. Facing a bet.",
        actor_index=0,
        acceptable_actions=[Action.CHECK_OR_FOLD],
        snapshot=StateSnapshot(
            hole_cards="2s3c", board_cards="AcKc5d9sJh", player_count=2, blinds_or_straddles=(1, 2),
            bets=[0.0, 10.0], stacks=[40.0, 30.0], in_hand=[True, True], pots=[20.0],
            min_bet=10.0, max_bet=30.0
        )
    ))

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

        # Display the 0 to 50 bounds
        ax.set_xlim(-2, 52)
        ax.set_ylim(-2, 52)

        # Draw a diagonal line indicating absolute confusion (Mean = 0.5)
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
    puzzles = build_puzzle_suite()

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

    # Track aggregate stats
    puzzle_pass_counts = {p.name: 0 for p in puzzles}
    player_scores = []

    # Store parameters for the plot
    puzzle_params = {p.name: {'alphas': [], 'betas': []} for p in puzzles}

    start_time = time.time()

    for model_path in model_files:
        player_id = os.path.splitext(os.path.basename(model_path))[0]
        print(f"\nEvaluating Player {player_id:>3}...")

        try:
            ai_player = load_ai_player(model_path, device)
            score = 0

            for puzzle in puzzles:
                # Inject the mocked state directly into the neural network
                with torch.no_grad():
                    policy = ai_player.get_model_policy(ai_player.network, (puzzle.snapshot, puzzle.actor_index))

                    # Extract Alpha (concentration1) and Beta (concentration0)
                    # Index [0] pulls the action choice dimension (ignoring bet sizing dimension)
                    alpha_val = policy.concentration1.squeeze()[0].item()
                    beta_val = policy.concentration0.squeeze()[0].item()

                    # Take the deterministic Mean
                    action_tensor = policy.mean.cpu()

                # Interpret what the model decided to do
                interpreted_action, _ = action_interpreter(
                    action_tensor, puzzle.snapshot.min_bet, puzzle.snapshot.max_bet
                )

                # Grade it
                passed = interpreted_action in puzzle.acceptable_actions
                if passed:
                    score += 1
                    puzzle_pass_counts[puzzle.name] += 1

                # Store for plotting
                puzzle_params[puzzle.name]['alphas'].append(alpha_val)
                puzzle_params[puzzle.name]['betas'].append(beta_val)

                # Print trace to console
                mark = "✅" if passed else "❌"
                print(
                    f"  {mark} [{puzzle.name:<18}] Mean: {action_tensor[0].item():.3f} | α: {alpha_val:>5.2f} | β: {beta_val:>5.2f} -> {interpreted_action.name}")

            player_scores.append({"id": player_id, "score": score, "total": len(puzzles)})
            print(f"  --> Score: {score}/{len(puzzles)}")

        except Exception as e:
            print(f"Player {player_id:>3}: FAILED TO RUN - {e}")

    elapsed = time.time() - start_time

    # --- Print Aggregate Report ---
    print("\n" + "=" * 60)
    print("🏆 POPULATION IQ REPORT 🏆".center(60))
    print("=" * 60)
    print(f"Evaluation Time: {elapsed:.1f}s")

    avg_score = np.mean([p["score"] for p in player_scores])
    print(f"Average Population Score: {avg_score:.1f} / {len(puzzles)} ({(avg_score / len(puzzles)) * 100:.1f}%)")
    print("-" * 60)

    print("🧩 SCENARIO PASS RATES:")
    for puzzle in puzzles:
        pass_rate = (puzzle_pass_counts[puzzle.name] / len(model_files)) * 100
        print(f"  {puzzle.name:<22}: {pass_rate:>5.1f}% Passed")

    print("-" * 60)
    best_player = max(player_scores, key=lambda x: x["score"])
    print(f"🌟 Smartest Model: ID {best_player['id']} with {best_player['score']}/{len(puzzles)}")
    print("=" * 60)

    # --- Print Aggregate Report ---
    print("\n" + "=" * 60)
    print("🏆 POPULATION IQ REPORT 🏆".center(60))
    print("=" * 60)
    print(f"Evaluation Time: {elapsed:.1f}s")

    avg_score = np.mean([p["score"] for p in player_scores])
    print(f"Average Population Score: {avg_score:.1f} / {len(puzzles)} ({(avg_score / len(puzzles)) * 100:.1f}%)")
    print("-" * 60)

    print("🧩 SCENARIO PASS RATES:")
    for puzzle in puzzles:
        pass_rate = (puzzle_pass_counts[puzzle.name] / len(model_files)) * 100
        print(f"  {puzzle.name:<22}: {pass_rate:>5.1f}% Passed")

    print("-" * 60)

    # ---------------------------------------------------------
    # AVERAGE PARAMETERS REPORT
    # ---------------------------------------------------------
    print("🧠 AVERAGE POPULATION PARAMETERS:")
    for puzzle in puzzles:
        alphas = puzzle_params[puzzle.name]['alphas']
        betas = puzzle_params[puzzle.name]['betas']

        # Avoid division by zero in case of an empty population
        if len(alphas) > 0:
            avg_alpha = np.mean(alphas)
            avg_beta = np.mean(betas)
            avg_action_mean = avg_alpha / (avg_alpha + avg_beta)
        else:
            avg_alpha, avg_beta, avg_action_mean = 0.0, 0.0, 0.0

        print(
            f"  {puzzle.name:<22}: α (Raise) = {avg_alpha:>5.2f} | β (Fold) = {avg_beta:>5.2f} | Action Mean = {avg_action_mean:.3f}")

    print("-" * 60)
    # ---------------------------------------------------------

    best_player = max(player_scores, key=lambda x: x["score"])
    print(f"🌟 Smartest Model: ID {best_player['id']} with {best_player['score']}/{len(puzzles)}")
    print("=" * 60)

    # Generate the plot
    plot_population_parameters(puzzle_params, run_folder)

    # Generate the plot
    plot_population_parameters(puzzle_params, run_folder)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Evaluate Poker AIs against fixed logic puzzles.")
    parser.add_argument("--run_folder", type=str, default=None, help="Path to the run folder")

    args = parser.parse_args()
    target_folder = args.run_folder if args.run_folder else get_latest_run_folder()

    evaluate_puzzles(target_folder)