import os
import glob
import argparse
import random
import time
import torch
import numpy as np
from pokerkit import NoLimitTexasHoldem, Automation

# --- Local Project Imports ---
from src.state_interpreter import extract_state_snapshot
from src.action_interpreter import ActionInterpreter, Action
from src.alg import PPO, PPOInferenceWrapper
from evaluate import get_latest_run_folder


def flatten_cards(cards):
    """Recursively flattens PokerKit's nested board representations to extract all cards."""
    flat = []
    for c in cards:
        if isinstance(c, (list, tuple)):
            flat.extend(flatten_cards(c))
        else:
            flat.append(c)
    return flat


def load_all_models(run_folder, device):
    """Loads the entire population of models into RAM for fast tournament play."""
    players_dir = os.path.join(run_folder, "players")
    if not os.path.exists(players_dir):
        raise FileNotFoundError(f"Players directory not found: {players_dir}")

    model_files = glob.glob(os.path.join(players_dir, "*.pt"))
    model_files.sort(key=lambda f: int(os.path.splitext(os.path.basename(f))[0]))

    if not model_files:
        raise ValueError("No .pt files found in the players directory!")

    population = {}
    print(f"Loading {len(model_files)} models into memory...", end=" ", flush=True)
    for model_path in model_files:
        player_id = os.path.splitext(os.path.basename(model_path))[0]
        # Initialize bare network
        models = PPO.init_networks(device, mode="beta", discrete=False)
        ai_player = PPOInferenceWrapper(models, discrete=False)
        # Load weights
        checkpoint, _ = torch.load(model_path, map_location=device, weights_only=True)
        ai_player.load_params(checkpoint)
        ai_player.to(device)
        population[player_id] = ai_player

    print("Done!")
    return population, list(population.keys())


def play_tournament_hand(player_ids, population, action_interpreter, small_blind=1, big_blind=2, starting_bb=100):
    """Plays a single hand between the provided player IDs and returns the profit in BB."""
    starting_chips = starting_bb * big_blind
    starting_stacks = [starting_chips] * len(player_ids)

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
        raw_blinds_or_straddles=(small_blind, big_blind),
        min_bet=big_blind,
        raw_starting_stacks=starting_stacks,
        player_count=len(player_ids)
    )

    while state.status:
        actor_idx = state.actor_index
        pid = player_ids[actor_idx]
        ai_player = population[pid]

        snapshot = extract_state_snapshot(state, actor_idx)
        with torch.no_grad():
            action_tensor = ai_player.get_action((snapshot, actor_idx))

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
            legal_min = state.min_completion_betting_or_raising_to_amount
            legal_max = state.max_completion_betting_or_raising_to_amount
            if legal_min is not None and legal_max is not None:
                clamped_bet = max(legal_min, min(bet_sizing, legal_max))
                state.complete_bet_or_raise_to(clamped_bet)
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

    profits_bb = []
    for i in range(len(player_ids)):
        profit_chips = float(state.stacks[i]) - float(starting_stacks[i])
        profits_bb.append(profit_chips / float(big_blind))

    return profits_bb


def run_tournament(population, player_ids, action_interpreter, hands_per_player, max_table_size=2):
    """Schedules random matchups until every player has played the target number of hands."""
    print(f"\n⚔️  STARTING TOURNAMENT: {hands_per_player} hands per model...")

    hands_played = {pid: 0 for pid in player_ids}
    total_profit_bb = {pid: 0.0 for pid in player_ids}
    current_total_hands = 0
    start_time = time.time()

    active_players = list(player_ids)
    while len(active_players) >= max_table_size:
        table_pids = random.sample(active_players, max_table_size)
        profits = play_tournament_hand(table_pids, population, action_interpreter)

        for i, pid in enumerate(table_pids):
            hands_played[pid] += 1
            total_profit_bb[pid] += profits[i]
            current_total_hands += 1

            if hands_played[pid] >= hands_per_player:
                active_players.remove(pid)

        if current_total_hands % 1000 == 0:
            print(f"  Played {current_total_hands} player-hands...")

    elapsed = time.time() - start_time

    leaderboard = []
    for pid in player_ids:
        if hands_played[pid] > 0:
            bb_100 = (total_profit_bb[pid] / hands_played[pid]) * 100
            leaderboard.append({
                "id": pid, "bb_100": bb_100, "hands": hands_played[pid], "profit": total_profit_bb[pid]
            })

    leaderboard.sort(key=lambda x: x["bb_100"], reverse=True)

    print(f"\n🏆 TOURNAMENT COMPLETE ({elapsed:.1f}s) 🏆")
    print("=" * 60)
    print(f"{'Rank':<5} | {'Model ID':<10} | {'Win Rate':<15} | {'Hands Played'}")
    print("-" * 60)
    for rank, p in enumerate(leaderboard):
        rank_str = f"#{rank + 1}"
        marker = "🔥" if rank < 3 else "  "
        print(f"{rank_str:<5} | ID: {p['id']:<6} {marker} | {p['bb_100']:+8.2f} bb/100 | {p['hands']}")
    print("=" * 60)

    return leaderboard


def showcase_hand(p1_id, p2_id, population, action_interpreter, hand_num):
    """Plays a single hand between two top models and prints an omniscient play-by-play."""
    print(f"\n🎬 --- SHOWCASE HAND #{hand_num} : [ID {p1_id}] vs [ID {p2_id}] ---")

    player_ids = [p1_id, p2_id]
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
        raw_blinds_or_straddles=(1, 2),
        min_bet=2,
        raw_starting_stacks=[200, 200],
        player_count=2
    )

    big_blind = state.blinds_or_straddles[-1]

    p1_cards = "".join([repr(c) for c in state.hole_cards[0]])
    p2_cards = "".join([repr(c) for c in state.hole_cards[1]])

    print(f"Seat 0 (SB): Player {p1_id:>3} | Stack: 100 BB | Hole Cards: [{p1_cards}]")
    print(f"Seat 1 (BB): Player {p2_id:>3} | Stack: 100 BB | Hole Cards: [{p2_cards}]")
    print("-" * 50)

    known_street = "PREFLOP"
    print(f"🌍 STREET: PREFLOP")

    while state.status:
        # Securely extract the board using the new flattener
        flat_board = flatten_cards(state.board_cards)
        current_board_len = len(flat_board)

        if current_board_len == 3 and known_street == "PREFLOP":
            known_street = "FLOP"
            formatted_board = " ".join([repr(c) for c in flat_board])
            print(f"\n🌍 STREET: FLOP | Board: [{formatted_board}]")

        elif current_board_len == 4 and known_street == "FLOP":
            known_street = "TURN"
            formatted_board = " ".join([repr(c) for c in flat_board])
            print(f"\n🌍 STREET: TURN | Board: [{formatted_board}]")

        elif current_board_len == 5 and known_street == "TURN":
            known_street = "RIVER"
            formatted_board = " ".join([repr(c) for c in flat_board])
            print(f"\n🌍 STREET: RIVER | Board: [{formatted_board}]")

        actor_idx = state.actor_index
        pid = player_ids[actor_idx]
        ai_player = population[pid]

        snapshot = extract_state_snapshot(state, actor_idx)
        with torch.no_grad():
            action_tensor = ai_player.get_action((snapshot, actor_idx))

        s_min_bet = state.min_completion_betting_or_raising_to_amount or max(state.bets)
        s_max_bet = state.max_completion_betting_or_raising_to_amount or s_min_bet
        interpreted_action, bet_sizing = action_interpreter(action_tensor, s_min_bet, s_max_bet)

        executed_action_str = ""

        if interpreted_action == Action.CHECK_OR_FOLD:
            if state.can_check_or_call() and state.checking_or_calling_amount == 0:
                state.check_or_call()
                executed_action_str = "CHECK"
            elif state.can_fold():
                state.fold()
                executed_action_str = "FOLD"

        elif interpreted_action == Action.CHECK_OR_CALL:
            if state.can_check_or_call():
                amt = state.checking_or_calling_amount
                state.check_or_call()
                executed_action_str = "CHECK" if amt == 0 else "CALL"
            elif state.can_fold():
                state.fold()
                executed_action_str = "FOLD (Fallback)"

        elif interpreted_action == Action.RAISE:
            legal_min = state.min_completion_betting_or_raising_to_amount
            legal_max = state.max_completion_betting_or_raising_to_amount
            print(legal_min, legal_max)
            if legal_min is not None and legal_max is not None:
                clamped_bet = max(legal_min, min(bet_sizing, legal_max))
                state.complete_bet_or_raise_to(clamped_bet)
                executed_action_str = f"RAISE to {float(clamped_bet) / float(big_blind):.1f} BB"
            elif state.can_check_or_call():
                amt = state.checking_or_calling_amount
                state.check_or_call()
                executed_action_str = "CHECK (Fallback)" if amt == 0 else "CALL (Fallback)"
            elif state.can_fold():
                state.fold()
                executed_action_str = "FOLD (Fallback)"

        else:  # ALL_IN
            all_in_size = state.max_completion_betting_or_raising_to_amount
            if state.can_complete_bet_or_raise_to(all_in_size):
                state.complete_bet_or_raise_to(all_in_size)
                executed_action_str = f"ALL IN ({float(all_in_size) / float(big_blind):.1f} BB)"
            elif state.can_check_or_call():
                amt = state.checking_or_calling_amount
                state.check_or_call()
                executed_action_str = "CHECK (Fallback)" if amt == 0 else "CALL (Fallback)"
            elif state.can_fold():
                state.fold()
                executed_action_str = "FOLD (Fallback)"

        print(f"  👉 Player {pid:>3} chooses: {executed_action_str}")

    print("-" * 50)

    # ---------------------------------------------------------
    # THE FIX: Robutsly extract the final board cards
    # ---------------------------------------------------------
    flat_final_board = flatten_cards(state.board_cards)
    if len(flat_final_board) > 0:
        final_board = " ".join([repr(c) for c in flat_final_board])
    else:
        final_board = ""

    print(f"🏁 FINAL BOARD: [{final_board}]")

    profit_p1 = (float(state.stacks[0]) - 200.0) / float(big_blind)
    profit_p2 = (float(state.stacks[1]) - 200.0) / float(big_blind)

    if profit_p1 > 0:
        print(f"💰 Player {p1_id} wins {profit_p1:+.1f} BB!")
    elif profit_p2 > 0:
        print(f"💰 Player {p2_id} wins {profit_p2:+.1f} BB!")
    else:
        print("🤝 Chopped Pot!")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description="Run a round-robin tournament between all trained AIs.")
    parser.add_argument("--hands", type=int, default=500, help="Hands to play per model in the tournament.")
    parser.add_argument("--showcase_hands", type=int, default=5,
                        help="Number of hands to showcase between the top 2 models.")
    parser.add_argument("--run_folder", type=str, default=None, help="Path to the specific run folder")

    args = parser.parse_args()
    target_folder = args.run_folder if args.run_folder else get_latest_run_folder()

    device = torch.device("cpu")
    action_interpreter = ActionInterpreter()

    print(f"\n==================================================")
    print(f"🌐 AI POPULATION TOURNAMENT 🌐")
    print(f"Run Folder: {target_folder}")
    print(f"==================================================")

    population, pids = load_all_models(target_folder, device)

    if len(pids) < 2:
        print("Need at least 2 models to run a tournament!")
        exit()

    leaderboard = run_tournament(population, pids, action_interpreter, args.hands, max_table_size=2)

    if args.showcase_hands > 0 and len(leaderboard) >= 2:
        top_1 = leaderboard[0]["id"]
        top_2 = leaderboard[1]["id"]

        print("\n\n" + "=" * 60)
        print(f"🍿 MAIN EVENT SHOWCASE: THE TOP 2 MODELS 🍿")
        print("=" * 60)
        print(f"Matchup: Model ID {top_1} (Rank 1) vs Model ID {top_2} (Rank 2)")

        for i in range(args.showcase_hands):
            if i % 2 == 0:
                showcase_hand(top_1, top_2, population, action_interpreter, i + 1)
            else:
                showcase_hand(top_2, top_1, population, action_interpreter, i + 1)