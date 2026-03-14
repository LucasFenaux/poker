import ray
from ray.util.queue import Queue, Empty
import json
import os
import asyncio
from fractions import Fraction


class PokerEncoder(json.JSONEncoder):
    """
    A custom JSON encoder that converts Fractions (and Decimals)
    into standard floats or strings so they can be saved to a file.
    """

    def default(self, obj):
        # If the object is a Fraction, convert it to a float for saving
        if isinstance(obj, Fraction):
            return float(obj)

            # If you prefer absolute precision in your save files,
        # you can save it as a string instead:
        # return str(obj)

        # Let the base class handle any standard Python types
        return super().default(obj)


@ray.remote(num_cpus=0)
class LeaderboardActor:
    def __init__(self, queue: Queue, player_ids, save_folder: str = "./"):
        self.queue = queue
        self.save_folder = save_folder
        self.history_player_winnings = {player_id: [] for player_id in player_ids}
        self.player_ids = player_ids
        # New per-player game counter
        self.number_games_played = {player_id: 0 for player_id in player_ids}
        self.is_done = False
        self.last_saved_avg = -1

    async def start(self):
        while not self.is_done:
            try:
                # Use block=False and an await so we don't freeze the actor!
                # This allows the GUI's .remote() calls to be processed.
                data = await asyncio.wait_for(self.queue.get_async(), timeout=1.0)

                if data is not None:
                    player_id, player_winnings = data
                    self.update(player_id, player_winnings)

            except (asyncio.TimeoutError, TimeoutError):  # <--- Fixed exception type!
                continue

        # Use asyncio.sleep instead of time.sleep in an async method
        await asyncio.sleep(10)

    def set_done(self):
        self.is_done = True

    def update(self, player_id, player_winnings):
        self.history_player_winnings[player_id].append(player_winnings)
        self.number_games_played[player_id] += 1
        # self.save()  # Ensures it saves to the file every time a game finishes

        # Calculate the current average as an integer (e.g., 15.8 becomes 15)
        total_games = sum(self.number_games_played.values())
        current_avg_int = int(total_games / max(1, len(self.player_ids)))

        # Only trigger the hard drive save if we passed a new integer threshold
        if current_avg_int > self.last_saved_avg:
            self.save()
            self.last_saved_avg = current_avg_int

    def generate_leaderboard_data(self):
        stats = []
        for p_id in self.player_ids:
            history = self.history_player_winnings[p_id]

            total_winnings = sum(history)
            # Slice the last 10 games and calculate average
            recent_history = history[-10:] if len(history) >= 10 else history
            recent_avg = sum(recent_history) / len(recent_history) if recent_history else 0

            stats.append({
                "id": p_id,
                "total": total_winnings,
                # Fetch the individual game count for this player
                "game_count": self.number_games_played[p_id],
                "recent_avg": recent_avg,
            })

        # Sort by total winnings descending for the main board
        all_time = sorted(stats, key=lambda x: x['total'], reverse=True)
        # Sort by recent performance descending
        recent_top = sorted(stats, key=lambda x: x['recent_avg'], reverse=True)

        return all_time, recent_top

    def get_leaderboard_stats(self):
        all_time, recent_top = self.generate_leaderboard_data()

        # Calculate the AVERAGE games played
        total_games = sum(self.number_games_played.values())
        avg_games = total_games / max(1, len(self.player_ids))

        return {
            "all_time": all_time,
            "recent": recent_top,
            "avg_games": avg_games,  # Replaced total_games
            "is_done": self.is_done
        }

    def save(self):
        # 1. Save the histogram
        os.makedirs(self.save_folder, exist_ok=True)
        with open(os.path.join(self.save_folder, "winning_histogram.json"), "w") as f:
            json.dump(self.history_player_winnings, f, indent=4, cls=PokerEncoder)

        # 2. Generate and Save Table
        all_time, recent_top = self.generate_leaderboard_data()

        # Calculate the AVERAGE games played
        total_games = sum(self.number_games_played.values())
        avg_games = total_games / max(1, len(self.player_ids))

        table_path = os.path.join(self.save_folder, "leaderboard.txt")
        with open(table_path, "w") as f:
            f.write(f"=== ALL-TIME LEADERBOARD (Avg Games/Player: {avg_games:.1f}) ===\n")
            f.write(f"{'Rank':<5} | {'Player ID':<10} | {'Games':<6} | {'Total Winnings':<15}\n")
            f.write("-" * 45 + "\n")
            for i, p in enumerate(all_time, 1):
                f.write(f"{i:<5} | {p['id']:<10} | {p['game_count']:<6} | {p['total']:<15.2f}\n")

            f.write("\n=== TOP PERFORMERS (LAST 10 GAMES) ===\n")
            f.write(f"{'Rank':<5} | {'Player ID':<10} | {'Games':<6} | {'Avg Winnings':<15}\n")
            f.write("-" * 45 + "\n")
            for i, p in enumerate(recent_top, 1):
                f.write(f"{i:<5} | {p['id']:<10} | {p['game_count']:<6} | {p['recent_avg']:<15.2f}\n")

        print(f"Avg Games: {avg_games:.1f} - Leaderboard saved to {table_path}")