import ray
import json
import os

@ray.remote(num_cpus=0.1)
class LeaderboardActor:
    def __init__(self, player_ids, save_folder: str = "./"):
        self.save_folder = save_folder
        self.game_counter = 0
        self.history_player_winnings = {player_id: [] for player_id in player_ids}
        self.player_ids = player_ids
        self.current_game_winnings = {player_id: 0 for player_id in player_ids}
        self.is_done = False

    def set_done(self):
        self.is_done = True

    def update(self, player_winnings: dict[int: float]):
        for player_id, player_winning in player_winnings.items():
            # assert isinstance(player_winning, float), print(type(player_winning))
            self.current_game_winnings[player_id] += player_winning

    def update_game_counter(self):
        self.game_counter += 1
        # push the current game winnings to the histogram
        for player_id, player_winning in self.current_game_winnings.items():
            self.history_player_winnings[player_id].append(player_winning)
            self.current_game_winnings[player_id] = 0

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
                "recent_avg": recent_avg,
            })

        # Sort by total winnings descending for the main board
        all_time = sorted(stats, key=lambda x: x['total'], reverse=True)
        # Sort by recent performance descending
        recent_top = sorted(stats, key=lambda x: x['recent_avg'], reverse=True)

        return all_time, recent_top

    def get_leaderboard_stats(self):
        # Re-use the logic we wrote before to get sorted lists
        all_time, recent_top = self.generate_leaderboard_data()
        return {
            "all_time": all_time,
            "recent": recent_top,
            "game_count": self.game_counter,
            "is_done": self.is_done  # done status sent to the GUI
        }

    def save(self):
        # save the histogram
        os.makedirs(self.save_folder, exist_ok=True)
        with open(self.save_folder + "/winning_histogram.json", "w") as f:
            json.dump(self.history_player_winnings, f, indent=4)

        # make a leaderboard table for all time winners, and the best performing players in the
        # last 10 games.

        # 2. Generate and Save Table
        all_time, recent_top = self.generate_leaderboard_data()

        table_path = os.path.join(self.save_folder, "leaderboard.txt")
        with open(table_path, "w") as f:
            f.write(f"=== ALL-TIME LEADERBOARD ({self.game_counter}) ===\n")
            f.write(f"{'Rank':<5} | {'Player ID':<10} | {'Total Winnings':<15}\n")
            f.write("-" * 40 + "\n")
            for i, p in enumerate(all_time, 1):
                f.write(f"{i:<5} | {p['id']:<10} | {p['total']:<15.2f}\n")

            f.write("\n=== TOP PERFORMERS (LAST 10 GAMES) ===\n")
            f.write(f"{'Rank':<5} | {'Player ID':<10} | {'Avg Winnings':<15}\n")
            f.write("-" * 40 + "\n")
            for i, p in enumerate(recent_top, 1):
                f.write(f"{i:<5} | {p['id']:<10} | {p['recent_avg']:<15.2f}\n")

        print(f"{self.game_counter} - Leaderboard saved to {table_path}")