import ray
from ray.util.queue import Queue, Empty
import json
import os
import asyncio
from fractions import Fraction
from src.ppo_self_play.global_settings import NUM_TRAINERS, NUM_TABLES


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
    def __init__(self, queue: Queue, table_send_queue: Queue, table_receive_queue: Queue,
                 trainer_send_queue: Queue, trainer_receive_queue: Queue,
                 player_ids, save_folder: str = "./"):
        self.queue = queue
        self.save_folder = save_folder
        self.history_player_winnings = {player_id: [] for player_id in player_ids}
        self.player_ids = player_ids
        self.recent_avg_lookback = 100
        # New per-player game counter
        self.number_games_played = {player_id: 0 for player_id in player_ids}
        self.is_done = False
        self.last_saved_avg = -1
        self.table_send_queue = table_send_queue
        self.table_receive_queue = table_receive_queue
        self.trainer_send_queue = trainer_send_queue
        self.trainer_receive_queue = trainer_receive_queue
        self.num_tables = 0
        self.num_trainers = 0
        self.target_num_tables = None
        self.target_num_trainers = None
        self.is_playing = {}
        self.is_training = {}
        self.is_playing_against = {}
        self.player_dispatch_times = {}

    async def start(self):
        while not self.is_done:
            try:
                # Use block=False and an await so we don't freeze the actor!
                # This allows the GUI's .remote() calls to be processed.
                # data = await asyncio.wait_for(self.queue.get_async(), timeout=1.0)
                data = self.queue.get_nowait()

                if data is not None:
                    # print("updating")
                    (player_id, player_winnings, num_tables, num_trainers, is_playing, is_training, is_playing_against,
                     player_dispatch_times) = data
                    self.num_tables = num_tables
                    self.num_trainers = num_trainers
                    self.is_playing = is_playing
                    self.is_training = is_training
                    self.is_playing_against = is_playing_against
                    self.player_dispatch_times = player_dispatch_times

                    self.update(player_id, player_winnings)

                await asyncio.sleep(0)
            # except (asyncio.TimeoutError, TimeoutError):  # <--- Fixed exception type!
            #     continue
            except Empty:
                await asyncio.sleep(0.05)

        # Use asyncio.sleep instead of time.sleep in an async method
        await asyncio.sleep(10)

        # ---> NEW: Methods to handle target counts and resets

    async def set_target_tables(self, target: int):
        # Initialize target to actual on the very first adjustment
        if self.target_num_tables is None:
            self.target_num_tables = self.num_tables

        diff = target - self.target_num_tables
        self.target_num_tables = target

        if diff > 0:
            for _ in range(diff):
                await self.request_table_creation()
        elif diff < 0:
            for _ in range(abs(diff)):
                await self.request_table_removal()

    async def set_target_trainers(self, target: int):
        if self.target_num_trainers is None:
            self.target_num_trainers = self.num_trainers

        diff = target - self.target_num_trainers
        self.target_num_trainers = target

        if diff > 0:
            for _ in range(diff):
                await self.request_trainer_creation()
        elif diff < 0:
            for _ in range(abs(diff)):
                await self.request_trainer_removal()

    async def reset_to_defaults(self):
        await self.set_target_tables(NUM_TABLES)
        await self.set_target_trainers(NUM_TRAINERS)

    async def request_table_removal(self):
        message = {
            "type": "message",
            "terminate": True
        }
        await self.table_send_queue.put_async(message)

    async def request_table_creation(self):
        message = {
            "type": "creation"
        }
        await self.table_receive_queue.put_async(message)

    async def request_trainer_removal(self):
        message = {
            "type": "message",
            "terminate": True
        }
        await self.trainer_send_queue.put_async(message)

    async def request_trainer_creation(self):
        message = {
            "type": "creation"
        }
        await self.trainer_receive_queue.put_async(message)

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
        if current_avg_int // 10 > self.last_saved_avg // 10:  # buffer read/write to every 10 games
            self.save()
            self.last_saved_avg = current_avg_int

    def generate_leaderboard_data(self):
        import time  # Ensure time is available locally
        current_time = time.time()
        stats = []
        for p_id in self.player_ids:
            history = self.history_player_winnings[p_id]

            total_winnings = sum(history)
            # Slice the last 100 games and calculate average
            recent_history = history[-self.recent_avg_lookback:] if len(history) >= self.recent_avg_lookback else history
            recent_avg = sum(recent_history) / len(recent_history) if recent_history else 0

            # Anomaly detection
            is_p = self.is_playing.get(p_id, False)
            is_t = self.is_training.get(p_id, False)
            vs = self.is_playing_against.get(p_id, [])

            appearance_count = sum(1 for opponents in self.is_playing_against.values() if p_id in opponents)

            # 1. Check if both Playing and Training
            if is_p and is_t:
                print(f"🚨 [ANOMALY ERROR] Player {p_id} is simultaneously PLAYING and TRAINING!")

            # 2. Check if at multiple tables (Opponents > Max Table Size - 1)
            if is_p and appearance_count > len(vs):
                print(f"🚨 [ANOMALY ERROR] Player {p_id} is at MULTIPLE TABLES! They see {len(vs)} opponents, "
                      f"but are seen by {appearance_count} players.")

            # 3. Check if Training but somehow has opponents
            if is_t and appearance_count > 0:
                print(f"🚨 [ANOMALY ERROR] Player {p_id} is TRAINING but seen as an opponent by {appearance_count} players! ({vs})")

            # 4. Check if Waiting but somehow has opponents
            if not is_p and not is_t and appearance_count > 0:
                print(f"🚨 [ANOMALY ERROR] Player {p_id} is WAITING but still seen as an opponent by {appearance_count} players! ({vs})")

            # Determine the player's current status and elapsed time
            status = "Waiting"
            if self.is_playing.get(p_id):
                elapsed = int(current_time - self.player_dispatch_times.get(p_id, current_time))
                vs_players = self.is_playing_against.get(p_id, [])
                status = f"Playing vs {vs_players} ({elapsed}s)"
            elif self.is_training.get(p_id):
                elapsed = int(current_time - self.player_dispatch_times.get(p_id, current_time))
                status = f"Training ({elapsed}s)"

            stats.append({
                "id": p_id,
                "total": total_winnings,
                "game_count": self.number_games_played[p_id],
                "recent_avg": recent_avg,
                "status": status  # <-- Add the status to the payload
            })

        # Sort by total winnings descending for the main board
        all_time = sorted(stats, key=lambda x: x['total'], reverse=True)
        # Sort by recent performance descending
        recent_top = sorted(stats, key=lambda x: x['recent_avg'], reverse=True)

        return all_time, recent_top

    def get_leaderboard_stats(self):
        all_time, recent_top = self.generate_leaderboard_data()
        total_games = sum(self.number_games_played.values())
        avg_games = total_games / max(1, len(self.player_ids))

        num_playing = sum(1 for v in self.is_playing.values() if v)
        num_training = sum(1 for v in self.is_training.values() if v)
        num_waiting = len(self.player_ids) - num_playing - num_training

        return {
            "all_time": all_time,
            "recent": recent_top,
            "avg_games": avg_games,
            "is_done": self.is_done,
            "num_players": len(self.player_ids),
            "num_waiting": num_waiting,
            "num_playing": num_playing,
            "num_training": num_training,
            "num_tables": self.num_tables,
            "num_trainers": self.num_trainers,
            "target_tables": self.target_num_tables if self.target_num_tables is not None else self.num_tables,
            "target_trainers": self.target_num_trainers if self.target_num_trainers is not None else self.num_trainers
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