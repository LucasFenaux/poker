import random
import threading
import ray
import os
import torch

from global_settings import NUM_PLAYERS, NUM_GAMES
from player_actor import PlayerActor
from table_actor import TableActor
from leaderboard_actor import LeaderboardActor
from leaderboard_gui import LeaderboardGUI


class CasinoManager:
    def __init__(self, device: torch.device, save_folder: str = "./", discrete: bool = False):
        self.player_ids = list(range(NUM_PLAYERS))
        self.device = device
        self.save_folder = save_folder
        os.makedirs(self.save_folder, exist_ok=True)
        self.player_save_folder = os.path.join(save_folder, "players")
        os.makedirs(self.player_save_folder, exist_ok=True)
        self.players = [PlayerActor.remote(i, self.player_save_folder, device, discrete) for i in self.player_ids]
        self.table_max_size = 2
        self.table_min_size = 2

        max_tables_needed = len(self.player_ids) // self.table_min_size
        print(f"Opening casino with {max_tables_needed} permanent tables...")
        self.tables = [TableActor.remote(device) for _ in range(max_tables_needed)]   # we spin up the tables at the beginning to avoid the churn
        self.min_stack = 50
        self.max_stack = 1000
        self.min_small_blind = 1
        self.max_small_blind = 3
        self.min_bb_ratio = 1
        self.max_bb_ratio = 5
        self.min_allowed_start_bb = 10
        self.leaderboard = LeaderboardActor.remote(self.player_ids, save_folder)
        self.leaderboard_gui = None

    def _start_gui(self):
        self.leaderboard_gui = LeaderboardGUI(self.leaderboard)

    def start_casino(self):
        for i in range(NUM_GAMES):
            active_game_refs = []
            available_tables = list(self.tables)

            players_left = [player for player in self.players]
            num_players_left = len(players_left)
            while num_players_left > 0:
                table = available_tables.pop()
                # first we select the tabe size
                # print(num_players_left)
                if num_players_left <= self.table_max_size:
                    table_size = len(players_left)
                else:
                    table_size = random.randint(self.table_min_size, self.table_max_size)
                    if num_players_left - table_size < self.table_min_size:
                        table_size = self.table_min_size

                # pick the players
                players = random.sample(players_left, table_size)

                # update the list of players left
                players_left = [player for player in players_left if player not in players]
                num_players_left = len(players_left)

                # we randomly pick the stack, sb, and bb
                small_blind = random.randint(self.min_small_blind, self.max_small_blind)
                big_blind = random.randint(self.min_bb_ratio, self.max_bb_ratio) * small_blind
                starting_stacks = random.randint(max(self.min_stack, big_blind*10) , max(self.max_stack, big_blind*10))

                # spin up the game
                ray.get(table.reset.remote(players, raw_blinds_or_straddles=(small_blind, big_blind), min_bet=big_blind,
                                   raw_starting_stacks=starting_stacks, player_count=table_size))

                game_ref = table.play_game.remote()
                active_game_refs.append(game_ref)

            print(f"Game {i+1} - Booted up {len(self.tables)} tables")

            round_results = ray.get(active_game_refs)
            for player_winnings in round_results:
                self.leaderboard.update.remote(player_winnings)

            self.leaderboard.update_game_counter.remote()
            self.leaderboard.save.remote()

        self.leaderboard.set_done.remote()
        print("Casino simulation complete!")

    def start(self):
        # we need the casino thread seperate so the gui doesn't block it
        # can't launch the gui inside the thread on Mac
        casino_thread = threading.Thread(target=self.start_casino, daemon=True)
        casino_thread.start()

        self._start_gui()
        # self.start_casino()

