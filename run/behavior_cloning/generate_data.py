import ray
import os
import glob
import pickle
from queue import Empty
from ray.util.queue import Queue
from pokerkit import Automation
from src.game_registry import get_current_game_config
from src.action_interpreter import to_exact_fraction, Action
from src.state_interpreter import extract_state_snapshot
from src.baseline_model import FastBaselineBot, get_valid_actions_dict
from src.ppo_self_play.global_settings import IS_RECURRENT

CHUNK_SIZE = 100


@ray.remote(num_cpus=1)
class Table:
    default_params = {
        "small_blind": 1,
        "big_blind": 2,
        "bb_starting_stacks": 100,
        "player_count": 2,
    }
    def __init__(self, table_id, queue: Queue, model_mode: str, num_players: int):
        self.table_id = table_id
        self.queue = queue
        self.num_players = num_players
        self.current_players = num_players
        self.states = []
        self.current_actors = []
        self.actions = []
        self.sample_weights = []

        self.game_states = [[] for _ in range(self.num_players)]
        self.game_current_actors = [[] for _ in range(self.num_players)]
        self.game_actions = [[] for _ in range(self.num_players)]
        self.game_sample_weights = [[] for _ in range(self.num_players)]

        self.rewards = []
        self.baseline_bot = FastBaselineBot(0)
        self.params = None
        self.game_params = None
        self.starting_stacks = None
        self.game_starting_stacks = None
        self.stacks = None

    def reset(self, **table_params):
        self.params = table_params
        self.game_params = {**table_params}
        
        # Determine actual starting stacks depending on how the game wants it
        game_config = get_current_game_config()
        self.table_param_generator = game_config['table_param_generator']
        self.PokerkitGame = game_config['pokerkit_game']
        self.pokerkit_automations = game_config['pokerkit_automations']
        
        self.actual_game_params = self.table_param_generator(
            table_size=self.num_players,
            small_blind=self.game_params["small_blind"],
            big_blind=self.game_params["big_blind"],
            bb_starting_stacks=self.game_params["bb_starting_stacks"]
        )
        
        starting_stacks = self.actual_game_params.get("raw_starting_stacks")
        if starting_stacks is None:
            default_stack = self.game_params["bb_starting_stacks"] * self.game_params["big_blind"]
            self.stacks = [default_stack] * self.num_players
            self.starting_stacks = [default_stack] * self.num_players
        elif isinstance(starting_stacks, (int, float)):
            self.stacks = [starting_stacks] * self.num_players
            self.starting_stacks = [starting_stacks] * self.num_players
        else:
            self.stacks = starting_stacks
            self.starting_stacks = starting_stacks[:]
        
        if "raw_starting_stacks" in self.actual_game_params:
            self.actual_game_params["raw_starting_stacks"] = self.starting_stacks[:]
        self.game_starting_stacks = self.starting_stacks[:]
        self.current_players = self.num_players
        self.num_games_played = 0

        self.states = []
        self.current_actors = []
        self.actions = []
        self.sample_weights = []
        self.rewards = []

        self.game_states = [[] for _ in range(self.num_players)]
        self.game_current_actors = [[] for _ in range(self.num_players)]
        self.game_actions = [[] for _ in range(self.num_players)]
        self.game_sample_weights = [[] for _ in range(self.num_players)]

    def _play_linear_round(self):
        try:
            if "raw_starting_stacks" in self.actual_game_params:
                self.actual_game_params["raw_starting_stacks"] = self.game_starting_stacks
            state = self.PokerkitGame.create_state(
                self.pokerkit_automations,
                **self.actual_game_params
            )
            
            # Snap true initial stacks (Kuhn may override requested stacks to 2)
            true_initial_stacks = list(state.stacks)

            while state.status:
                current_actor: int = state.actor_index

                snapshot = extract_state_snapshot(state, current_actor)

                self.baseline_bot.update_index(current_actor)
                valid_actions = get_valid_actions_dict(state)
                action, amt = self.baseline_bot.get_action(state, valid_actions)
                action: Action
                amt = to_exact_fraction(amt)

                min_bet = state.min_completion_betting_or_raising_to_amount
                if min_bet is None:
                    min_bet = max(state.bets)

                max_bet = state.max_completion_betting_or_raising_to_amount
                if max_bet is None:
                    max_bet = min_bet  # Or some other logical fallback

                amt = min(max(amt, min_bet), max_bet)

                self.game_states[current_actor].append(snapshot)
                self.game_current_actors[current_actor].append(current_actor)
                self.game_actions[current_actor].append((action, amt))
                self.game_sample_weights[current_actor].append(1.)

                self._take_action(state, action, amt)

            # compute rewards
            final_game_stacks = state.stacks[:]
            busted_out = []
            
            blinds_or_straddles = self.actual_game_params.get("raw_blinds_or_straddles")
            bb_amount = blinds_or_straddles[-1] if blinds_or_straddles else self.actual_game_params.get("min_bet", 1)
            
            for i, (final_game_stack, game_starting_stack) in enumerate(zip(final_game_stacks,
                                                                            true_initial_stacks)):
                reward = float(final_game_stack - game_starting_stack)/float(bb_amount)

                # save the hand info
                hand_rewards = [reward] * len(self.game_states[i])

                if IS_RECURRENT:
                    # Keep sequences intact for RNN
                    self.states.append(self.game_states[i])
                    self.current_actors.append(self.game_current_actors[i])
                    self.actions.append(self.game_actions[i])
                    self.sample_weights.append(self.game_sample_weights[i])
                    self.rewards.append(hand_rewards)
                else:
                    # Flatten sequences for standard PPO (Original Code)
                    self.states.extend(self.game_states[i])
                    self.current_actors.extend(self.game_current_actors[i])
                    self.actions.extend(self.game_actions[i])
                    self.sample_weights.extend(self.game_sample_weights[i])
                    self.rewards.extend(hand_rewards)

                if final_game_stack < self.actual_game_params.get("min_bet", 1):
                    # player is busted out
                    # update game view to remove the busted out player
                    busted_out.append(i)

            busted_out.sort(reverse=True)   # bust from largest index to lowest index so we don't change indices as we bust

            # bust out players
            for i in busted_out:
                final_game_stacks.pop(i)
                self.game_starting_stacks.pop(i)
                self.current_players -= 1

            # update the player game stacks
            self.game_starting_stacks = final_game_stacks

            self.game_states = [[] for _ in range(self.num_players)]
            self.game_current_actors = [[] for _ in range(self.num_players)]
            self.game_actions = [[] for _ in range(self.num_players)]
            self.game_sample_weights = [[] for _ in range(self.num_players)]

            if self.current_players < 2:
                return True

            # rotate the spots
            self.game_starting_stacks.append(self.game_starting_stacks.pop(0))

            return False

        except Exception as e:
            print(f"Exception: {e} encountered in Table {self.table_id} in linear round fn")
            return True  # terminate the table to avoid players from getting stuck in the void

    def play_game(self):
        try:
            done = False
            counter = 0
            while not done:
                done = self._play_linear_round()
                counter += 1

            self.num_games_played += 1
            return True

        except Exception as e:
            print(f"Exception: {e} encountered in Table {self.table_id} in play game fn")
            return False

    def _take_action(self, state, action: Action, amt):
        # we convert the action into something we can use
        if action == Action.CHECK_OR_FOLD:
            state.fold()
        elif action == Action.CHECK_OR_CALL:
            state.check_or_call()
        elif action == Action.RAISE:
            state.complete_bet_or_raise_to(amt)
        else:
            raise NotImplementedError

    def start(self):
        while True:
            self.reset(**self.default_params)
            success = self.play_game()

            # Send back results
            data = {
                "states": self.states,
                "current_actors": self.current_actors,
                "actions": self.actions,
                "sample_weights": self.sample_weights,
                "rewards": self.rewards,
            }

            self.queue.put(data)


def generate_data(save_folder: str, num_games=100_000, num_workers=4, model_mode="baseline", num_players=2):
    """
    Starts Ray workers to simulate poker games using individual queues per worker.
    """
    # 1. Initialize Ray
    if not ray.is_initialized():
        ray.init(ignore_reinit_error=True)

    # 2. Prepare the save directory & check for existing chunks
    os.makedirs(save_folder, exist_ok=True)
    existing_chunks = glob.glob(os.path.join(save_folder, "chunk_*.pkl"))
    chunk_index = len(existing_chunks)
    games_saved = chunk_index * CHUNK_SIZE

    print(f"Found {chunk_index} existing chunks. {games_saved}/{num_games} games already saved.")

    if games_saved >= num_games:
        print("Target number of games already reached!")
        return

    # 3. Create individual queues for each table
    # Limiting maxsize prevents memory bloat if the main loop saves slower than generation
    queues = [Queue(maxsize=10) for _ in range(num_workers)]

    # Start tables, assigning each its specific queue
    tables = [
        Table.remote(table_id=i, queue=queues[i], model_mode=model_mode, num_players=num_players)
        for i in range(num_workers)
    ]

    for table in tables:
        table.start.remote()

    # 4. Prepare variables for the aggregation loop
    games_to_collect = num_games - games_saved
    games_collected_in_current_chunk = 0
    total_collected_this_run = 0

    chunk_data = {
        "states": [],
        "current_actors": [],
        "actions": [],
        "sample_weights": [],
        "rewards": []
    }

    try:
        print("Starting data collection...")
        queue_index = 0

        while total_collected_this_run < games_to_collect:
            try:
                # 5. Non-blocking/timeout fetch from the current queue
                game_data = queues[queue_index].get_nowait()

                # Append the newly pulled game data
                for key in chunk_data:
                    chunk_data[key].extend(game_data[key])

                games_collected_in_current_chunk += 1
                total_collected_this_run += 1

                # 6. Save the chunk if we've hit CHUNK_SIZE or finished the quota
                if games_collected_in_current_chunk == CHUNK_SIZE or total_collected_this_run == games_to_collect:
                    save_path = os.path.join(save_folder, f"chunk_{chunk_index}.pkl")

                    with open(save_path, "wb") as f:
                        pickle.dump(chunk_data, f)

                    print(
                        f"Saved {save_path} | Total Progress: {games_saved + total_collected_this_run}/{num_games} games")

                    # Reset for the next chunk
                    chunk_index += 1
                    games_collected_in_current_chunk = 0
                    chunk_data = {key: [] for key in chunk_data}

            except Empty:
                # The current queue didn't have data ready. We just silently pass and check the next one.
                pass

            # Rotate to the next queue index (Round-robin)
            queue_index = (queue_index + 1) % num_workers

    except KeyboardInterrupt:
        print("\nProcess interrupted by user. Shutting down gracefully...")

    finally:
        # 7. Clean up the background workers
        for table in tables:
            ray.kill(table)
        print("Data generation process concluded.")


if __name__ == '__main__':
    from src.ppo_self_play.global_settings import GAME_TYPE
    save_folder = f"./data/{GAME_TYPE}_{'rnn' if IS_RECURRENT else 'no_mem'}/"
    generate_data(save_folder, num_workers=4)
