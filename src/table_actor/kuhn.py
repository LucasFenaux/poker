import ray
from ray.util.queue import Queue
from pokerkit import Automation
from .holdem import HoldemTable
import traceback
from src.global_settings import IS_RECURRENT
from src.state_interpreter import extract_state_snapshot

class KuhnTable(HoldemTable):
    def __init__(self, table_id, device, in_queue: Queue, out_queue: Queue,
                 historical_sampling_receive_queue: Queue,
                 max_table_size: int, discrete: bool,
                 model_mode: str, batch_size: int, log_folder: str):
        super().__init__(table_id, device, in_queue, out_queue, historical_sampling_receive_queue,
                         max_table_size, discrete, model_mode, batch_size, log_folder)
        self.replay = 100   # since games are so quick

    def _play_tree_round(self):
        from pokerkit import KuhnPoker
        try:
            state = KuhnPoker.create_state(
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
                )
            )
            with self.timer.time("TreeRound_Play_Level_Recursive"):
                snapshots, current_actors, player_actions, sample_weights, rewards, expected_rewards = self._play_tree_level(
                    state, 0)

            with self.timer.time("TreeRound_Process_Bustouts"):
                busted_out = []
                final_game_stacks = []
                for i, player_id in enumerate(self.game_player_ids):
                    reward_in_bbs = expected_rewards[player_id]
                    chip_delta = reward_in_bbs * self.game_params.get("raw_blinds_or_straddles", [1])[-1]
                    final_game_stack = self.game_starting_stacks[i] + chip_delta
                    final_game_stacks.append(final_game_stack)

                    self.hand_info[player_id]["states"].extend(snapshots[player_id])
                    self.hand_info[player_id]["current_actors"].extend(current_actors[player_id])
                    self.hand_info[player_id]["actions"].extend(player_actions[player_id])
                    self.hand_info[player_id]["rewards"].extend(rewards[player_id])
                    self.hand_info[player_id]["sample_weights"].extend(sample_weights[player_id])

                    if final_game_stack < self.game_params.get("min_bet", 1):
                        busted_out.append(i)

                busted_out.sort(reverse=True)

                for i in busted_out:
                    player_id = self.game_player_ids[i]
                    j = self.player_ids.index(player_id)
                    self.player_winnings[player_id] += final_game_stacks[i] - self.starting_stacks[j]

                    final_game_stacks.pop(i)
                    self.game_players.pop(i)
                    self.game_starting_stacks.pop(i)
                    self.game_player_ids.pop(i)

                    self.game_params["player_count"] = len(self.game_player_ids)

                button_busted = (0 in busted_out)
                self.game_starting_stacks = final_game_stacks

                if self.game_params["player_count"] < 2:
                    return True

                if not button_busted:
                    self.game_players.append(self.game_players.pop(0))
                    self.game_player_ids.append(self.game_player_ids.pop(0))
                    self.game_starting_stacks.append(self.game_starting_stacks.pop(0))

            return False
        except Exception as e:
            print(f"Exception: {e} encountered in Table {self.table_id} in tree round fn")
            if self.table_id == 0:
                traceback.print_exc()
            return True

    def _play_linear_round(self):
        from pokerkit import KuhnPoker
        try:
            state = KuhnPoker.create_state(
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
                )
            )

            while state.status:
                current_actor = state.actor_index
                player = self.game_players[current_actor]
                player_id = self.game_player_ids[current_actor]
                snapshot = extract_state_snapshot(state, current_actor)

                if IS_RECURRENT:
                    hand_memory = self.hand_memories[player_id]
                    game_memory = self.game_memories[player_id]
                else:
                    hand_memory = None
                    game_memory = None

                try:
                    player_action, new_hand_memory = self._get_action(player, snapshot, current_actor,
                                                                      hand_hidden=hand_memory,
                                                                      game_hidden=game_memory)
                except Exception as e:
                    print("linear_round", state)
                    if self.table_id == 0:
                        traceback.print_exc()
                    raise e

                self.current_hand[player_id]["new_hands"].append(self.hands_since_last_action[player_id])
                self.hands_since_last_action[player_id] = 0

                self.current_hand[player_id]["states"].append(snapshot)
                self.current_hand[player_id]["current_actors"].append(current_actor)
                self.current_hand[player_id]["actions"].append(player_action)
                self.current_hand[player_id]["sample_weights"].append(1.)
                self.current_hand[player_id]["hand_memories"].append(hand_memory)

                self._take_action(state, player_action)
                self.hand_memories[player_id] = new_hand_memory
            if IS_RECURRENT:
                for player, player_id in zip(self.game_players, self.game_player_ids):
                    game_memory = self.game_memories[player_id]
                    last_hand_memory = self.hand_memories[player_id]
                    # type: ignore
                    new_game_memory = player.update_game_memory(last_hand_memory, game_memory)
                    self.current_hand[player_id]["game_memories"] = game_memory
                    self.game_memories[player_id] = new_game_memory

            true_starting_stacks = getattr(state, "starting_stacks", None)
            if true_starting_stacks is None:
                true_starting_stacks = [2] * len(self.game_player_ids)

            state_final_stacks = state.stacks[:]
            final_game_stacks = []
            busted_out = []

            for i, (state_final_stack, initial_stack, game_starting_stack) in enumerate(
                    zip(state_final_stacks, true_starting_stacks, self.game_starting_stacks)):
                chip_delta = state_final_stack - initial_stack
                reward = chip_delta  # No blinds in Kuhn
                new_game_stack = game_starting_stack + chip_delta
                final_game_stacks.append(new_game_stack)

                player_id = self.game_player_ids[i]
                hand_rewards = [reward] * len(self.current_hand[player_id]["states"])
                self.hand_info[player_id]["states"].extend(self.current_hand[player_id]["states"])
                self.hand_info[player_id]["current_actors"].extend(self.current_hand[player_id]["current_actors"])
                self.hand_info[player_id]["actions"].extend(self.current_hand[player_id]["actions"])
                self.hand_info[player_id]["rewards"].extend(hand_rewards)
                self.hand_info[player_id]["sample_weights"].extend(self.current_hand[player_id]["sample_weights"])
                self.hand_info[player_id]["hand_memories"].extend(self.current_hand[player_id]["hand_memories"])
                self.hand_info[player_id]["game_memories"].extend([self.current_hand[player_id]["game_memories"], ])

                self.hand_info[player_id]["new_hands"].extend(self.current_hand[player_id]["new_hands"])

                if new_game_stack < self.game_params.get("min_bet", 1):
                    busted_out.append(i)

            busted_out.sort(reverse=True)

            for i in busted_out:
                player_id = self.game_player_ids[i]
                j = self.player_ids.index(player_id)
                self.player_winnings[player_id] += final_game_stacks[i] - self.starting_stacks[j]

                final_game_stacks.pop(i)
                self.game_players.pop(i)
                self.game_starting_stacks.pop(i)
                self.game_player_ids.pop(i)

                self.game_params["player_count"] = len(self.game_player_ids)

            self.game_starting_stacks = final_game_stacks

            if self.game_params["player_count"] < 2:
                return True

            self.game_players.append(self.game_players.pop(0))
            self.game_player_ids.append(self.game_player_ids.pop(0))
            self.game_starting_stacks.append(self.game_starting_stacks.pop(0))

            return False

        except Exception as e:
            print(f"Exception: {e} encountered in Table {self.table_id} in linear round fn")
            if self.table_id == 0:
                traceback.print_exc()
            return True


@ray.remote(num_cpus=0)
class KuhnTableActor(KuhnTable):
    pass