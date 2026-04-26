import pickle
from collections import deque
import pokerkit
import ray
from typing import Union
from ray.util.queue import Queue, Empty
import os
from torch.utils.tensorboard import SummaryWriter
from pokerkit import NoLimitTexasHoldem, Automation
import random
import torch
from src.action_interpreter import ActionInterpreter, Action
from src.state_interpreter import extract_state_snapshot
from src.ppo_self_play.alg import PPOInferenceWrapper, PPO
from src.player_ai import PlayerAI, RNNPlayerAI
from src.shared import SemanticTimer
from src.ppo_self_play.global_settings import IS_RECURRENT
from src.ppo_self_play.alg import RNNPPOInferenceWrapper


@ray.remote(num_cpus=0)
class TableActor:
    default_params = {
        "ante_trimming_status": True,
        "raw_antes": 0,
        "raw_blinds_or_straddles": (1, 2),
        "min_bet": 2,
        "raw_starting_stacks": 200,   # 200 for 100 BB
        "player_count": 2,
        "mode": "tree"
    }

    def __init__(self, table_id, device, in_queue: Queue, out_queue: Queue, max_table_size: int, discrete: bool,
                 model_mode: str, batch_size: int, log_folder: str):

        self.table_id = table_id
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.device = device
        self.discrete = discrete
        self.max_table_size = max_table_size
        self.action_interpreter: ActionInterpreter = ActionInterpreter(model_mode)
        self.model_mode = model_mode
        self.num_games_played = 0
        # number of games to play
        # when players are bad, games are quick, can have it between 1 and 20. More than that could get weird
        # for deep stacks and lots of players as each game would mean more hands played per game ->  less table
        # variety per batch -> lower quality data
        self.replay = 1  # not really necessary with tree game. Useful with linear game.
        self.tree_expansion = 3   # good options are 3, 4, 5
        self.use_early_stopping = True
        self.batch_size = batch_size
        self.log_folder = log_folder
        log_path = os.path.join(self.log_folder, "tensorboard_logs")
        self.writer = SummaryWriter(log_dir=log_path)
        self.timer = SemanticTimer()

        # for every parameter, we have an initial version and a game state view version as the game state evolves
        self.player_ids = None   # table facing view
        self.game_player_ids = None   # game state facing view
        self.players = [PPOInferenceWrapper(PPO.init_networks(self.device, self.discrete, self.model_mode), self.discrete)
                        for _ in range(max_table_size)]
        self.game_players = self.players[:]
        self.trainable_players = None
        self.params = None
        self.game_params = None
        self.starting_stacks = None
        self.game_starting_stacks = None
        self.stacks = None

        self.current_player_versions = None
        self.hand_info = None
        self.current_hand = None
        self.player_winnings: dict[int, float] = None
        self.mode = None
        self._play_round = None

        # for recurrent models
        self.hand_memories = None
        self.game_memories = None

    def reset(self, players: list[Union[PlayerAI, RNNPlayerAI]], player_ids, **table_params):
        self.player_ids = player_ids
        self.game_player_ids = self.player_ids[:]
        players_params_list = [player.get_params() for player in players]
        self.trainable_players = players
        # if recurrent, we init the game and hand memories
        if IS_RECURRENT:
            for player in players:
                assert isinstance(player, RNNPlayerAI)
            self.game_memories = [player.init_game_memory(batch_size=1) for player in players]
            self.hand_memories = [player.init_hand_memory(batch_size=1) for player in players]

        self.player_winnings = {player_id: 0.0 for player_id in self.player_ids}

        self.players = [PPOInferenceWrapper(PPO.init_networks(self.device, self.discrete, self.model_mode), self.discrete)
                        for _ in range(len(player_ids))]
        self.game_players = self.players[:]

        for i, (player, player_params) in enumerate(zip(self.players, players_params_list)):
            player.load_params(player_params)
            player.to(self.device)

        # we need to add the default params for the ones not explicitly specified
        for param_name, param_value in self.default_params.items():
            if param_name not in table_params:
                table_params[param_name] = param_value

        self.params = table_params
        self.game_params = {**table_params}

        starting_stacks = self.game_params.pop("raw_starting_stacks")
        if isinstance(starting_stacks, int):
            self.stacks = [starting_stacks] * len(self.players)
            self.starting_stacks = [starting_stacks] * len(self.players)  # will use it to compute game winnings
        else:
            self.stacks = starting_stacks
            self.starting_stacks = starting_stacks[:]  # will use it to compute game winnings

        self.game_starting_stacks = self.starting_stacks[:]

        self.mode = self.game_params.pop("mode")
        if IS_RECURRENT:
            # only support linear mode for recurrent models, since it requires to keep the sequential aspect.
            # TODO: think to see if tree is compatible. Maybe by duplicating previous steps to keep parallel linear paths rather than branches at storage time.
            # TODO: [continue] could set all sample_weights to 1, but use the expected reward rather than the specific path reward to still help with variance
            self.mode = "linear"

        if self.mode == "linear":
            self._play_round = self._play_linear_round
        elif self.mode == "tree":
            self._play_round = self._play_tree_round
        else:
            raise NotImplementedError(self.mode)

        self._reset_hand_info()
            
    def _reset_hand_info(self):
        self.hand_info = {}
        for player_id in self.player_ids:  # only care about current players
            self.hand_info[player_id] = {
                "states": [],
                "current_actors": [],
                "actions": [],
                "rewards": [],
                "sample_weights": [],
                "hand_memories": [],
                "game_memories": []
            }
        self._reset_current_hand()

    def _reset_current_hand(self):
        self.current_hand = {}
        for player_id in self.game_player_ids:  # only care about current players
            self.current_hand[player_id] = {
                "states": [],
                "current_actors": [],
                "actions": [],
                "sample_weights": [],
                "hand_memories": [],
                "game_memories": []
            }
        if IS_RECURRENT:
            # we reset the hand memories
            self.hand_memories = [player.init_hand_memory(batch_size=1) for player in self.trainable_players]

    def _take_action(self, state, player_action):
        # we convert the action into something we can use
        min_bet = state.min_completion_betting_or_raising_to_amount
        if min_bet is None:
            min_bet = max(state.bets)

        max_bet = state.max_completion_betting_or_raising_to_amount
        if max_bet is None:
            max_bet = min_bet  # Or some other logical fallback

        interpreted_action, bet_sizing = self.action_interpreter(player_action, min_bet, max_bet)

        if interpreted_action == Action.CHECK_OR_FOLD:
            if state.can_check_or_call() and state.checking_or_calling_amount == 0:
                state.check_or_call()
            elif state.can_fold():
                state.fold()
            else:
                raise RuntimeError("No legal action in CHECK_OR_FOLD branch")
        elif interpreted_action == Action.CHECK_OR_CALL:
            if state.can_check_or_call():
                state.check_or_call()
            elif state.can_fold():
                # we can't actually check or call so we have to fold
                state.fold()
            else:
                raise RuntimeError("No legal fallback from CHECK_OR_CALL")
        elif interpreted_action == Action.RAISE:
            # If the bet is invalid, force it into legal bounds
            legal_min = state.min_completion_betting_or_raising_to_amount
            legal_max = state.max_completion_betting_or_raising_to_amount

            # If they can legally raise, clamp their bet to the legal window
            if legal_min is not None and legal_max is not None:
                # Snap the sizing to valid boundaries
                clamped_bet = max(legal_min, min(bet_sizing, legal_max))
                state.complete_bet_or_raise_to(clamped_bet)
            elif state.can_check_or_call():
                state.check_or_call()  # Legitimate fallback (e.g. facing an all-in)
            elif state.can_fold():
                state.fold()
            else:
                raise RuntimeError("No legal fallback from RAISE")
        else:
            raise RuntimeError("No legal fallback")

    def _play_tree_level(self, state: pokerkit.State, depth):
        assert depth < 4
        snapshots = {player_id: [] for player_id in self.game_player_ids}
        current_actors = {player_id: [] for player_id in self.game_player_ids}
        player_actions = {player_id: [] for player_id in self.game_player_ids}
        sample_weights = {player_id: [] for player_id in self.game_player_ids}
        rewards = {player_id: [] for player_id in self.game_player_ids}

        with self.timer.time("TreeLevel_Play_Street"):
            while state.status and not state.can_deal_board():
                # play this level until the next street
                current_actor = state.actor_index
                player = self.game_players[current_actor]
                player_id = self.game_player_ids[current_actor]
                snapshot = extract_state_snapshot(state, current_actor)

                try:
                    with torch.no_grad():
                        player_action_tensor = player.get_action((snapshot, current_actor))
                        player_action = player_action_tensor.detach().cpu().squeeze(0)
                except Exception as e:
                    # print(state)
                    print("tree_level", e)
                    raise e

                # we log the state and action for player training
                snapshots[player_id].append(snapshot)
                current_actors[player_id].append(current_actor)
                player_actions[player_id].append(player_action)
                sample_weights[player_id].append(1./(self.tree_expansion ** depth))

                self._take_action(state, player_action)

        if not state.status:
            with self.timer.time("TreeLevel_Compute_Rewards"):
                # we reached the last street level, we can compute the rewards
                final_game_stacks = state.stacks[:]
                expected_rewards = {}  # Store scalar expected values safely
                for i, (final_game_stack, game_starting_stack) in enumerate(zip(final_game_stacks,
                                                                                self.game_starting_stacks)):
                    reward = (final_game_stack - game_starting_stack) / (self.game_params["raw_blinds_or_straddles"][-1])
                    player_id = self.game_player_ids[i]
                    hand_rewards = [reward] * len(snapshots[player_id])
                    rewards[player_id] = hand_rewards
                    expected_rewards[player_id] = reward  # Safe scalar value
                return snapshots, current_actors, player_actions, sample_weights, rewards, expected_rewards

        current_level_counts = {player_id: len(snapshots[player_id]) for player_id in self.game_player_ids}

        with self.timer.time("TreeLevel_State_Deepcopy"):
            pickle_state = pickle.dumps(state, protocol=-1)
            children_states = [pickle.loads(pickle_state) for _ in range(self.tree_expansion - 1)]
            children_states.append(state)  # Saves 1 expensive deepcopy per node

        # we create the children
        # children_states = [copy.deepcopy(state) for _ in range(self.tree_expansion)]
        level_rewards = {player_id: 0 for player_id in self.game_player_ids}
        for i, child_state in enumerate(children_states):
            # we need to shuffle the deck to make sure its different from the main deck
            if i != 0 :  # we can save a shuffle
                cards_list = list(child_state.deck_cards)
                random.shuffle(cards_list)
                child_state.deck_cards = deque(cards_list)
            # deal the board cards to make the street index move forward
            child_state.deal_board()
            (child_snapshots, child_current_actors, child_player_actions, child_sample_weights, child_rewards,
             child_expected_rewards) = self._play_tree_level(child_state, depth + 1)

            # we assume that everything is already in the correct order, so we can use the 0 element to get the child's
            # reward
            for player_id in self.game_player_ids:
                level_rewards[player_id] += child_expected_rewards[player_id]

                snapshots[player_id].extend(child_snapshots[player_id])
                current_actors[player_id].extend(child_current_actors[player_id])
                player_actions[player_id].extend(child_player_actions[player_id])
                sample_weights[player_id].extend(child_sample_weights[player_id])
                rewards[player_id].extend(child_rewards[player_id])

        for player_id in self.game_player_ids:
            level_rewards[player_id] /= len(children_states)  # average between children
            # prepend the rewards
            rewards[player_id] = [level_rewards[player_id]] * current_level_counts[player_id] + rewards[player_id]

        return snapshots, current_actors, player_actions, sample_weights, rewards, level_rewards

    def _play_tree_round(self):
        try:
            state = NoLimitTexasHoldem.create_state(
                (
                    Automation.ANTE_POSTING,
                    Automation.BET_COLLECTION,
                    Automation.BLIND_OR_STRADDLE_POSTING,
                    Automation.CARD_BURNING,
                    Automation.HOLE_DEALING,
                    # Automation.BOARD_DEALING,
                    Automation.HOLE_CARDS_SHOWING_OR_MUCKING,
                    Automation.HAND_KILLING,
                    Automation.CHIPS_PUSHING,
                    Automation.CHIPS_PULLING,
                ),
                raw_starting_stacks=self.game_starting_stacks,
                **self.game_params
            )
            with self.timer.time("TreeRound_Play_Level_Recursive"):
                snapshots, current_actors, player_actions, sample_weights, rewards, expected_rewards = self._play_tree_level(state, 0)

            with self.timer.time("TreeRound_Process_Bustouts"):
                # for the reward to update the overarching game state (and leaderboard), we simply use the mean reward
                # over all rollouts, which should be given to use by the first reward in the list for each player
                busted_out = []
                final_game_stacks = []
                for i, player_id in enumerate(self.game_player_ids):
                    reward_in_bbs = expected_rewards[player_id]
                    chip_delta = reward_in_bbs * self.game_params["raw_blinds_or_straddles"][-1]
                    final_game_stack = self.game_starting_stacks[i] + chip_delta
                    final_game_stacks.append(final_game_stack)

                    self.hand_info[player_id]["states"].extend(snapshots[player_id])
                    self.hand_info[player_id]["current_actors"].extend(current_actors[player_id])
                    self.hand_info[player_id]["actions"].extend(player_actions[player_id])
                    self.hand_info[player_id]["rewards"].extend(rewards[player_id])
                    self.hand_info[player_id]["sample_weights"].extend(sample_weights[player_id])

                    if final_game_stack < self.game_params["min_bet"]:
                        # player is busted out
                        # update game view to remove the busted out player
                        busted_out.append(i)

                busted_out.sort(reverse=True)   # bust from largest index to lowest index so we don't change indices as we bust

                for i in busted_out:
                    player_id = self.game_player_ids[i]

                    j = self.player_ids.index(player_id)
                    self.player_winnings[player_id] += final_game_stacks[i] - self.starting_stacks[j]  # we look at the game starting stacks, not the hand starting stacks

                    final_game_stacks.pop(i)
                    self.game_players.pop(i)
                    self.game_starting_stacks.pop(i)
                    self.game_player_ids.pop(i)

                    self.game_params["player_count"] = len(self.game_player_ids)

                self.game_starting_stacks = final_game_stacks

                # Keep track of if the current button busted
                button_busted = (0 in busted_out)

                if self.game_params["player_count"] < 2:
                    return True

                if not button_busted:
                    # rotate the spots
                    self.game_players.append(self.game_players.pop(0))
                    self.game_player_ids.append(self.game_player_ids.pop(0))
                    self.game_starting_stacks.append(self.game_starting_stacks.pop(0))

            return False
        except Exception as e:
            # raise e
            print(f"Exception: {e} encountered in Table {self.table_id} in tree round fn")
            return True

    def _play_linear_round(self):
        try:
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
                raw_starting_stacks=self.game_starting_stacks,
                **self.game_params
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
                    player_action, new_hand_memory = self._get_action(player, snapshot, current_actor, hand_hidden=hand_memory,
                                                     game_hidden=game_memory)
                except Exception as e:
                    print("linear_round", state)
                    raise e

                # we log the state and action for player training
                self.current_hand[player_id]["states"].append(snapshot)
                self.current_hand[player_id]["current_actors"].append(current_actor)
                self.current_hand[player_id]["actions"].append(player_action)
                self.current_hand[player_id]["sample_weights"].append(1.)
                self.current_hand[player_id]["hand_memory"].append(hand_memory)
                # self.current_hand[player_id]["game_memory"].append(game_memory)

                self._take_action(state, player_action)
                self.hand_memories[player_id] = new_hand_memory
            if IS_RECURRENT:
                # update the game state
                for player, player_id in zip(self.game_players, self.game_player_ids):
                    game_memory = self.game_memories[player_id]
                    last_hand_memory = self.hand_memories[player_id]
                    player: RNNPPOInferenceWrapper
                    new_game_memory = player.update_game_memory(last_hand_memory, game_memory)
                    self.current_hand[player_id]["game_memory"] = game_memory  # we only update it once per hand so only need to store one instance
                    self.game_memories[player_id] = new_game_memory

            # compute rewards
            final_game_stacks = state.stacks[:]
            busted_out = []
            for i, (final_game_stack, game_starting_stack) in enumerate(zip(final_game_stacks,
                                                                            self.game_starting_stacks)):
                reward = (final_game_stack - game_starting_stack)/(self.game_params["raw_blinds_or_straddles"][-1])

                # save the hand info
                player_id = self.game_player_ids[i]
                hand_rewards = [reward] * len(self.current_hand[player_id]["states"])
                if IS_RECURRENT:
                    # we want to keep the hand structure so we append rather than extend
                    self.hand_info[player_id]["states"].append(self.current_hand[player_id]["states"])
                    self.hand_info[player_id]["current_actors"].append(self.current_hand[player_id]["current_actors"])
                    self.hand_info[player_id]["actions"].append(self.current_hand[player_id]["actions"])
                    self.hand_info[player_id]["rewards"].append(hand_rewards)
                    self.hand_info[player_id]["sample_weights"].append(self.current_hand[player_id]["sample_weights"])
                    self.hand_info[player_id]["hand_memory"].append(self.current_hand[player_id]["hand_memory"])
                    self.hand_info[player_id]["game_memory"].append(self.current_hand[player_id]["game_memory"])
                else:
                    self.hand_info[player_id]["states"].extend(self.current_hand[player_id]["states"])
                    self.hand_info[player_id]["current_actors"].extend(self.current_hand[player_id]["current_actors"])
                    self.hand_info[player_id]["actions"].extend(self.current_hand[player_id]["actions"])
                    self.hand_info[player_id]["rewards"].extend(hand_rewards)
                    self.hand_info[player_id]["sample_weights"].extend(self.current_hand[player_id]["sample_weights"])
                    self.hand_info[player_id]["hand_memory"].extend(self.current_hand[player_id]["hand_memory"])
                    self.hand_info[player_id]["game_memory"].extend([self.current_hand[player_id]["game_memory"],])

                if final_game_stack < self.game_params["min_bet"]:
                    # player is busted out
                    # update game view to remove the busted out player
                    busted_out.append(i)

            busted_out.sort(reverse=True)   # bust from largest index to lowest index so we don't change indices as we bust

            # bust out players
            for i in busted_out:
                player_id = self.game_player_ids[i]
                j = self.player_ids.index(player_id)
                self.player_winnings[player_id] += final_game_stacks[i] - self.starting_stacks[j]  # we look at the game starting stacks, not the hand starting stacks

                final_game_stacks.pop(i)
                self.game_players.pop(i)
                self.game_starting_stacks.pop(i)
                self.game_player_ids.pop(i)

                self.game_params["player_count"] = len(self.game_player_ids)

            # update the player game stacks
            self.game_starting_stacks = final_game_stacks

            if self.game_params["player_count"] < 2:
                return True

            # rotate the spots
            self.game_players.append(self.game_players.pop(0))
            self.game_player_ids.append(self.game_player_ids.pop(0))
            self.game_starting_stacks.append(self.game_starting_stacks.pop(0))

            return False

        except Exception as e:
            print(f"Exception: {e} encountered in Table {self.table_id} in linear round fn")
            return True  # terminate the table to avoid players from getting stuck in the void

    def _get_action(self, player, snapshot, current_actor, hand_hidden=None, game_hidden=None):
        with torch.no_grad():
            if IS_RECURRENT:
                player_action_tensor, new_hand_hidden = player.get_action((snapshot, current_actor), hand_hidden, game_hidden)
                player_action = player_action_tensor.detach().cpu().squeeze(0)
                return player_action, new_hand_hidden
            else:
                player_action_tensor = player.get_action((snapshot, current_actor))
                player_action = player_action_tensor.detach().cpu().squeeze(0)
                return player_action

    def play_game(self):
        try:
            self.timer.reset()  # Reset the timer at the start of a fresh game
            with self.timer.time("Game_Total_Duration"):
                done = False
                counter = 0
                while not done:
                    with self.timer.time("Game_Play_Round"):
                        done = self._play_round()
                    self._reset_current_hand()
                    counter += 1

                    if self.use_early_stopping and not done:
                        # early stopping is useful with the timeout, as otherwise really long game could potentially timeout
                        early_stopping = False
                        for player_id in self.game_player_ids:
                            if len(self.hand_info[player_id]["rewards"]) > self.batch_size:
                                done = True
                                early_stopping = True
                        if early_stopping:
                            print(f"Early stopping on Table {self.table_id}, reached batch size limit of {self.batch_size}")

                    # if counter % 100 == 0:
                # print(self.table_id, self.num_games_played+1, counter)
                # need to save the remaining player's winnings
                # update player winnings
                for i, player_id in enumerate(self.game_player_ids):
                    # we need the index of the player id in the original starting stack list
                    j = self.player_ids.index(player_id)
                    self.player_winnings[player_id] += self.game_starting_stacks[i] - self.starting_stacks[j]  # we look at the game starting stacks, not the hand starting stacks

            self.num_games_played += 1

            self.timer.log_to_tensorboard(self.writer, self.table_id, self.num_games_played)

            return True

        except Exception as e:
            print(f"Exception: {e} encountered in Table {self.table_id} in play game fn")
            return False

    def start(self):
        while True:
            try:
                data = self.in_queue.get(block=True, timeout=1)
            except Empty:
                continue

            if data is not None:
                if data["type"] == "message":
                    terminate = data.get("terminate", False)  # by default we assume that we do not need to terminate in case of a malformed message
                    if terminate:
                        # we need to send a message to the manager to alert him that we are terminating
                        message = {
                            "type": "termination",
                            "table_id": self.table_id,
                        }
                        self.out_queue.put(message)
                        return True
                # otherwise, we are good to go
                assert data["type"] == "players"

                player_ids = data["player_ids"]

                try:
                    players = ray.get(data["player_refs"])
                    self.current_player_versions = data["player_versions"]

                    # player_params_list = [player.get_params() for player in players]

                    session_hand_info = {
                        pid: {"states": [], "current_actors": [], "actions": [], "rewards": [], "sample_weights": [],
                              "hand_memory": [], "game_memory": []}
                        for pid in player_ids
                    }
                    session_player_winnings = {pid: 0.0 for pid in player_ids}

                    for _ in range(self.replay):
                        shuffle = list(range(len(player_ids)))
                        random.shuffle(shuffle)
                        # shuffled_player_params_list = [player_params_list[i] for i in shuffle]
                        shuffled_players = [players[i] for i in shuffle]
                        shuffled_player_ids = [player_ids[i] for i in shuffle]
                        self.reset(shuffled_players, shuffled_player_ids, **data["table_params"])
                        success = self.play_game()

                        if success:
                            # Aggregate data into the session accumulators
                            for pid in player_ids:
                                if IS_RECURRENT:
                                    # we again append rather than extend to keep the per-game structure
                                    # so for recurrent models, the structure becomes [per_game[per_hand[]]]
                                    session_hand_info[pid]["states"].append(self.hand_info[pid]["states"])
                                    session_hand_info[pid]["current_actors"].append(self.hand_info[pid]["current_actors"])
                                    session_hand_info[pid]["actions"].append(self.hand_info[pid]["actions"])
                                    session_hand_info[pid]["rewards"].append(self.hand_info[pid]["rewards"])
                                    session_hand_info[pid]["sample_weights"].append(self.hand_info[pid]["sample_weights"])
                                    session_hand_info[pid]["hand_memory"].append(self.hand_info[pid]["hand_memory"])
                                    session_hand_info[pid]["game_memory"].append(self.hand_info[pid]["game_memory"])
                                else:
                                    session_hand_info[pid]["states"].extend(self.hand_info[pid]["states"])
                                    session_hand_info[pid]["current_actors"].extend(self.hand_info[pid]["current_actors"])
                                    session_hand_info[pid]["actions"].extend(self.hand_info[pid]["actions"])
                                    session_hand_info[pid]["rewards"].extend(self.hand_info[pid]["rewards"])
                                    session_hand_info[pid]["sample_weights"].extend(self.hand_info[pid]["sample_weights"])
                                    session_hand_info[pid]["hand_memory"].extend(self.hand_info[pid]["hand_memory"])
                                    session_hand_info[pid]["game_memory"].extend([self.hand_info[pid]["game_memory"],])

                                session_player_winnings[pid] += self.player_winnings[pid]

                    batch = []
                    # Send the batched data exactly once per session
                    num_samples = []
                    for pid in player_ids:
                        p_index = player_ids.index(pid)
                        p_version = self.current_player_versions[p_index]
                        num_samples.append(len(session_hand_info[pid]["states"]))
                        batch.append({
                            "type": "data",
                            "table_id": self.table_id,
                            "player_id": pid,
                            "hand_info": ray.put(session_hand_info[pid]),
                            "player_winnings": session_player_winnings[pid],
                            "num_samples": len(session_hand_info[pid]["states"]),
                            "version": p_version
                        })
                    # now we send back the players
                    for player_id in player_ids:
                        other_players = [(pid, self.current_player_versions[player_ids.index(pid)]) for pid in player_ids if player_id != pid]
                        batch.append({
                            "type": "player",
                            "table_id": self.table_id,
                            "player_id": player_id,
                            "other_players": other_players
                        })
                    self.out_queue.put_nowait_batch(batch)

                except Exception as e:
                    print(f"Exception: {e} encountered in Table {self.table_id} in start fn")
                    batch = []

                    # now we send back the players
                    for player_id in player_ids:
                        batch.append({
                            "type": "player",
                            "table_id": self.table_id,
                            "player_id": player_id,
                            "other_players": None   # we have safety logic that prevents a weight with itself
                        })
                    self.out_queue.put_nowait_batch(batch)