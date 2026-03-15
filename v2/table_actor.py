import copy
import ray
from ray.util.queue import Queue, Empty
from pokerkit import NoLimitTexasHoldem, Automation
from action_interpreter import ActionInterpreter, Action
from state_interpreter import extract_state_snapshot
import random
import torch
from alg import PPOInferenceWrapper, PPO



# @ray.remote(num_cpus=1)
@ray.remote(num_cpus=0)
class TableActor:
    default_params = {
        "ante_trimming_status": True,
        "raw_antes": 0,
        "raw_blinds_or_straddles": (1, 2),
        "min_bet": 2,
        "raw_starting_stacks": 100,
        "player_count": 2
    }

    def __init__(self, table_id, device, in_queue: Queue, out_queue: Queue, max_table_size: int, discrete: bool):
        self.table_id = table_id
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.device = device
        self.discrete = discrete
        self.max_table_size = max_table_size
        self.action_interpreter: ActionInterpreter = ActionInterpreter()
        self.num_games_played = 0

        self.replay = 1  # number of games to play
        # for every parameter, we have an initial version and a game state view version as the game state evolves
        self.player_ids = None   # table facing view
        self.game_player_ids = None   # game state facing view
        self.players = [PPOInferenceWrapper(PPO.init_networks(self.device, self.discrete), self.discrete)
                        for _ in range(max_table_size)]
        self.game_players = self.players[:]
        self.params = None
        self.game_params = None
        self.starting_stacks = None
        self.game_starting_stacks = None

        self.current_player_versions = None
        self.hand_info = None
        self.current_hand = None
        self.player_winnings: dict[int, float] = None

    def reset(self, players_params_list, player_ids, **table_params):
        self.player_ids = player_ids
        self.game_player_ids = self.player_ids[:]

        self.player_winnings = {player_id: 0.0 for player_id in self.player_ids}

        self.players = [PPOInferenceWrapper(PPO.init_networks(self.device, self.discrete), self.discrete)
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

        self._reset_hand_info()
            
    def _reset_hand_info(self):
        self.hand_info = {}
        for player_id in self.player_ids:  # only care about current players
            self.hand_info[player_id] = {
                "states": [],
                "current_actors": [],
                "actions": [],
                "rewards": []
            }
        self._reset_current_hand()

    def _reset_current_hand(self):
        self.current_hand = {}
        for player_id in self.game_player_ids:  # only care about current players
            self.current_hand[player_id] = {
                "states": [],
                "current_actors": [],
                "actions": [],
            }

    def _play_round(self):
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

                try:
                   with torch.no_grad():
                        player_action_tensor = player.get_action((snapshot, current_actor))
                        player_action = player_action_tensor.detach().cpu()
                except Exception as e:
                    print(state)
                    raise e

                # we log the state and action for player training
                self.current_hand[player_id]["states"].append(snapshot)
                self.current_hand[player_id]["current_actors"].append(current_actor)
                self.current_hand[player_id]["actions"].append(player_action)

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
                    # we want to raise, we use the bet_sizing provided
                    if state.can_complete_bet_or_raise_to(bet_sizing):
                        state.complete_bet_or_raise_to(bet_sizing)
                    elif state.can_check_or_call():
                        state.check_or_call()
                    elif state.can_fold():
                        state.fold()
                    else:
                        raise RuntimeError("No legal fallback from RAISE")
                else:
                    assert interpreted_action == Action.ALL_IN
                    all_in_size = state.max_completion_betting_or_raising_to_amount
                    if state.can_complete_bet_or_raise_to(all_in_size):
                        state.complete_bet_or_raise_to(all_in_size)
                    elif state.can_check_or_call():
                        state.check_or_call()
                    elif state.can_fold():
                        state.fold()
                    else:
                        raise RuntimeError("No legal fallback from ALL_IN")

            # compute rewards
            final_game_stacks = state.stacks[:]
            busted_out = []
            for i, (final_game_stack, game_starting_stack) in enumerate(zip(final_game_stacks,
                                                                            self.game_starting_stacks)):
                reward = (final_game_stack - game_starting_stack)/(self.game_params["raw_blinds_or_straddles"][-1])

                # save the hand info
                player_id = self.game_player_ids[i]
                hand_rewards = [reward] * len(self.current_hand[player_id]["states"])

                self.hand_info[player_id]["states"].extend(self.current_hand[player_id]["states"])
                self.hand_info[player_id]["current_actors"].extend(self.current_hand[player_id]["current_actors"])
                self.hand_info[player_id]["actions"].extend(self.current_hand[player_id]["actions"])
                self.hand_info[player_id]["rewards"].extend(hand_rewards)

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
            print(f"Exception: {e} encountered in Table {self.table_id}")
            return True  # terminate the table to avoid players from getting stuck in the void

    def play_game(self):
        try:
            done = False
            while not done:
                done = self._play_round()

                self._reset_current_hand()

            # need to save the remaining player's winnings
            # update player winnings
            for i, player_id in enumerate(self.game_player_ids):
                # we need the index of the player id in the original starting stack list
                j = self.player_ids.index(player_id)
                self.player_winnings[player_id] += self.game_starting_stacks[i] - self.starting_stacks[j]  # we look at the game starting stacks, not the hand starting stacks

            # put the winnings and hand data in the queue
            batch = []
            for player_id, hand_info in self.hand_info.items():
                player_winnings = self.player_winnings[player_id]
                hand_info_ref = ray.put(hand_info)
                p_index = self.player_ids.index(player_id)
                p_version = self.current_player_versions[p_index]

                batch.append({
                    "type": "data",
                    "table_id": self.table_id,
                    "player_id": player_id,
                    "hand_info": hand_info_ref,
                    "player_winnings": player_winnings,
                    "num_samples": len(hand_info["states"]),
                    "version": p_version
                })
            self.out_queue.put_nowait_batch(batch)
            self.num_games_played += 1
            return True

        except Exception as e:
            print(f"Exception: {e} encountered in Table {self.table_id}")
            return False

    def start(self):
        while True:
            try:
                data = self.in_queue.get(block=True, timeout=1)
            except Empty:
                continue

            if data is not None:
                if data["type"] == "message":
                    terminate = data.get("terminate", True)  # by default we assume that we need to terminate in case of a malformed message
                    if terminate:
                        return True

                # otherwise, we are good to go
                assert data["type"] == "players"
                players, player_ids = ray.get(data["player_refs"]), data["player_ids"]
                self.current_player_versions = data["player_versions"]

                player_params_list = [player.get_params() for player in players]

                for _ in range(self.replay):
                    shuffle = list(range(len(player_ids)))
                    random.shuffle(shuffle)
                    shuffled_player_params_list = [player_params_list[i] for i in shuffle]
                    shuffled_player_ids = [player_ids[i] for i in shuffle]
                    self.reset(shuffled_player_params_list, shuffled_player_ids, **data["table_params"])
                    self.play_game()

                batch = []
                # now we send back the players
                for player_id in player_ids:
                    batch.append({
                        "type": "player",
                        "table_id": self.table_id,
                        "player_id": player_id
                    })
                self.out_queue.put_nowait_batch(batch)
