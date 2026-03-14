import copy
import ray
from ray.util.queue import Queue, Empty
from pokerkit import NoLimitTexasHoldem, Automation
from action_interpreter import ActionInterpreter, Action
from state_interpreter import extract_state_snapshot
from models import load_dummy_model
import torch
from alg import PPOInferenceWrapper



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
        self.players: list[PPOInferenceWrapper] = [PPOInferenceWrapper(load_dummy_model(device, discrete), discrete)
                                                   for _ in range(max_table_size)]
        self.max_table_size = max_table_size
        self.player_ids = None
        self.params = None
        self.action_interpreter: ActionInterpreter = ActionInterpreter()
        self.player_winnings: dict[int, float] = None
        self.stacks = None
        self.starting_stacks = None
        self.hand_info = None
        self.current_hand = None
        self.num_games_played = 0

    def reset(self, players_params_list: list[dict[str, torch.Tensor]], player_ids, **table_params):
        self.players = [PPOInferenceWrapper(load_dummy_model(self.device, self.discrete), self.discrete)
                        for _ in range(len(player_ids))]

        for i, (player, player_params) in enumerate(zip(self.players, players_params_list)):
            player.load_network_params(player_params)
            player.to(self.device)

        self.player_ids = player_ids

        self.params = table_params
        for param in self.default_params:
            if param not in self.params:
                self.params[param] = self.default_params[param]

        assert len(self.players) == self.params["player_count"]

        self.player_winnings: dict[int: float] = {player_id: 0 for player_id in self.player_ids}
        starting_stacks = self.params.pop("raw_starting_stacks")
        if isinstance(starting_stacks, int):
            self.stacks = [starting_stacks] * len(self.players)
            self.starting_stacks = [starting_stacks] * len(self.players)  # will use it to compute game winnings
        else:
            self.stacks = starting_stacks
            self.starting_stacks = copy.deepcopy(starting_stacks)  # will use it to compute game winnings

        self.hand_info = {}
        self.current_hand = {}
        self._reset_hand_info()
            
    def _reset_hand_info(self):
        self.hand_info = {}
        for player_id in self.player_ids:
            self.hand_info[player_id] = {
                "states": [],
                "current_actors": [],
                "actions": [],
                "rewards": []
            }
        self._reset_current_hand()

    def _reset_current_hand(self):
        self.current_hand = {}
        for player_id in self.player_ids:
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
                raw_starting_stacks=self.stacks,
                **self.params
            )

            starting_stacks = copy.deepcopy(self.stacks)

            while state.status:
                # find the current player and gather the necessary information
                current_actor = state.actor_index
                player = self.players[current_actor]
                player_id = self.player_ids[current_actor]
                snapshot = extract_state_snapshot(state, current_actor)
                try:
                   with torch.no_grad():
                        player_action_tensor = player.get_action((snapshot, current_actor))
                        player_action = player_action_tensor.detach().cpu()

                except Exception as e:
                    print(state)
                    raise e
                # time.sleep(0.01)

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

            # compute rewards and update player stacks
            final_stacks = copy.deepcopy(state.stacks)
            self.stacks = final_stacks
            reward_list = [(final - start)/self.params["raw_blinds_or_straddles"][-1] for final, start in
                      zip(final_stacks, starting_stacks)]

            busted_out = []
            for i, player_id in enumerate(self.player_ids):
                reward = reward_list[i]
                rewards = [reward] * len(self.current_hand[player_id]["states"])

                assert len(rewards) == len(self.current_hand[player_id]["current_actors"])
                assert len(rewards) == len(self.current_hand[player_id]["actions"])

                self.hand_info[player_id]["states"].extend(self.current_hand[player_id]["states"])
                self.hand_info[player_id]["current_actors"].extend(self.current_hand[player_id]["current_actors"])
                self.hand_info[player_id]["actions"].extend(self.current_hand[player_id]["actions"])
                self.hand_info[player_id]["rewards"].extend(rewards)

                # check which players busted out
                final_stack = final_stacks[i]
                if final_stack <= self.params["min_bet"]:
                    busted_out.append(i)

            busted_out.sort(reverse=True)

            for busted_player_idx in busted_out:
                # update player winnings
                busted_player_id = self.player_ids[busted_player_idx]
                self.player_winnings[busted_player_id] += (final_stacks[busted_player_idx] -
                                                           self.starting_stacks[busted_player_idx])  # we look at the game starting stacks, not the hand starting stacks

                self.players.pop(busted_player_idx)
                self.stacks.pop(busted_player_idx)
                self.player_ids.pop(busted_player_idx)
                self.starting_stacks.pop(busted_player_idx)

            # We take the player at index 0 and move them to the end of the line.
            self.players.append(self.players.pop(0))
            self.player_ids.append(self.player_ids.pop(0))
            self.stacks.append(self.stacks.pop(0))
            self.starting_stacks.append(self.starting_stacks.pop(0))

            if len(self.players) < 2:
                return True

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
            for i, player_id in enumerate(self.player_ids):
                self.player_winnings[player_id] += self.stacks[i] - self.starting_stacks[i]  # we look at the game starting stacks, not the hand starting stacks

            # put the winnings and hand data in the queue
            batch = []
            for player_id, hand_info in self.hand_info.items():
                player_winnings = self.player_winnings[player_id]
                batch.append({
                    "table_id": self.table_id,
                    "player_id": player_id,
                    "hand_info": hand_info,
                    "player_winnings": player_winnings
                })
            self.out_queue.put_nowait_batch(batch)
            self.num_games_played += 1
            return True
        except Exception as e:
            print(f"Exception: {e} encountered in Table {self.table_id}")

            # need to save the remaining player's winnings
            # update player winnings
            for i, player_id in enumerate(self.player_ids):
                self.player_winnings[player_id] += self.stacks[i] - self.starting_stacks[i]  # we look at the game starting stacks, not the hand starting stacks

            # put the winnings and hand data in the queue
            batch = []
            for player_id, hand_info in self.hand_info.items():
                player_winnings = self.player_winnings[player_id]
                batch.append({
                    "table_id": self.table_id,
                    "player_id": player_id,
                    "hand_info": hand_info,
                    "player_winnings": player_winnings
                })
            self.out_queue.put_nowait_batch(batch)
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
                self.reset(data["players_params_list"], data["player_ids"], **data["table_params"])

                self.play_game()
