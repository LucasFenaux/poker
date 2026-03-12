import copy

import ray
from pokerkit import NoLimitTexasHoldem, Automation
from player_actor import PlayerActor
from action_interpreter import ActionInterpreter, Action
from state_interpreter import extract_state_snapshot


@ray.remote(num_cpus=0.1)
class TableActor:
    default_params = {
        "ante_trimming_status": True,
        "raw_antes": 0,
        "raw_blinds_or_straddles": (1, 2),
        "min_bet": 2,
        "raw_starting_stacks": 100,
        "player_count": 2
    }

    def __init__(self, players: list[PlayerActor], **params):
        self.players: list[PlayerActor] = players
        self.player_ids = ray.get([player.get_id.remote() for player in self.players])
        self.params = params
        self.action_interpreter: ActionInterpreter = ActionInterpreter()
        for param in self.default_params:
            if param not in self.params:
                self.params[param] = self.default_params[param]

        assert len(self.players) == self.params["player_count"]

        ids = ray.get([p.get_id.remote() for p in players])
        self.player_winnings: dict[int: float] = {player_id: 0 for player_id in ids}
        starting_stacks = self.params.pop("raw_starting_stacks")
        if isinstance(starting_stacks, int):
            self.stacks = [starting_stacks] * len(self.players)
            self.starting_stacks = [starting_stacks] * len(self.players)  # will use it to compute game winnings

        else:
            self.stacks = starting_stacks
            self.starting_stacks = copy.deepcopy(starting_stacks)  # will use it to compute game winnings

        self.hand_info = {}
        self._reset_hand_info()
            
    def _reset_hand_info(self):
        self.hand_info = {}
        for i in range(len(self.players)):
            self.hand_info[i] = {
                "states": [],
                "current_actors": [],
                "actions": [],
            }

    def _play_round(self):
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
            try:
                player_action = ray.get(player.get_action.remote(state, current_actor))
            except Exception as e:
                print(state)
                raise e
            # we log the state and action for player training
            # self.hand_info[current_actor]["states"].append(copy.deepcopy(state))
            snapshot = extract_state_snapshot(state, current_actor)
            self.hand_info[current_actor]["states"].append(snapshot)
            self.hand_info[current_actor]["current_actors"].append(current_actor)
            self.hand_info[current_actor]["actions"].append(player_action)

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
                    RuntimeError("No legal action in CHECK_OR_FOLD branch")
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
        reward = [(final - start)/self.params["raw_blinds_or_straddles"][-1] for final, start in
                  zip(final_stacks, starting_stacks)]

        busted_out = []
        for i, player in enumerate(self.players):
            # send the players their hand info for training
            player.store_hand.remote(states=self.hand_info[i]["states"],
                                     current_actors=self.hand_info[i]["current_actors"],
                                     actions=self.hand_info[i]["actions"],
                                     reward=reward[i])
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
            self.hand_info.pop(busted_player_idx)
            self.player_ids.pop(busted_player_idx)

        if len(self.players) < 2:
            return True

        return False

    def play_game(self):
        try:
            done = False
            while not done:
                done = self._play_round()

                for player in self.players:
                    player.update.remote()

                self._reset_hand_info()

            # need to save th remaining player's winnings
            # update player winnings
            for i, player_id in enumerate(self.player_ids):
                self.player_winnings[player_id] += self.stacks[i] - self.starting_stacks[i]  # we look at the game starting stacks, not the hand starting stacks

            # the players save themselves at the end of the game. We could save at each update
            # but it might be too disk intensive and unnecessary
            for player in self.players:
                player.save.remote()

            return self.player_winnings
        except Exception as e:
            print(f"Encountered exception: {e}")
            raise e
