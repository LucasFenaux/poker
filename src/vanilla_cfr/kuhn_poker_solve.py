import argparse
from dataclasses import dataclass
import random
import math
import time


@dataclass
class Action:
    CHECK = "h"
    BET = "b"
    CALL = "c"
    FOLD = "f"
    JACK = "j"
    QUEEN = "q"
    KING = "k"

    @classmethod
    def get_cards(cls):
        return cls.JACK, cls.QUEEN, cls.KING

    @classmethod
    def get_cards_except(cls, exc: str):
        cards = [cls.JACK, cls.QUEEN, cls.KING]
        cards.remove(exc)
        return cards

    @classmethod
    def get_player_actions(cls):
        return [cls.CHECK, cls.BET, cls.CALL, cls.FOLD]

    @classmethod
    def get_all_actions(cls):
        return [cls.CHECK, cls.BET, cls.CALL, cls.FOLD, cls.JACK, cls.QUEEN, cls.KING]


@dataclass
class Card:
    mapping = {Action.JACK: 0,
               Action.QUEEN: 1,
               Action.KING: 2}

    def __init__(self, card: str):
        self.card = card
        self.value = self.mapping[card]

    def __gt__(self, other):
        return self.value > other.value

    def __eq__(self, other):
        return self.value == other.value

    def __lt__(self, other):
        return self.value < other.value

    def __le__(self, other):
        return self.value <= other.value

    def __ge__(self, other):
        return self.value >= other.value

    def __str__(self):
        return self.card

    def __repr__(self):
        return self.card


class InfoSet:
    def __init__(self, node, player: int):
        self.node = node
        assert player == 1 or player == 2
        self.player = player
        self.repr = str(self.node)
        if self.player == 1:
            # remove the second character
            self.repr = self.repr[0] + self.repr[2:]
        else:
            # remove the first character
            self.repr = self.repr[1:]

    def get_valid_actions(self):
        return self.node.get_valid_actions()  # no matter the original node, all info states with the same repr must have the same available actions

    def __str__(self):
        return self.repr

    def __repr__(self):
        return self.repr

    def __hash__(self):
        return hash(self.repr)

    def __eq__(self, other):
        return self.repr == other.repr


# nodes representing the full game state
class Node:
    @classmethod
    def generate_all_info_states(cls, player: int, parent = None):
        if parent is None:
            current_node = Node()
        else:
            current_node = parent

        info_states = set()

        for action, child in current_node.get_children().items():
            if not child.is_chance_node and not child.is_terminal and child.get_player_to_act() == player:
                info_states.add(child.convert_to_info_state(player))
            for info_state in Node.generate_all_info_states(player, child):
                info_states.add(info_state)

        return info_states

    def __init__(self, parent=None, append: str = ""):
        self.parent = parent
        self.repr = "" if parent is None else str(parent)
        # need to add the action used to create that node
        if append != "":
            self.repr += append

        self.valid_actions = self.get_valid_actions()
        self.is_terminal = len(self.valid_actions) == 0
        self.is_chance_node = len(self.repr) < 2

    def __repr__(self):
        return self.repr

    def __str__(self):
        return self.repr

    def __hash__(self):
        return hash(self.repr)

    def get_player_to_act(self):
        if len(self.repr) % 2 == 0:
            return 1
        return 2

    def convert_to_info_state(self, player: int) -> InfoSet:
        # map the node to an info state
        return InfoSet(node=self, player=player)

    def get_children(self):
        children = {}
        for action in self.valid_actions:
            children[action] = Node(parent=self, append=action)
        return children

    def get_utility(self) -> tuple[float, float]:
        assert self.is_terminal
        # first we see if the node ends with a fold
        if self.repr.endswith(Action.FOLD):
            # it doesn't actually matter if one of the players bet, because you cannot fold after
            # placing/calling a bet, so you lose at most 1 for folding (the initial wager)
            if self.get_player_to_act() == 1:
                # player 2 folded
                return 1., -1.
            else:
                return -1., 1.

        # if the game didn't end with a fold, it can only end with two checks, or a bet then a call
        if self.repr.endswith(Action.CHECK):
            gain = 1.
        else:
            assert self.repr.endswith(Action.CALL)
            gain = 2.

        # now we determine who won, map card action to a Card object and compare them
        player_one_card = Card(self.repr[0])
        player_two_card = Card(self.repr[1])
        if player_one_card > player_two_card:
            return gain, -gain
        return -gain, gain

    def get_player_utility(self, player: int):
        assert self.is_terminal
        return self.get_utility()[player-1]

    def get_valid_actions(self):
        if len(self.repr) == 0:
            # any of the three cards are valid
            return Action.get_cards()
        elif len(self.repr) == 1:
            # need to remove the current card
            return Action.get_cards_except(self.repr[0])
        elif len(self.repr) == 2:
            # first player's move to act
            return Action.CHECK, Action.BET
        else:
            # terminal states
            if (self.repr.endswith(Action.CHECK + Action.CHECK) or self.repr.endswith(Action.FOLD) or
                    self.repr.endswith(Action.CALL)):
                return ()

            # check what the last action was
            if self.repr.endswith(Action.CHECK):
                return Action.CHECK, Action.BET
            else:
                # last player bet
                return Action.FOLD, Action.CALL


class Player:
    """
    Represents the current strategy of a player as well as the average strategy
    """
    def __init__(self, player: int, all_info_states: set):
        self.order = player
        self.all_info_states = all_info_states
        self.cumulative_regret = {state: {action: 0 for action in state.get_valid_actions()} for state in all_info_states}
        self.average_strategy = {state: {action: 0 for action in state.get_valid_actions()} for state in all_info_states}
        self.num_iterations = 0

    def get_probs(self, node: Node) -> tuple[list[str], list[float]]:
        valid_actions = []
        regrets = []
        all_negative_or_zero = True
        info_state = node.convert_to_info_state(self.order)
        for action in node.get_valid_actions():
            # we look at the info states of the children
            valid_actions.append(action)
            regret = self.cumulative_regret[info_state][action]
            regrets.append(regret)
            if regret > 0:
                all_negative_or_zero = False

        assert len(valid_actions) > 0

        if all_negative_or_zero:
            # select randomly among the actions
            return valid_actions, [1/len(valid_actions)]*len(valid_actions)
        else:
            # we use the regrets as probabilities
            # first we normalize them to get the reach probabilities
            non_zero_sum = sum([r if r > 0 else 0. for r in regrets])
            probs = [r / non_zero_sum if r > 0 else 0. for r in regrets]
            assert math.isclose(sum(probs), 1, abs_tol=1e-5)
            return valid_actions, probs

    def get_average_strategy_probs(self):
        probs = {}
        for key in self.average_strategy.keys():
            total = sum(self.average_strategy[key].values())
            assert total != 0  # we traverse every node, we shoudn't have a zero node
            probs[key] = {}
            for action in self.average_strategy[key].keys():
                probs[key][action] = self.average_strategy[key][action] / total
        return probs


    def __call__(self, node: Node, *args, **kwargs):
        # for the current InfoState, we look at all the possible suffix of it, if they exist we add them to the list to
        # choose from, otherwise we ignore them
        # since by default the info state should already have player order and card, we can just try to add every
        # move and only keep the ones that exist in our strategy (meaning they are valid)
        valid_actions, probs = self.get_probs(node)
        # then we random sample according to the weight
        idx = random.choices(range(len(valid_actions)), weights=probs, k=1)[0]
        return valid_actions[idx], probs[idx]


def cfr_traversal(node, player: int, player_one: Player, player_two: Player, player_one_prob: float,
                  player_two_prob: float, chance_prob: float):
    if node.is_terminal:
        return node.get_player_utility(player)

    if node.is_chance_node:
        node_value = 0
        for action, child in node.get_children().items():
            node_value += cfr_traversal(child, player, player_one, player_two, player_one_prob, player_two_prob,
                                        chance_prob/(len(node.valid_actions)))  # equal probability in this case for each possible card
        return node_value

    # get the information state from the node
    player_to_act = node.get_player_to_act()
    if player_to_act == 1:
        to_act = player_one
    else:
        to_act = player_two

    # we then compute the strategy for this node
    valid_actions, probs = to_act.get_probs(node)

    info_set_value = 0
    action_values = {}
    for action, prob in zip(valid_actions, probs):
        child = Node(parent=node, append=action)
        if player_to_act == 1:
            action_value = cfr_traversal(child, player, player_one, player_two, player_one_prob*prob,
                                         player_two_prob, chance_prob)
        else:
            action_value = cfr_traversal(child, player, player_one, player_two, player_one_prob,
                                         player_two_prob*prob, chance_prob)

        action_values[action] = action_value

        assert isinstance(action_value, float)
        info_set_value += action_value * prob

    # update regret table
    if player_to_act == player:
        if player == 1:
            player_prob = player_one_prob
            other_probs = player_two_prob * chance_prob
        else:
            player_prob = player_two_prob
            other_probs = player_one_prob * chance_prob

        for action, prob in zip(valid_actions, probs):
            info_set = node.convert_to_info_state(player)

            to_act.cumulative_regret[info_set][action] += other_probs * (action_values[action] - info_set_value)
            to_act.average_strategy[info_set][action] += player_prob * prob

    return info_set_value


def display_strategy(player: Player):
    print(f"Player {player.order}'s strategy:")
    strategy = player.get_average_strategy_probs()
    for info_state in sorted(list(strategy.keys()), key=lambda x: str(x)):
        probs = strategy[info_state]
        probs_str = ", ".join([f"{action}: {prob*100:.2f}%" for action, prob in probs.items()])
        print(f"  {info_state}: {probs_str}")


def main(args):
    # precompute all info states
    all_info_states_player_one = Node.generate_all_info_states(player=1)
    all_info_states_player_two = Node.generate_all_info_states(player=2)

    # create our two players
    player_one = Player(player=1, all_info_states=all_info_states_player_one)
    player_one_prob = 1

    player_two = Player(player=2, all_info_states=all_info_states_player_two)
    player_two_prob = 1

    chance_prob = 1

    # we repeat our cfr walk
    for i in range(args.num_iterations):
        # we alternate between the players
        for j in range(1, 3):
            # initialize the root node
            root_node = Node()
            cfr_traversal(root_node, j, player_one, player_two, player_one_prob, player_two_prob, chance_prob)

        if i % 10_000 == 0:
            print(f"Iteration {i} / {args.num_iterations}")
            display_strategy(player_one)
            print()
            display_strategy(player_two)

    display_strategy(player_one)
    print()
    display_strategy(player_two)


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-n", "--num_iterations", default=100_000)
    return parser.parse_args()


if __name__ == '__main__':
    start = time.time()
    args = parse_args()
    main(args)
    print(f"Took {time.time() - start:.1f} seconds")