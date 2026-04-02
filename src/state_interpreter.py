"""
Converts model actions into table instructions that are understood by the TableActor.
Transforms state information provided by the TableActor into a state vector for the model using nn.Embedding models.
It is trainable and should be part of the model.
"""
import torch
import torch.nn as nn
import pokerkit
from dataclasses import dataclass
from typing import Optional, Union
import math
from global_settings import MAX_TABLE_SIZE


def sign_fn(x):
    return 1 if x >= 0 else -1

@dataclass(slots=True)
class StateSnapshot:
    hole_cards: str
    board_cards: str
    player_count: int
    blinds_or_straddles: tuple[int, ...]
    bets: list[float]
    stacks: list[float]
    in_hand: list[bool]
    pots: list[float]  # We will store just the raw amounts here
    min_bet: Optional[float]
    max_bet: Optional[float]


def extract_state_snapshot(state, current_actor) -> StateSnapshot:
    """Creates a tiny, memory-efficient dataclass of the current table state."""

    cards = state.hole_cards[current_actor]
    cards = "".join([repr(card) for card in cards])
    if len(state.board_cards) > 0 and isinstance(state.board_cards[0], list):
        board_cards = state.board_cards[0][:]  # we will never encounter multiple boards when making decisions
    elif len(state.board_cards) > 0:
        raise TypeError(f"Board is single depth all of the time: {state.board_cards}")
    else:
        board_cards = []
    to_add = 5 - len(board_cards)

    board_cards = "".join([repr(card) for card in board_cards] + ["??"] * to_add)

    return StateSnapshot(
        hole_cards=cards,
        board_cards=board_cards,
        player_count=state.player_count,
        blinds_or_straddles=tuple(state.blinds_or_straddles),
        bets=list(state.bets),
        stacks=list(state.stacks),
        in_hand=list(state.statuses),
        pots=[p.amount for p in state.pots] if state.pots else [],  # Extract raw numbers instantly
        min_bet=state.min_completion_betting_or_raising_to_amount,
        max_bet=state.max_completion_betting_or_raising_to_amount
    )

def safe_div(a, b, default=0):
    return a / b if (b is not None and b != 0) else default

def safe_log(x):
    if isinstance(x, torch.Tensor):
        sign = torch.sign(x)
        x = sign * torch.log(x.abs() + 1)
    else:
        sign = sign_fn(x)
        x = sign * math.log(math.fabs(x) + 1)
    return x


class CardEmbedding(nn.Module):
    rank_mapping = {
        "2": 0,
        "3": 1,
        "4": 2,
        "5": 3,
        "6": 4,
        "7": 5,
        "8": 6,
        "9": 7,
        "T": 8,
        "J": 9,
        "Q": 10,
        "K": 11,
        "A": 12,
        "?": 13
    }

    suit_mapping = {
        "c": 0,
        "d": 1,
        "h": 2,
        "s": 3,
        "?": 4
    }

    def __init__(self, rank_dim, suit_dim):
        super(CardEmbedding, self).__init__()
        self.rank_embedding = nn.Embedding(14, rank_dim)
        self.suit_embedding = nn.Embedding(5, suit_dim)

    @classmethod
    def parse_cards(cls, cards: str):
        """ Assumes the cards is a string with {Rank}{Suit}...{Rank}{Suit} format.

        :return:
        """
        assert len(cards) % 2 == 0, print(f"please specify the suit of all the cards: {cards}")
        ranks = [cls.rank_mapping[cards[i]] for i in range(len(cards)) if i % 2 == 0]
        suits = [cls.suit_mapping[cards[i]] for i in range(len(cards)) if i % 2 == 1]
        return torch.tensor(ranks, dtype=torch.long), torch.tensor(suits, dtype=torch.long)

    def forward(self, ranks, suits):
        r = self.rank_embedding(ranks)
        s = self.suit_embedding(suits)
        return torch.cat([r, s], dim=-1)



class StateInterpreter(nn.Module):
    def __init__(self, device, rank_dim: int = 16, suit_dim: int = 4):
        super().__init__()
        self.device = device
        self.num_player_embedding_size = 4
        self.rel_to_button_embedding_size = 4

        self.rank_dim = rank_dim
        self.suit_dim = suit_dim
        self.card_embedding = CardEmbedding(rank_dim, suit_dim).to(device)
        self.max_num_players = MAX_TABLE_SIZE

        # self.max_num_players = 9
        self.num_player_embedding = nn.Embedding(self.max_num_players+1, self.num_player_embedding_size)

        self.rel_to_button_embedding = nn.Embedding(self.max_num_players+1, self.rel_to_button_embedding_size)

    def expected_input_size(self):
        size = (self.rank_dim + self.suit_dim) * 2  # the 2 player cards
        size += (self.rank_dim + self.suit_dim) * 5   # the 5 board cards
        size += 10  # sb, bb, pot, min_bet, max_bet features
        size += 3  # log features
        size += (13 + 2) * self.max_num_players   # player features
        size += self.rel_to_button_embedding_size * self.max_num_players  # rel to button features
        size += self.num_player_embedding_size
        size += 6 * self.max_num_players   # table state + player mask
        return size

    def forward(self, state: Union[pokerkit.State,StateSnapshot], current_actor: int):
        # --- Handle both Live States and Training Snapshots ---
        if isinstance(state, StateSnapshot):
            # Training Mode: Clean dot-notation access!
            cards = state.hole_cards
            board_cards = state.board_cards
            num_players = state.player_count
            sb, bb = state.blinds_or_straddles
            bets = state.bets
            stacks = state.stacks
            in_hand = state.in_hand

            # Since we extracted the float amounts in the dataclass:
            pot = sum(bets) + sum(state.pots)

            min_bet = state.min_bet
            max_bet = state.max_bet

        else:
            # Live Gameplay Mode (same as before)
            cards = state.hole_cards[current_actor]
            cards = "".join([repr(card) for card in cards])
            if len(state.board_cards) > 0 and isinstance(state.board_cards[0], list):
                board_cards = state.board_cards[0][:]  # we will never encounter multiple boards when making decisions
            elif len(state.board_cards) > 0:
                raise TypeError(f"Board is single depth all of the time: {state.board_cards}")
            else:
                board_cards = []
            to_add = 5 - len(board_cards)

            board_cards = "".join([repr(card) for card in board_cards] + ["??"] * to_add)
            # board_cards = "".join([repr(card) for card in board_cards_list])

            num_players = state.player_count
            sb, bb = state.blinds_or_straddles
            bets = list(state.bets)
            stacks = list(state.stacks)
            in_hand = list(state.statuses)

            # Live mode calculates the pot amounts on the fly
            pot = sum(bets) + sum(p.amount for p in (state.pots or []))

            min_bet = state.min_completion_betting_or_raising_to_amount
            max_bet = state.max_completion_betting_or_raising_to_amount

        # min/max bet info
        # min_bet = state.min_completion_betting_or_raising_to_amount
        if min_bet is None:
            min_bet = max(bets)

        # max_bet = state.max_completion_betting_or_raising_to_amount

        if max_bet is None:
            max_bet = min_bet  # Or some other logical fallback

        # player info
        # collecting all the info

        is_button = [i == num_players - 1 for i in range(num_players)]
        is_sb =  [i == 0 for i in range(num_players)]
        is_bb =  [i == 1 for i in range(num_players)]
        all_in = [active and stack == 0 for active, stack in zip(in_hand, stacks)]
        button = num_players - 1
        rel_to_button = [(button - i) % num_players for i in range(num_players)]

        # centering about the hero
        def center_on_hero(x):
            return x[current_actor:] + x[:current_actor]

        bets = center_on_hero(bets)
        stacks = center_on_hero(stacks)
        in_hand = center_on_hero(in_hand)
        is_button = center_on_hero(is_button)
        is_sb = center_on_hero(is_sb)
        is_bb = center_on_hero(is_bb)
        all_in = center_on_hero(all_in)
        rel_to_button = center_on_hero(rel_to_button)

        to_pad = self.max_num_players - num_players
        player_mask = [1] * num_players + [0] * to_pad

        def pad_list(x, pad_val):
            return x + [pad_val] * to_pad

        bets = pad_list(bets, 0)
        stacks = pad_list(stacks, 0)
        in_hand = pad_list(in_hand, False)
        is_button = pad_list(is_button, False)
        is_sb = pad_list(is_sb, False)
        is_bb = pad_list(is_bb, False)
        all_in = pad_list(all_in, False)
        rel_to_button = pad_list(rel_to_button, 0)

        return self._embed_forward(player_cards=cards, board_cards=board_cards, num_players=num_players,
                                   small_blind=sb, big_blind=bb, min_bet=min_bet, max_bet=max_bet, pot=pot,
                                   bets=bets, stacks=stacks, in_hand=in_hand,
                                   is_button=is_button, is_sb=is_sb, is_bb=is_bb, all_in=all_in,
                                   rel_to_button=rel_to_button, player_mask=player_mask)

    def _embed_forward(self, player_cards: str,  # need to embed
                       board_cards: str,   # need to embed
                       num_players: int,   # need to embed
                       small_blind: float,  # need to compute features from
                       big_blind: float,   # need to compute features from
                       min_bet: float,  # need to compute features from
                       max_bet: float,  # need to compute features from
                       pot: float,   # need to compute features from
                       bets: list[float],  # need to compute features from
                       stacks: list[float],  # need to compute features from
                       in_hand: list[bool],   # convert to 0-1 tensor
                       is_button: list[bool],   # convert to 0-1 tensor
                       is_sb: list[bool],    # convert to 0-1 tensor
                       is_bb: list[bool],    # convert to 0-1 tensor
                       all_in: list[bool],    # convert to 0-1 tensor
                       rel_to_button: list[int],  # need to embed
                       player_mask: list[int]   # which players are real and which are padding
                       ):

        # feature tensor where we collect non-tensor input features
        features = []

        # we will need the max stack for future computations
        max_stack = max(stacks)
        assert max_stack > 0, print("all stacks cannot be 0 otherwise all players are already all in and there is no decision to make")

        # embed player cards
        player_ranks, player_suits = self.card_embedding.parse_cards(player_cards)
        player_ranks, player_suits = player_ranks.to(self.device), player_suits.to(self.device)
        player_card_embeddings = self.card_embedding(player_ranks, player_suits).flatten()

        # embed board cards
        board_ranks, board_suits = self.card_embedding.parse_cards(board_cards)
        board_ranks, board_suits = board_ranks.to(self.device), board_suits.to(self.device)
        board_card_embeddings = self.card_embedding(board_ranks, board_suits).flatten()

        # embed num players
        num_player_embeddings = self.num_player_embedding(torch.tensor(num_players, dtype=torch.long).to(self.device)).flatten()

        # compute sb features
        # compute the ratio of the small blind to the pot
        # we compute the ratio of the small blind to the big blind
        sb_to_pot = small_blind / pot
        sb_to_bb = small_blind / big_blind

        features.append(sb_to_pot)
        features.append(sb_to_bb)

        # compute bb features
        # we compute the ratio of the big blind to the pot
        bb_to_pot = big_blind / pot

        features.append(bb_to_pot)

        # compute the min bet features
        sb_to_min_bet = small_blind / min_bet
        bb_to_min_bet = big_blind / min_bet
        min_bet_to_pot = min_bet / pot
        min_bet_to_max_bet = min_bet / max_bet
        # adding log features
        log_min_bet_to_bb = safe_log(safe_div(min_bet, big_blind))

        features.append(sb_to_min_bet)
        features.append(bb_to_min_bet)
        features.append(min_bet_to_pot)
        features.append(min_bet_to_max_bet)
        # adding log features
        features.append(log_min_bet_to_bb)

        # compute the max bet features
        sb_to_max_bet = small_blind / max_bet
        bb_to_max_bet = big_blind / max_bet
        # adding log features
        log_max_bet_to_bb = safe_log(safe_div(max_bet, big_blind))
        # can't compute max_bet to pot as it might blow up for early hands in deep stacked games

        features.append(sb_to_max_bet)
        features.append(bb_to_max_bet)
        # adding log features
        features.append(log_max_bet_to_bb)

        # compute pot features
        # we compute the ratio of the pot to the max stack
        pot_to_max_stack = safe_div(pot, max_stack, default=None)
        # adding log features
        log_pot_to_bb = safe_log(safe_div(pot, big_blind))

        features.append(pot_to_max_stack)
        # adding log features
        features.append(log_pot_to_bb)

        # compute bets and stack features for each player

        for player_index, is_real_player in enumerate(player_mask):
            if is_real_player:
                bet = bets[player_index]
                stack = stacks[player_index]

                # we compute for each:
                # - the ratio to the player's stack
                # - the ratio to the max stack
                # - the ratio to the pot
                # - the ratio of the small blind to the bet
                # - the ratio of the big blind to the bet
                # - the ratio of the bet to the min_bet
                # - the ratio of the bet to the max_bet

                bet_to_stack = safe_div(bet, stack)   # happen when player is all-in
                bet_to_max_stack = safe_div(bet, max_stack, None) # can only happen when every player is all in
                bet_to_pot = bet / pot
                sb_to_bet = safe_div(small_blind, bet)
                bb_to_bet = safe_div(big_blind, bet)
                # adding log features
                log_bet_to_bb = safe_log(safe_div(bet, big_blind))

                bet_to_min_bet = bet / min_bet
                bet_to_max_bet = bet / max_bet

                # compute stacks features
                # - the ratio to the max stack
                # - the ratio of pot to the stack
                # - the ratio of the small blind to the stack
                # - the ratio of the big blind to the stack
                # - the ratio of the min bet to the stack
                # - the ratio of the max bet to the stack
                stack_to_max_stack = safe_div(stack, max_stack, None)
                pot_to_stack = safe_div(pot, stack, 0)
                sb_to_stack = safe_div(small_blind, stack, 0)
                bb_to_stack = safe_div(big_blind, stack, 0)
                min_bet_to_stack = safe_div(min_bet, stack, 0)
                max_bet_to_stack = safe_div(max_bet, stack, 0)
                # adding log features
                log_stack_to_bb = safe_log(safe_div(stack, big_blind))

                features.extend([bet_to_stack, bet_to_max_stack, bet_to_pot, sb_to_bet, bb_to_bet, log_bet_to_bb,
                                 bet_to_min_bet, bet_to_max_bet, stack_to_max_stack, pot_to_stack, sb_to_stack, bb_to_stack,
                                 min_bet_to_stack, max_bet_to_stack, log_stack_to_bb])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0])

        # convert boolean features
        in_hand = torch.tensor(in_hand, dtype=torch.float32).to(self.device).flatten()
        is_button = torch.tensor(is_button, dtype=torch.float32).to(self.device).flatten()
        is_sb = torch.tensor(is_sb, dtype=torch.float32).to(self.device).flatten()
        is_bb = torch.tensor(is_bb, dtype=torch.float32).to(self.device).flatten()
        all_in = torch.tensor(all_in, dtype=torch.float32).to(self.device).flatten()

        # embed relative position to button
        rel_to_button = torch.tensor(rel_to_button, dtype=torch.long).to(self.device)
        rel_to_button_embedding = self.rel_to_button_embedding(rel_to_button)

        # convert the player mask to a tensor so we can use it to zero out non-existent player's embeddings
        player_mask = torch.tensor(player_mask, dtype=torch.float32).to(self.device)
        rel_to_button_embedding = (rel_to_button_embedding * player_mask.unsqueeze(-1)).flatten()

        # we finally concat everything to make the final feature vector
        features = torch.tensor(features, dtype=torch.float32).to(self.device).flatten()

        input_features = torch.cat([features, player_card_embeddings, board_card_embeddings, num_player_embeddings,
                                    in_hand, is_button, is_sb, is_bb, all_in, rel_to_button_embedding,
                                    player_mask], dim=0).to(self.device).contiguous()

        assert input_features.shape[0] == self.expected_input_size(), print(f"Input size {input_features.shape[0]} does"
                                                                             f" not match expected input size "
                                                                             f"{self.expected_input_size()}")
        return input_features


# if __name__ == '__main__':
#     cards = "AcKs"
#     ranks, suits = CardEmbedding.parse_cards(cards)
#     interpreter = Interpreter(device=torch.device("cpu"), rank_dim=16, suit_dim=8)
#     print(interpreter(cards))
