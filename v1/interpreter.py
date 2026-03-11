"""
Converts model actions into table instructions that are understood by the TableActor.
Transforms state information provided by the TableActor into a state vector for the model using nn.Embedding models.
It is trainable and should be part of the model.
"""
import torch
import torch.nn as nn
import pokerkit

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



class Interpreter(nn.Module):
    def __init__(self, device, rank_dim: int = 16, suit_dim: int = 4, max_num_players: int = 9):
        super().__init__()
        self.device = device

        self.rank_dim = rank_dim
        self.suit_dim = suit_dim
        self.card_embedding = CardEmbedding(rank_dim, suit_dim).to(device)

        self.max_num_players = max_num_players
        self.num_player_embedding = nn.Embedding(max_num_players, 4)



    def forward(self, state: pokerkit.State, current_actor: int):
        # that player's cards
        cards = state.hole_cards[current_actor]
        cards = "".join([repr(card) for card in cards])

        # the current board (+ any unknown cards)
        board_cards = state.board_cards[0]  # we will never encounter multiple boards when making decisions
        to_add = 5 - len(board_cards)

        board_cards = "".join([repr(card) for card in board_cards] + ["??"] * to_add)

        num_players = state.player_count

        # blind info
        sb, bb = state.blinds_or_straddles

        # pot info
        pot = sum(list(state.pot_amounts))  # for now ignore side-pots TODO: handle side-pots

        # player info
        # collecting all the info
        bets = state.bets
        stacks = state.stacks
        in_hand = state.statuses
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
                                   small_blind=sb, big_blind=bb, pot=pot, bets=bets, stacks=stacks, in_hand=in_hand,
                                   is_button=is_button, is_sb=is_sb, is_bb=is_bb, all_in=all_in,
                                   rel_to_button=rel_to_button, player_mask=player_mask)


    def _embed_forward(self, player_cards: str,  # need to embed
                       board_cards: str,   # need to embed
                       num_players: int,   # need to embed
                       small_blind: float,  # need to compute features from
                       big_blind: float,   # need to compute features from
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

        # embed player cards
        player_ranks, player_suits = self.card_embedding.parse_cards(player_cards)
        player_ranks, player_suits = player_ranks.to(self.device), player_suits.to(self.device)
        player_card_embeddings = self.card_embedding(player_ranks, player_suits)

        # embed board cards
        board_ranks, board_suits = self.card_embedding.parse_cards(board_cards)
        board_ranks, board_suits = board_ranks.to(self.device), board_suits.to(self.device)
        board_card_embeddings = self.card_embedding(board_ranks, board_suits)

        # embed num players
        num_player_embeddings = self.num_player_embedding(num_players)

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

        # compute pot features
        # we compute the ratio of the pot to the max stack
        pot_to_max_stack = pot / max_stack

        features.append(pot_to_max_stack)

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

                bet_to_stack = bet / stack
                bet_to_max_stack = bet / max_stack
                bet_to_pot = bet / pot
                sb_to_bet = small_blind / bet
                bb_to_bet = big_blind / bet

                # compute stacks features
                # - the ratio to the max stack
                # - the ratio of pot to the stack
                # - the ratio of the small blind to the stack
                # - the ratio of the big blind to the stack
                stack_to_max_stack = stack / max_stack
                pot_to_stack = pot / stack
                sb_to_stack = small_blind / stack
                bb_to_stack = big_blind / stack

                features.extend([bet_to_stack, bet_to_max_stack, bet_to_pot, sb_to_bet, bb_to_bet, stack_to_max_stack,
                                 pot_to_stack, sb_to_stack, bb_to_stack])
            else:
                features.extend([0, 0, 0, 0, 0, 0, 0, 0, 0])


        # convert boolean features

        # embed relative position to button

        # append player mask and multiply embeddings by player mask to zero embeddings

        # convert the player mask to a tensor so we can use it to zero out non-existent player's embeddings
        player_mask = torch.tensor(player_mask, dtype=torch.float32).to(self.device)

# if __name__ == '__main__':
#     cards = "AcKs"
#     ranks, suits = CardEmbedding.parse_cards(cards)
#     interpreter = Interpreter(device=torch.device("cpu"), rank_dim=16, suit_dim=8)
#     print(interpreter(cards))
