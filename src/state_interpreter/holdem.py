import torch.nn as nn
import torch
from typing import Dict, Union
import pokerkit
from src.global_settings import MAX_TABLE_SIZE
from src.state_interpreter.state_interpreter import StateInterpreter, StatePreprocessor, StateSnapshot, extract_state_snapshot,safe_div, safe_lin_sqrt


class HoldemStatePreprocessor:

    def process(self, state: Union[pokerkit.State, StateSnapshot], current_actor: int) -> Dict[str, list]:
        if isinstance(state, StateSnapshot):
            cards, board_cards = state.hole_cards, state.board_cards
            num_players = state.player_count
            sb, bb = state.blinds_or_straddles
            bets, stacks, in_hand = state.bets, state.stacks, state.in_hand
            pot = sum(bets) + sum(state.pots)
            min_bet, max_bet = state.min_bet, state.max_bet
        else:
            snapshot = extract_state_snapshot(state, current_actor)
            return self.process(snapshot, current_actor)

        if min_bet is None: min_bet = max(bets) if bets else 0
        if max_bet is None: max_bet = min_bet

        # Centering variables
        is_button = [i == num_players - 1 for i in range(num_players)]
        is_sb = [i == 0 for i in range(num_players)]
        is_bb = [i == 1 for i in range(num_players)]
        all_in = [active and stack == 0 for active, stack in zip(in_hand, stacks)]
        rel_to_button = [(num_players - 1 - i) % num_players for i in range(num_players)]

        def center_on_hero(x):
            return x[current_actor:] + x[:current_actor]

        def pad_list(x, pad_val):
            return x + [pad_val] * (self.max_num_players - num_players)

        bets = pad_list(center_on_hero(bets), 0.0)
        stacks = pad_list(center_on_hero(stacks), 0.0)
        in_hand = pad_list(center_on_hero(in_hand), False)
        is_button = pad_list(center_on_hero(is_button), False)
        is_sb = pad_list(center_on_hero(is_sb), False)
        is_bb = pad_list(center_on_hero(is_bb), False)
        all_in = pad_list(center_on_hero(all_in), False)
        rel_to_button = pad_list(center_on_hero(rel_to_button), 0)
        player_mask = pad_list([1.0] * num_players, 0.0)

        # Base Features
        features = []
        max_stack = max(stacks) if max(stacks) > 0 else 1.0

        features.extend([
            safe_div(sb, pot), safe_div(sb, bb), safe_div(bb, pot),
            safe_div(sb, min_bet), safe_div(bb, min_bet), safe_div(min_bet, pot),
            safe_div(min_bet, max_bet), safe_lin_sqrt(safe_div(min_bet, bb)),
            safe_div(sb, max_bet), safe_div(bb, max_bet), safe_lin_sqrt(safe_div(max_bet, bb)),
            safe_div(pot, max_stack), safe_lin_sqrt(safe_div(pot, bb))
        ])

        # Player-specific features
        for i in range(self.max_num_players):
            if player_mask[i]:
                bet, stack = bets[i], stacks[i]
                features.extend([
                    safe_div(bet, stack), safe_div(bet, max_stack), safe_div(bet, pot),
                    safe_div(sb, bet), safe_div(bb, bet), safe_lin_sqrt(safe_div(bet, bb)),
                    safe_div(bet, min_bet), safe_div(bet, max_bet), safe_div(stack, max_stack),
                    safe_div(pot, stack), safe_div(sb, stack), safe_div(bb, stack),
                    safe_div(min_bet, stack), safe_div(max_bet, stack), safe_lin_sqrt(safe_div(stack, bb))
                ])
            else:
                features.extend([0.0] * 15)

        p_ranks, p_suits = self.parse_cards(cards)
        b_ranks, b_suits = self.parse_cards(board_cards)

        return {
            "player_ranks": p_ranks, "player_suits": p_suits,
            "board_ranks": b_ranks, "board_suits": b_suits,
            "num_players": [num_players],
            "float_features": features,
            "in_hand": [float(x) for x in in_hand],
            "is_button": [float(x) for x in is_button],
            "is_sb": [float(x) for x in is_sb],
            "is_bb": [float(x) for x in is_bb],
            "all_in": [float(x) for x in all_in],
            "rel_to_button": rel_to_button,
            "player_mask": player_mask
        }


class HoldemStateInterpreter(StateInterpreter):
    """
    Pure PyTorch module. Takes a batched dictionary of preprocessed numeric features,
    embeds the integers, and concatenates them with the floats on the GPU.
    """

    def __init__(self, device, rank_dim: int = 16, suit_dim: int = 4):
        super().__init__(device, rank_dim, suit_dim)
        self.num_player_embedding_size = 4
        self.rel_to_button_embedding_size = 4
        self.max_num_players = MAX_TABLE_SIZE

        self.num_player_embedding = nn.Embedding(self.max_num_players + 1, self.num_player_embedding_size)
        self.rel_to_button_embedding = nn.Embedding(self.max_num_players + 1, self.rel_to_button_embedding_size)

    def expected_input_size(self):
        size = (self.rank_dim + self.suit_dim) * 2
        size += (self.rank_dim + self.suit_dim) * 5
        size += 13
        size += 15 * self.max_num_players
        size += self.rel_to_button_embedding_size * self.max_num_players
        size += self.num_player_embedding_size
        size += 6 * self.max_num_players
        return size

    def forward(self, preprocessed_batch: Dict[str, torch.Tensor]):
        """Expects a dictionary where values are PyTorch Tensors (batched or unbatched)"""

        # 1. Embeddings
        p_emb = self.card_embedding(preprocessed_batch["player_ranks"], preprocessed_batch["player_suits"])
        b_emb = self.card_embedding(preprocessed_batch["board_ranks"], preprocessed_batch["board_suits"])

        num_emb = self.num_player_embedding(preprocessed_batch["num_players"]).squeeze(-2)

        rel_emb = self.rel_to_button_embedding(preprocessed_batch["rel_to_button"])

        # Multiply rel_to_button embeddings by the player mask to zero out padding
        mask = preprocessed_batch["player_mask"].unsqueeze(-1)
        rel_emb = (rel_emb * mask).flatten(start_dim=-2)

        # 2. Concatenate all float vectors
        concat_list = [
            preprocessed_batch["float_features"],
            p_emb,
            b_emb,
            num_emb,
            preprocessed_batch["in_hand"],
            preprocessed_batch["is_button"],
            preprocessed_batch["is_sb"],
            preprocessed_batch["is_bb"],
            preprocessed_batch["all_in"],
            rel_emb,
            preprocessed_batch["player_mask"]
        ]

        # Use dim=-1 so this safely handles both single vectors (dim=0) and batched inputs (dim=1)
        input_features = torch.cat(concat_list, dim=-1)

        return input_features

