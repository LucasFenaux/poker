from .state_interpreter import StatePreprocessor, StateInterpreter, StateSnapshot, extract_state_snapshot
import pokerkit
import torch
from typing import Union, Dict

class KuhnStatePreprocessor(StatePreprocessor):
    def process(self, state: Union[pokerkit.State, StateSnapshot], current_actor: int) -> Dict[str, list]:
        if isinstance(state, StateSnapshot):
            snapshot = state
        else:
            snapshot = extract_state_snapshot(state, current_actor)

        cards = snapshot.hole_cards
        p_ranks, p_suits = self.parse_cards(cards)
        if len(p_ranks) == 0:
            p_ranks, p_suits = [13], [4]

        bets = snapshot.bets
        if not bets:
            bets = [0.0, 0.0]
        elif len(bets) < 2:
            bets = bets + [0.0] * (2 - len(bets))

        bet = float(bets[current_actor])
        opp_bet = float(bets[1 - current_actor])

        pot = float(sum(bets) + sum(snapshot.pots) if snapshot.pots else sum(bets))

        return {
            "player_ranks": p_ranks[:1],
            "player_suits": p_suits[:1],
            "float_features": [bet, opp_bet, pot, float(current_actor)],
        }


class KuhnStateInterpreter(StateInterpreter):
    def expected_input_size(self):
        return (16 + 4) * 1 + 4

    def forward(self, preprocessed_batch: Dict[str, torch.Tensor]):
        p_emb = self.card_embedding(preprocessed_batch["player_ranks"], preprocessed_batch["player_suits"])
        return torch.cat([preprocessed_batch["float_features"], p_emb], dim=-1)