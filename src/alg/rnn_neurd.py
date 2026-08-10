from typing import Union
import numpy as np
import torch
from torch.distributions import Categorical
import pokerkit
from torch.nn.utils.rnn import pad_sequence
from src.action_interpreter import Action
from src.models import HierarchicalPokerModel
from .neurd import NeuRD, NeuRDInferenceWrapper
from .rnn_ppo import RNNPPOInferenceWrapper


class RNNNeuRDInferenceWrapper(RNNPPOInferenceWrapper):
    def get_model_policy(self, network, state, hand_hidden: torch.Tensor = None, game_hidden: torch.Tensor = None):
        if isinstance(state, tuple) and len(state) == 2 and not isinstance(state[0], dict):
            s, a = state
            batched_dict = self.preprocess_batch([s], [a])
            state_args = (batched_dict,)
        else:
            state_args = state

        if hand_hidden is None:
            hand_hidden = self.init_hand_memory(batch_size=1)
        else:
            hand_hidden = hand_hidden.to(self.device)

        if game_hidden is None:
            game_hidden = self.init_game_memory(batch_size=1)
        else:
            game_hidden = game_hidden.to(self.device)

        (decision_logits, bet_logits), new_hand_hidden = network(*state_args, hand_hidden=hand_hidden, game_hidden=game_hidden)
        
        if bet_logits is not None:
            return (Categorical(logits=decision_logits), Categorical(logits=bet_logits)), new_hand_hidden
        else:
            return (Categorical(logits=decision_logits), None), new_hand_hidden


class RNNNeuRD(NeuRD):
    def __init__(self, lr, device, value_lr, reward_normalization_scaler, grad_clip_norm, mini_batch_size, mode="categorical", discrete=True):
        super(RNNNeuRD, self).__init__(lr, device, value_lr, reward_normalization_scaler, grad_clip_norm, mini_batch_size, mode, discrete)
        self.hand_memory_sizes = [self.network.hand_memory_size, self.value_network.hand_memory_size]
        self.game_memory_sizes = [self.network.game_memory_size, self.value_network.game_memory_size]

    def init_hand_memory(self, batch_size: int = 1):
        return [torch.zeros(batch_size, self.network.hand_memory_size, device=self.device), 
                torch.zeros(batch_size, self.value_network.hand_memory_size, device=self.device)]

    def init_game_memory(self, batch_size: int = 1):
        return [torch.zeros(batch_size, self.network.game_memory_size, device=self.device),
                torch.zeros(batch_size, self.value_network.game_memory_size, device=self.device)]

    def preprocess_sequence_batch(self, nested_states, nested_actors):
        """Preprocesses nested lists [games[steps]] into a padded dictionary of tensors."""
        flat_states = [s for game in nested_states for s in game]
        flat_actors = [a for game in nested_actors for a in game]

        flat_tensor_dict = super().preprocess_batch(flat_states, flat_actors)

        padded_dict = {}
        seq_lengths = [len(game) for game in nested_states]
        max_seq_len = max(seq_lengths)
        batch_size = len(nested_states)

        mask = torch.arange(max_seq_len, device=self.device).expand(batch_size, max_seq_len) < torch.tensor(seq_lengths, device=self.device).unsqueeze(1)

        start_idx = 0
        for k, v in flat_tensor_dict.items():
            sequences = []
            current_idx = 0
            for length in seq_lengths:
                sequences.append(v[current_idx: current_idx + length])
                current_idx += length

            padded_dict[k] = pad_sequence(sequences, batch_first=True, padding_value=0.0)

        return padded_dict, mask

    def _unroll_logits(self, states_dict, h_0, g_0, new_hand_mask):
        max_seq_len = next(iter(states_dict.values())).size(1)
        all_logits = []
        h = h_0
        g = g_0

        for t in range(max_seq_len):
            elapsed_hands = new_hand_mask[:, t]
            max_elapsed = int(elapsed_hands.max().item())

            if max_elapsed > 0:
                if t > 0:
                    mask = (elapsed_hands >= 1).unsqueeze(1)
                    updated_g = self.network.update_game_memory(h, g)
                    g = torch.where(mask, updated_g, g)

                if max_elapsed > 1:
                    zeros_h = torch.zeros_like(h)
                    for elapsed in range(2, max_elapsed + 1):
                        mask = (elapsed_hands >= elapsed).unsqueeze(1)
                        updated_g = self.network.update_game_memory(zeros_h, g)
                        g = torch.where(mask, updated_g, g)

            h = torch.where((elapsed_hands >= 1).unsqueeze(1), torch.zeros_like(h), h)

            step_dict = {k: v[:, t] for k, v in states_dict.items()}

            (decision_logits, bet_logits), h = self.network(step_dict, hand_hidden=h, game_hidden=g)

            all_logits.append((decision_logits, bet_logits))

        stacked_decision_logits = torch.stack([p[0] for p in all_logits], dim=1)
        
        stacked_bet_logits = None
        if all_logits[0][1] is not None:
            stacked_bet_logits = torch.stack([p[1] for p in all_logits], dim=1)

        return stacked_decision_logits, stacked_bet_logits

    def _unroll_value(self, states_dict, h_0, g_0, new_hand_mask):
        max_seq_len = next(iter(states_dict.values())).size(1)
        all_values = []
        h = h_0
        g = g_0

        for t in range(max_seq_len):
            elapsed_hands = new_hand_mask[:, t]
            max_elapsed = int(elapsed_hands.max().item())

            if max_elapsed > 0:
                if t > 0:
                    mask = (elapsed_hands >= 1).unsqueeze(1)
                    updated_g = self.value_network.update_game_memory(h, g)
                    g = torch.where(mask, updated_g, g)

                if max_elapsed > 1:
                    zeros_h = torch.zeros_like(h)
                    for elapsed in range(2, max_elapsed + 1):
                        mask = (elapsed_hands >= elapsed).unsqueeze(1)
                        updated_g = self.value_network.update_game_memory(zeros_h, g)
                        g = torch.where(mask, updated_g, g)

            h = torch.where((elapsed_hands >= 1).unsqueeze(1), torch.zeros_like(h), h)

            step_dict = {k: v[:, t] for k, v in states_dict.items()}
            val, h = self.value_network(step_dict, hand_hidden=h, game_hidden=g)
            all_values.append(val)

        return torch.stack(all_values, dim=1)

    def update(self, batch_states, batch_rewards, batch_actions, batch_rnn_states=None, sample_weights=None, *args, **kwargs):
        nested_states, nested_actors = batch_states
        num_sequences = len(nested_states)

        batched_states_dict, mask = self.preprocess_sequence_batch(nested_states, nested_actors)

        padded_actions = pad_sequence(
            [torch.stack(a).to(dtype=torch.long, device=self.device) for a in batch_actions], 
            batch_first=True
        )

        padded_rewards = pad_sequence(
            [torch.tensor(r, dtype=torch.float32, device=self.device) for r in batch_rewards],
            batch_first=True
        )

        nested_new_hands = kwargs.get("new_hands", [])
        padded_new_hands = pad_sequence(
            [torch.tensor(nh, dtype=torch.long, device=self.device) for nh in nested_new_hands],
            batch_first=True, padding_value=0
        )

        if sample_weights is not None:
            sample_weights = torch.tensor(sample_weights, device=self.device)
            assert sample_weights.dim() == 1
            normalized_sample_weights = sample_weights / sample_weights.mean()
            prob_sample_weights = sample_weights / sample_weights.sum()

        h_0 = torch.zeros(num_sequences, self.network.hand_memory_size, device=self.device)
        g_0 = torch.zeros(num_sequences, self.network.game_memory_size, device=self.device)

        with torch.no_grad():
            value_function = self._unroll_value(batched_states_dict, h_0, g_0, padded_new_hands).squeeze(-1)
            advantages = padded_rewards - value_function.clone().detach()

        count = 0
        avg_v_loss = 0
        avg_p_loss = 0

        indices = torch.randperm(num_sequences, device=self.device)
        for start_idx in range(0, num_sequences, self.mini_batch_size):
            mini_batch_indices = indices[start_idx:start_idx + self.mini_batch_size]

            mb_states_dict = {k: v[mini_batch_indices] for k, v in batched_states_dict.items()}
            mb_mask = mask[mini_batch_indices]
            mb_rewards = padded_rewards[mini_batch_indices]
            mb_advantages = advantages[mini_batch_indices]
            mb_actions = padded_actions[mini_batch_indices]

            mb_h_0 = h_0[mini_batch_indices]
            mb_g_0 = g_0[mini_batch_indices]
            mb_new_hands = padded_new_hands[mini_batch_indices]

            if sample_weights is not None:
                mb_sample_weights = normalized_sample_weights[mini_batch_indices]

            self.value_optimizer.zero_grad()
            value_function = self._unroll_value(mb_states_dict, mb_h_0, mb_g_0, mb_new_hands).squeeze(-1)
            
            v_loss_unreduced = torch.nn.functional.smooth_l1_loss(value_function, mb_rewards, reduction="none")
            if sample_weights is None:
                value_loss = (v_loss_unreduced * mb_mask).sum() / mb_mask.sum()
            else:
                value_loss = (v_loss_unreduced * mb_mask * mb_sample_weights.unsqueeze(-1)).sum() / mb_mask.sum()
                
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.grad_clip_norm)
            self.value_optimizer.step()

            self.optimizer.zero_grad()

            decision_logits, bet_logits = self._unroll_logits(mb_states_dict, mb_h_0, mb_g_0, mb_new_hands)

            if bet_logits is not None:
                decision_actions = mb_actions[..., 0].unsqueeze(-1)
                decision_logits = torch.gather(decision_logits, -1, decision_actions)
                bet_actions = mb_actions[..., 1].unsqueeze(-1)
                bet_logits = torch.gather(bet_logits, -1, bet_actions)
                logits = torch.cat((decision_logits, bet_logits), dim=-1)
            else:
                decision_actions = mb_actions.unsqueeze(-1)
                decision_logits = torch.gather(decision_logits, -1, decision_actions)
                logits = decision_logits

            policy_loss_unreduced = -logits * mb_advantages.unsqueeze(-1)

            if sample_weights is None:
                policy_loss = (policy_loss_unreduced * mb_mask.unsqueeze(-1)).sum() / mb_mask.sum()
            else:
                policy_loss = (policy_loss_unreduced * mb_mask.unsqueeze(-1) * mb_sample_weights.unsqueeze(-1).unsqueeze(-1)).sum() / mb_mask.sum()

            if not torch.isfinite(policy_loss):
                print("WARNING: loss is not finite")
            policy_loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)

            self.optimizer.step()
            avg_v_loss += value_loss.item()
            avg_p_loss += policy_loss.item()
            count += 1

        if count != 0:
            value_loss = avg_v_loss / count
            policy_loss = avg_p_loss / count
        else:
            value_loss = avg_v_loss
            policy_loss = avg_p_loss

        with torch.no_grad():
            action_logits, bet_logits = self._unroll_logits(batched_states_dict, h_0, g_0, padded_new_hands)
            
            valid_action_logits = action_logits[mask]
            action_hist = Categorical(logits=valid_action_logits).sample().float()
            
            if bet_logits is not None:
                valid_bet_logits = bet_logits[mask]
                bet_hist = Categorical(logits=valid_bet_logits).sample().float()
            else:
                bet_hist = None

        return {"loss": policy_loss, "value_loss": value_loss, "policy_loss": policy_loss,
                "action_hist": action_hist, "betting_size": bet_hist, "rewards": padded_rewards[mask],
                "update_count": count}
