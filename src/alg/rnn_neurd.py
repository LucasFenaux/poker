import torch
import copy
from torch.distributions import Categorical
from torch.nn.utils.rnn import pad_sequence
from .neurd import NeuRD
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

        with torch.no_grad():
            (decision_logits, bet_logits), new_hand_hidden = network(*state_args, hand_hidden=hand_hidden, game_hidden=game_hidden)

        
        if bet_logits is not None:
            return (Categorical(logits=decision_logits), Categorical(logits=bet_logits)), new_hand_hidden
        else:
            return (Categorical(logits=decision_logits), None), new_hand_hidden


class RNNNeuRD(NeuRD):
    default_hyperparameters = {
        "mini_batch_size": 500,
        "lr": 2e-3,
        "value_lr": 1e-2,
        "grad_clip_norm": 0.5,
        "reward_normalization_scaler": 1,
        "entropy_coef": 0.1,
        "beta": 2.0,
        "critic_update_ratio": 4,
        "target_network_update_freq": 100,
        "target_network_reg_weight": 0.01,
        "logit_penalty_weight": 100.0,
    }
    def __init__(self, lr, device, value_lr, reward_normalization_scaler, grad_clip_norm, mini_batch_size, mode="categorical", discrete=True, **kwargs):
        super(RNNNeuRD, self).__init__(lr, device, value_lr, reward_normalization_scaler, grad_clip_norm, mini_batch_size, mode, discrete, **kwargs)
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

    def _unroll_logits(self, network, states_dict, h_0, g_0, new_hand_mask):
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
                    updated_g = network.update_game_memory(h, g)
                    g = torch.where(mask, updated_g, g)

                if max_elapsed > 1:
                    zeros_h = torch.zeros_like(h)
                    for elapsed in range(2, max_elapsed + 1):
                        mask = (elapsed_hands >= elapsed).unsqueeze(1)
                        updated_g = network.update_game_memory(zeros_h, g)
                        g = torch.where(mask, updated_g, g)

            h = torch.where((elapsed_hands >= 1).unsqueeze(1), torch.zeros_like(h), h)

            step_dict = {k: v[:, t] for k, v in states_dict.items()}

            (decision_logits, bet_logits), h = network(step_dict, hand_hidden=h, game_hidden=g)

            all_logits.append((decision_logits, bet_logits))

        stacked_decision_logits = torch.stack([p[0] for p in all_logits], dim=1)
        
        stacked_bet_logits = None
        if all_logits[0][1] is not None:
            stacked_bet_logits = torch.stack([p[1] for p in all_logits], dim=1)

        return stacked_decision_logits, stacked_bet_logits

        h_0 = torch.zeros(num_sequences, self.network.hand_memory_size, device=self.device)
        g_0 = torch.zeros(num_sequences, self.network.game_memory_size, device=self.device)

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


            self.value_optimizer.zero_grad()
            q_dec, q_bet = self._unroll_logits(self.value_network, mb_states_dict, mb_h_0, mb_g_0, mb_new_hands)
            
            with torch.no_grad():
                target_q_dec, target_q_bet = self._unroll_logits(self.target_value_network, mb_states_dict, mb_h_0, mb_g_0, mb_new_hands)
                
            action_dec_indices = mb_actions[..., 0] if q_bet is not None else mb_actions
            
            # 1. Main loss (only on taken actions)
            q_dec_taken = q_dec.gather(dim=-1, index=action_dec_indices.unsqueeze(-1)).squeeze(-1)
            
            main_loss_dec = torch.nn.functional.smooth_l1_loss(q_dec_taken, mb_rewards, reduction="none")
            main_loss_dec = (main_loss_dec * mb_mask).sum() / mb_mask.sum()
            
            reg_loss_dec = torch.nn.functional.smooth_l1_loss(q_dec, target_q_dec, reduction="none")
            reg_loss_dec = (reg_loss_dec.mean(dim=-1) * mb_mask).sum() / mb_mask.sum()
            
            value_loss_dec = main_loss_dec + self.target_network_reg_weight * reg_loss_dec
            
            if q_bet is not None:
                q_bet_taken = q_bet.gather(dim=-1, index=mb_actions[..., 1].unsqueeze(-1)).squeeze(-1)
                
                main_loss_bet = torch.nn.functional.smooth_l1_loss(q_bet_taken, mb_rewards, reduction="none")
                main_loss_bet = (main_loss_bet * mb_mask).sum() / mb_mask.sum()
                
                reg_loss_bet = torch.nn.functional.smooth_l1_loss(q_bet, target_q_bet, reduction="none")
                reg_loss_bet = (reg_loss_bet.mean(dim=-1) * mb_mask).sum() / mb_mask.sum()
                
                value_loss_bet = main_loss_bet + self.target_network_reg_weight * reg_loss_bet
                value_loss = value_loss_dec + value_loss_bet
            else:
                value_loss = value_loss_dec
                            
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.grad_clip_norm)
            self.value_optimizer.step()

            self.optimizer.zero_grad()
            
            with torch.no_grad():
                old_decision_logits, old_bet_logits = self._unroll_logits(self.network, mb_states_dict, mb_h_0, mb_g_0, mb_new_hands)

            self.total_update_count += 1
            if self.total_update_count % self.critic_update_ratio == 0:
                decision_logits, bet_logits = self._unroll_logits(self.network, mb_states_dict, mb_h_0, mb_g_0, mb_new_hands)
                
                # Calculate V(s) = sum_a pi(a|s) Q(s,a) to compute advantages for all actions
                with torch.no_grad():
                    pi_dec = Categorical(logits=decision_logits).probs
                    reg_q_dec = q_dec - self.entropy_coef * torch.log(pi_dec.clamp(min=1e-8))
                    v_dec = torch.sum(pi_dec * reg_q_dec, dim=-1)
                    
                policy_loss_unreduced_dec = -torch.sum(decision_logits * (reg_q_dec.detach() - v_dec.unsqueeze(-1).detach()), dim=-1)
                bound_penalty_unreduced_dec = torch.nn.functional.relu(decision_logits.abs() - self.beta).pow(2).sum(dim=-1)

                if bet_logits is not None:
                    with torch.no_grad():
                        pi_bet = Categorical(logits=bet_logits).probs
                        reg_q_bet = q_bet - self.entropy_coef * torch.log(pi_bet.clamp(min=1e-8))
                        v_bet = torch.sum(pi_bet * reg_q_bet, dim=-1)
                    policy_loss_unreduced_bet = -torch.sum(bet_logits * (reg_q_bet.detach() - v_bet.unsqueeze(-1).detach()), dim=-1)
                    bound_penalty_unreduced_bet = torch.nn.functional.relu(bet_logits.abs() - self.beta).pow(2).sum(dim=-1)
                    
                    policy_loss_unreduced = policy_loss_unreduced_dec + policy_loss_unreduced_bet
                    bound_penalty_unreduced = bound_penalty_unreduced_dec + bound_penalty_unreduced_bet
                else:
                    policy_loss_unreduced = policy_loss_unreduced_dec
                    bound_penalty_unreduced = bound_penalty_unreduced_dec

                policy_loss = (policy_loss_unreduced * mb_mask).sum() / mb_mask.sum()
                bound_penalty = (bound_penalty_unreduced * mb_mask).sum() / mb_mask.sum()
                
                total_loss = policy_loss + self.logit_penalty_weight * bound_penalty

                if not torch.isfinite(total_loss):
                    print("WARNING: loss is not finite")
                    
                total_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
                self.optimizer.step()
                policy_loss = policy_loss.item()
            else:
                policy_loss = 0.0

            if self.total_update_count % self.target_network_update_freq == 0:
                self.target_value_network_update()

            avg_v_loss += value_loss.item()
            avg_p_loss += policy_loss
            count += 1

        if count != 0:
            value_loss = avg_v_loss / count
            policy_loss = avg_p_loss / count
        else:
            value_loss = avg_v_loss
            policy_loss = avg_p_loss

        with torch.no_grad():
            action_logits, bet_logits = self._unroll_logits(self.network, batched_states_dict, h_0, g_0, padded_new_hands)
            
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
