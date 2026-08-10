
from typing import Union
import numpy as np
import torch
from torch.distributions import Categorical, Normal
import pokerkit
from torch.nn.utils.rnn import pad_sequence
from src.models import get_value_model, load_dummy_model
from src.action_interpreter import Action
from src.state_interpreter import safe_log, safe_lin_sqrt
from src.models import HierarchicalPokerModel
from torch.distributions.beta import Beta
from .ppo import PPO, PPOInferenceWrapper


class RNNPPOInferenceWrapper(PPOInferenceWrapper):
    def __init__(self, models, discrete: bool = False):
        super().__init__(models, discrete)
        self.network: HierarchicalPokerModel
        self.hand_memory_size = self.network.hand_memory_size
        self.game_memory_size = self.network.game_memory_size

    def init_hand_memory(self, batch_size: int = 1):
        # FIX: Use inherited self.device
        return torch.zeros(batch_size, self.hand_memory_size, device=self.device)

    def init_game_memory(self, batch_size: int = 1):
        return torch.zeros(batch_size, self.game_memory_size, device=self.device)

    def update_game_memory(self, final_hand_hidden: torch.Tensor, current_game_hidden: torch.Tensor):
        # FIX: Ensure tensors are on the GPU before passing to the network
        if final_hand_hidden is not None:
            final_hand_hidden = final_hand_hidden.to(self.device)
        if current_game_hidden is not None:
            current_game_hidden = current_game_hidden.to(self.device)

        new_game_hidden = self.network.update_game_memory(final_hand_hidden, current_game_hidden)
        # FIX: Detach and send to CPU to prevent massive GPU memory leaks in TableActor
        return new_game_hidden.detach().cpu()

    def get_action(self, state: tuple[pokerkit.State, int], hand_hidden: torch.Tensor = None,
                   game_hidden: torch.Tensor = None):
        (action_dist, bet_sizing_dist), new_hand_hidden = self.get_model_policy(self.network, state,
                                                                                hand_hidden=hand_hidden,
                                                                                game_hidden=game_hidden)
        action = action_dist.sample().cpu()
        if bet_sizing_dist is not None:
            bet_size = bet_sizing_dist.sample().cpu()
            action_tensor = torch.stack([action, bet_size], dim=-1).squeeze(0)
        else:
            action_tensor = action.squeeze(0)
        # Detach and send memory to CPU so the TableActor doesn't hoard GPU RAM
        return action_tensor, new_hand_hidden.detach().cpu()

    def get_model_policy(self, network, state, hand_hidden: torch.Tensor = None, game_hidden: torch.Tensor = None) -> \
            tuple[Union[Categorical, Normal], torch.Tensor]:
        if isinstance(state, tuple) and len(state) == 2 and not isinstance(state[0], dict):
            s, a = state
            batched_dict = self.preprocess_batch([s], [a])
            state_args = (batched_dict,)
        else:
            state_args = state

        # Initialize or move memories to the active device
        if hand_hidden is None:
            hand_hidden = self.init_hand_memory(batch_size=1)
        else:
            hand_hidden = hand_hidden.to(self.device)

        if game_hidden is None:
            game_hidden = self.init_game_memory(batch_size=1)
        else:
            game_hidden = game_hidden.to(self.device)

        # Retrieve logits/distribution alongside the updated RNN memory
        (action_dist, bet_sizing_dist), new_hand_hidden = network(*state_args, hand_hidden=hand_hidden,
                                                                  game_hidden=game_hidden)
        return (action_dist, bet_sizing_dist), new_hand_hidden


class RNNPPO(PPO):
    default_hyperparameters = {
        "sgd_steps": 5,
        "base_batch_size": 500,
        "mini_batch_size": 10,  # number of games in one mini batch
        "clip_threshold": 0.2,
        "target_kl": 0.01,
        "lr": 1e-4,
        "value_lr": 5e-4,
        "reward_normalization_scaler": 1,
        "entropy_coef": 1e-4,
        "grad_clip_norm": 0.5,
        "update_mode": "mini_batch"
    }
    key = "ppo"

    def __init__(self, lr, device, value_lr, sgd_steps, clip_threshold, target_kl, reward_normalization_scaler,
                 grad_clip_norm, entropy_coef, base_batch_size, mini_batch_size, update_mode, mode,
                 discrete: bool = False):
        super(RNNPPO, self).__init__(lr, device, value_lr, sgd_steps, clip_threshold, target_kl,
                                     reward_normalization_scaler,
                                     grad_clip_norm, entropy_coef, base_batch_size, mini_batch_size, update_mode, mode,
                                     discrete)
        self.models = self.init_networks(device, discrete, mode)
        self.hand_memory_sizes = [model.hand_memory_size for model in self.models]
        self.game_memory_sizes = [model.game_memory_size for model in self.models]

    def init_hand_memory(self, batch_size: int = 1):
        return [torch.zeros(batch_size, self.hand_memory_sizes[i], device=model.device) for i, model in
                enumerate(self.models)]

    def init_game_memory(self, batch_size: int = 1):
        return [torch.zeros(batch_size, self.game_memory_sizes[i], device=model.device) for i, model in
                enumerate(self.models)]

    def preprocess_sequence_batch(self, nested_states, nested_actors):
        """Preprocesses nested lists [games[steps]] into a padded dictionary of tensors."""
        flat_states = [s for game in nested_states for s in game]
        flat_actors = [a for game in nested_actors for a in game]

        flat_tensor_dict = super().preprocess_batch(flat_states, flat_actors)

        padded_dict = {}
        seq_lengths = [len(game) for game in nested_states]
        max_seq_len = max(seq_lengths)
        batch_size = len(nested_states)

        mask = torch.arange(max_seq_len, device=self.device).expand(batch_size, max_seq_len) < torch.tensor(seq_lengths,
                                                                                                            device=self.device).unsqueeze(
            1)

        start_idx = 0
        for k, v in flat_tensor_dict.items():
            sequences = []
            current_idx = 0
            for length in seq_lengths:
                sequences.append(v[current_idx: current_idx + length])
                current_idx += length

            padded_dict[k] = pad_sequence(sequences, batch_first=True, padding_value=0.0)

        return padded_dict, mask

    def _unroll_policy(self, states_dict, h_0, g_0, new_hand_mask):
        max_seq_len = next(iter(states_dict.values())).size(1)
        all_dist_params = []
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

            (action_dist, bet_sizing_dist), h = self.network(step_dict, hand_hidden=h, game_hidden=g)

            action_logits = action_dist.logits
            bet_params = None
            if bet_sizing_dist is not None:
                if self.mode == "normal":
                    bet_params = (bet_sizing_dist.loc, bet_sizing_dist.scale)
                elif self.mode == "beta":
                    bet_params = (bet_sizing_dist.concentration1, bet_sizing_dist.concentration0)
            all_dist_params.append((action_logits, bet_params))

        stacked_action_logits = torch.stack([p[0] for p in all_dist_params], dim=1)
        action_dist = Categorical(logits=stacked_action_logits)

        bet_sizing_dist = None
        if all_dist_params[0][1] is not None:
            if self.mode == "normal":
                mu = torch.stack([p[1][0] for p in all_dist_params], dim=1)
                std = torch.stack([p[1][1] for p in all_dist_params], dim=1)
                bet_sizing_dist = Normal(mu, std)
            elif self.mode == "beta":
                alpha = torch.stack([p[1][0] for p in all_dist_params], dim=1)
                beta = torch.stack([p[1][1] for p in all_dist_params], dim=1)
                bet_sizing_dist = Beta(alpha, beta)

        return action_dist, bet_sizing_dist

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

    def mini_batch_update(self, batch_states, batch_rewards, batch_actions, batch_rnn_states=None, sample_weights=None,
                          *args, **kwargs):
        nested_states, nested_actors = batch_states
        num_sequences = len(nested_states)

        batched_states_dict, mask = self.preprocess_sequence_batch(nested_states, nested_actors)

        padded_actions = pad_sequence(
            [torch.stack(a).to(dtype=torch.float32, device=self.device) for a in
             batch_actions], batch_first=True)

        safe_actions = padded_actions.clone()
        if safe_actions.shape[-1] > 1 and safe_actions.dim() > 2:
            safe_actions[..., 1:] = torch.clamp(safe_actions[..., 1:], min=1e-2, max=1.0 - 1e-2)

        padded_rewards = pad_sequence([torch.tensor(r, dtype=torch.float32, device=self.device) for r in batch_rewards],
                                      batch_first=True)

        nested_new_hands = kwargs.get("new_hands", [])
        padded_new_hands = pad_sequence(
            [torch.tensor(nh, dtype=torch.long, device=self.device) for nh in nested_new_hands],
            batch_first=True, padding_value=0)

        # Initialize Base Memory for GRU
        h_0 = torch.zeros(num_sequences, self.network.hand_memory_size, device=self.device)
        g_0 = torch.zeros(num_sequences, self.network.game_memory_size, device=self.device)

        with torch.no_grad():
            action_dist_old, bet_sizing_dist_old = self._unroll_policy(batched_states_dict, h_0, g_0, padded_new_hands)

            if bet_sizing_dist_old is not None:
                logp_old = action_dist_old.log_prob(safe_actions[..., 0].long())
                logp_bet_old = bet_sizing_dist_old.log_prob(safe_actions[..., 1])
                is_raise = (safe_actions[..., 0].long() == Action.RAISE.value).float()
                logp_old = logp_old + (logp_bet_old * is_raise)
            else:
                logp_old = action_dist_old.log_prob(safe_actions.long())

            value_function = self._unroll_value(batched_states_dict, h_0, g_0, padded_new_hands).squeeze(-1)
            advantages = padded_rewards - value_function.clone().detach()

            valid_advantages = advantages[mask]
            adv_mean = valid_advantages.mean()
            adv_std = valid_advantages.std()
            advantages = (advantages - adv_mean) / (adv_std + 1e-8)

        avg_loss = avg_v_loss = avg_p_loss = avg_e_loss = count = 0

        for i in range(self.sgd_steps):
            indices = torch.randperm(num_sequences, device=self.device)

            for start_idx in range(0, num_sequences, self.mini_batch_size):
                mini_batch_indices = indices[start_idx: start_idx + self.mini_batch_size]

                mb_states_dict = {k: v[mini_batch_indices] for k, v in batched_states_dict.items()}
                mb_mask = mask[mini_batch_indices]
                mb_rewards = padded_rewards[mini_batch_indices]
                mb_advantages = advantages[mini_batch_indices]
                mb_safe_actions = safe_actions[mini_batch_indices]
                mb_logp_old = logp_old[mini_batch_indices]

                mb_h_0 = h_0[mini_batch_indices]
                mb_g_0 = g_0[mini_batch_indices]
                mb_new_hands = padded_new_hands[mini_batch_indices]

                # ---- Value Update ----
                self.value_optimizer.zero_grad()
                value_function = self._unroll_value(mb_states_dict, mb_h_0, mb_g_0, mb_new_hands).squeeze(-1)

                v_loss_unreduced = torch.nn.functional.smooth_l1_loss(value_function, mb_rewards, reduction="none")
                value_loss = (v_loss_unreduced * mb_mask).sum() / mb_mask.sum()

                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.grad_clip_norm)
                self.value_optimizer.step()

                # ---- Policy Update ----
                self.optimizer.zero_grad()
                action_dist, bet_sizing_dist = self._unroll_policy(mb_states_dict, mb_h_0, mb_g_0, mb_new_hands)

                entropy = action_dist.entropy()
                if bet_sizing_dist is not None:
                    logp = action_dist.log_prob(mb_safe_actions[..., 0].long())
                    logp_bet = bet_sizing_dist.log_prob(mb_safe_actions[..., 1])
                    entropy_bet = bet_sizing_dist.entropy()
                    is_raise = (mb_safe_actions[..., 0].long() == Action.RAISE.value).float()
                    logp = logp + (logp_bet * is_raise)
                    entropy = entropy + (entropy_bet * is_raise)
                else:
                    logp = action_dist.log_prob(mb_safe_actions.long())

                with torch.no_grad():
                    logratio_no_grad = logp - mb_logp_old
                    ratio_no_grad = torch.exp(logratio_no_grad)
                    approx_kl = (((ratio_no_grad - 1) - logratio_no_grad) * mb_mask).sum() / mb_mask.sum()

                if approx_kl.item() > 1.5 * self.target_kl:
                    print(
                        f"KL early stopping triggered! KL: {approx_kl.item():.4f} > {1.5 * self.target_kl:.4f} at epoch {i}, start_idx {start_idx}")
                    break

                # MUST compute ratio outside torch.no_grad() for gradients to flow!
                logratio = logp - mb_logp_old
                ratio = torch.exp(logratio)

                surr1 = ratio * mb_advantages
                surr2 = torch.clamp(ratio, 1.0 - self.clip_threshold, 1.0 + self.clip_threshold) * mb_advantages
                policy_loss_unreduced = -torch.min(surr1, surr2)

                policy_loss = (policy_loss_unreduced * mb_mask).sum() / mb_mask.sum()
                entropy_loss = (entropy * mb_mask).sum() / mb_mask.sum()

                loss = policy_loss - (self.entropy_coef * entropy_loss)

                loss.backward()
                grad_norm = torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
                self.optimizer.step()

                avg_v_loss += value_loss.item()
                avg_loss += loss.item()
                avg_p_loss += policy_loss.item()
                avg_e_loss += entropy_loss.item()
                if "grad_norm" not in kwargs:
                    kwargs["grad_norm"] = 0.0
                kwargs["grad_norm"] += grad_norm.item()
                count += 1

        final_loss = avg_loss / count if count != 0 else avg_loss
        final_value_loss = avg_v_loss / count if count != 0 else avg_v_loss
        final_policy_loss = avg_p_loss / count if count != 0 else avg_p_loss
        final_entropy_loss = avg_e_loss / count if count != 0 else avg_e_loss

        with torch.no_grad():
            action_dist, bet_sizing_dist = self._unroll_policy(batched_states_dict, h_0, g_0, padded_new_hands)
            action_samples = action_dist.sample((1000,))
            action_hist = action_samples.float()

            if bet_sizing_dist is not None:
                betting_size = bet_sizing_dist.sample((1000,))
            else:
                betting_size = torch.zeros_like(action_hist)

            alpha_tensor = beta_tensor = None

            if self.mode == "normal" and bet_sizing_dist is not None:
                betting_size = torch.sigmoid(betting_size)
            elif self.mode == "beta" and bet_sizing_dist is not None:
                alpha_tensor = bet_sizing_dist.concentration1[0]
                beta_tensor = bet_sizing_dist.concentration0[0]

            trip_kl = torch.distributions.kl.kl_divergence(action_dist_old, action_dist)
            trip_kl = (trip_kl * mask).sum() / mask.sum()
            trip_kl = trip_kl.item()

        valid_rewards = padded_rewards[mask]

        return {"loss": final_loss, "value_loss": final_value_loss, "policy_loss": final_policy_loss,
                "entropy_loss": final_entropy_loss, "action_hist": action_hist, "alpha_hist": alpha_tensor,
                "beta_hist": beta_tensor, "betting_size": betting_size, "rewards": valid_rewards, "update_count": count,
                "grad_norm": kwargs.get("grad_norm", 0.0) / count if count > 0 else 0.0,
                "trip_kl": trip_kl}