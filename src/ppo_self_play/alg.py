"""
Implementation PPOClip
"""
from typing import Union
import numpy as np
import torch
from torch.distributions import Categorical, Normal
import pokerkit
from src.models import get_value_model, load_dummy_model
from src.action_interpreter import Action
from src.state_interpreter import StatePreprocessor


class BaseAlgorithm:
    default_hyperparameters = {}
    key = None
    def __init__(self, lr, device):
        self.lr = lr
        self.device = device
        self.network = None

    def update(self, *args, **kwargs):
        raise NotImplementedError

    def get_action(self, *args, **kwargs):
        raise NotImplementedError

    def get_network(self):
        """
        Return the network/networks used in get_action, needs to have .state_dict implemented
        :return: nn.Module
        """
        return self.network

    def set_network(self, network):
        self.network = network


class OnPolicyAlgorithm(BaseAlgorithm):
    def __init__(self, lr, device):
        super().__init__(lr, device)

    def update(self, batch_states, batch_rewards, batch_actions, batch_rnn_states = None, *args, **kwargs):
        raise NotImplementedError

    def get_action(self, *args, **kwargs):
        raise NotImplementedError


class PPO(OnPolicyAlgorithm):
    available_update_modes = ["full_batch", "mini_batch"]

    default_hyperparameters = {
                "sgd_steps": 5,
                "base_batch_size": 5000,
                "mini_batch_size": 5000,
                "clip_threshold": 0.2,
                "target_kl": 0.01,
                "lr": 1e-4,
                "value_lr": 5e-4,
                "reward_normalization_scaler": 1,
                "entropy_coef": 5e-3,
                "grad_clip_norm": 0.5,
                "update_mode": "mini_batch"
                }
    key = "ppo"
    def __init__(self, lr, device, value_lr, sgd_steps, clip_threshold, target_kl, reward_normalization_scaler,
                 grad_clip_norm, entropy_coef, base_batch_size, mini_batch_size, update_mode, mode, discrete: bool = False):
        super(PPO, self).__init__( lr, device)
        self.sgd_steps = sgd_steps
        self.clip_threshold = clip_threshold
        self.grad_clip_norm = grad_clip_norm
        self.target_kl = target_kl
        self.value_lr = value_lr
        network, value_network = self.init_networks(device, discrete, mode)
        self.mode = mode
        self.network = network
        self.value_network = value_network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.value_lr)

        self.discrete = discrete
        self.reward_normalization_scaler = reward_normalization_scaler
        self.entropy_coef = entropy_coef
        self.base_batch_size = base_batch_size
        self.mini_batch_size = mini_batch_size
        self.update_mode = update_mode if update_mode in self.available_update_modes else "mini_batch"
        if update_mode not in self.available_update_modes:
            print(f"update_mode {update_mode} not implemented - Falling back to {self.update_mode}")

    @staticmethod
    def init_networks(device, discrete, mode):
        network = load_dummy_model(device, discrete, mode)
        value_network = get_value_model(device)
        return network, value_network

    def set_network(self, network):
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def get_network(self):
        return self.network

    def load_params(self, param_dicts):
        network_param_dict, value_param_dict = param_dicts
        self.network.load_state_dict(network_param_dict)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

        self.value_network.load_state_dict(value_param_dict)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.value_lr)

    def load_optimizer_params(self, optimizer_params):
        network_opt_params, value_opt_params = optimizer_params
        self.optimizer.load_state_dict(network_opt_params)
        self.value_optimizer.load_state_dict(value_opt_params)

    def get_params(self):
        return [self.network.state_dict(), self.value_network.state_dict()]

    def get_optimizer_params(self):
        return [self.optimizer.state_dict(), self.value_optimizer.state_dict()]

    def preprocess_batch(self, states_list, actors_list):
        """Converts raw Python states/actors into a batched dictionary of PyTorch tensors."""
        preprocessor = StatePreprocessor()
        batch_dict = {}

        # Process every state
        for s, a in zip(states_list, actors_list):
            processed = preprocessor.process(s, a)
            for k, v in processed.items():
                if k not in batch_dict:
                    batch_dict[k] = []
                batch_dict[k].append(v)

        tensor_dict = {}
        for k, v in batch_dict.items():
            if k in ["num_players", "rel_to_button", "player_ranks", "player_suits", "board_ranks", "board_suits"]:
                tensor_dict[k] = torch.tensor(v, dtype=torch.long, device=self.device)
            else:
                tensor_dict[k] = torch.tensor(v, dtype=torch.float32, device=self.device)
        return tensor_dict

    def mini_batch_update(self, batch_states, batch_rewards, batch_actions, batch_rnn_states=None, sample_weights=None, *args,
               **kwargs):
        batch_size = len(batch_rewards)
        if isinstance(batch_rewards[0], torch.Tensor):
            batch_rewards = torch.stack(batch_rewards).to(self.device).to(torch.float32)
        else:
            clean_rewards = [float(r) for r in batch_rewards]
            batch_rewards_np = np.array(clean_rewards, dtype=np.float32)
            batch_rewards = torch.as_tensor(batch_rewards_np, device=self.device)

        # 1. Preprocess the states outside the SGD loop
        states_list, current_actors_list = batch_states
        batched_states_dict = self.preprocess_batch(states_list, current_actors_list)
        states = (batched_states_dict,)

        if isinstance(batch_actions[0], torch.Tensor):
            actions = torch.stack(batch_actions).to(self.device).to(torch.long if self.discrete else torch.float32)
        else:
            actions = torch.as_tensor(
                np.array(batch_actions),
                device=self.device,
                dtype=torch.long if self.discrete else torch.float32,
            )
        safe_actions = torch.clamp(actions, min=1e-5, max=1.0 - 1e-5)

        if batch_rnn_states is not None:
            new_hs = []
            new_cs = []
            for (h, c) in batch_rnn_states:
                new_hs.append(h)
                new_cs.append(c)
            batch_rnn_states = (torch.cat(new_hs, dim=1), torch.cat(new_cs, dim=1))

        if sample_weights is not None:
            sample_weights = torch.tensor(sample_weights, device=self.device)
            assert sample_weights.dim() == 1
            normalized_sample_weights = sample_weights / sample_weights.mean()
            prob_sample_weights = sample_weights / sample_weights.sum()

        sign = torch.sign(batch_rewards)
        batch_rewards = sign * torch.log(batch_rewards.abs() + 1)

        with torch.no_grad():
            dist_old = self.get_model_policy(self.network, states, batch_rnn_states)
            if not self.discrete:
                logp_all = dist_old.log_prob(safe_actions)
                is_raise = (safe_actions[:, 0] >= Action.get_raise_threshold()).float()
                logp_old = logp_all[:, 0] + (logp_all[:, 1] * is_raise)
            else:
                logp_old = dist_old.log_prob(safe_actions)

            value_function = self.value_network(*states).squeeze(-1)
            advantages = batch_rewards - value_function.clone().detach()

            if sample_weights is None:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                weighted_mean = (advantages * prob_sample_weights).sum()
                weighted_var = (prob_sample_weights * ((advantages - weighted_mean) ** 2)).sum()
                weighted_std = torch.sqrt(weighted_var + 1e-8)
                advantages = (advantages - weighted_mean) / (weighted_std + 1e-8)

        avg_loss = 0
        count = 0
        avg_v_loss = 0
        avg_p_loss = 0
        avg_e_loss = 0

        for i in range(self.sgd_steps):
            indices = torch.randperm(batch_size, device=self.device)
            for start_idx in range(0, batch_size, self.mini_batch_size):
                mini_batch_indices = indices[start_idx:start_idx + self.mini_batch_size]

                # Slice the dictionary tensors for the mini batch
                mini_batch_dict = {k: v[mini_batch_indices] for k, v in states[0].items()}
                mini_batch_states = (mini_batch_dict,)

                mini_batch_rewards = batch_rewards[mini_batch_indices]
                mini_batch_advantages = advantages[mini_batch_indices]
                mini_batch_safe_actions = safe_actions[mini_batch_indices]
                mini_batch_logp_old = logp_old[mini_batch_indices]
                if batch_rnn_states is not None:
                    mini_batch_rnn_states = batch_rnn_states[mini_batch_indices]
                else:
                    mini_batch_rnn_states = None

                if sample_weights is not None:
                    mini_batch_sample_weights = normalized_sample_weights[mini_batch_indices]

                self.value_optimizer.zero_grad()
                value_function = self.value_network(*mini_batch_states).squeeze(-1)

                if sample_weights is None:
                    value_loss = torch.nn.functional.smooth_l1_loss(value_function, mini_batch_rewards)
                else:
                    value_loss = torch.nn.functional.smooth_l1_loss(value_function, mini_batch_rewards, reduction="none")
                    value_loss = (value_loss * mini_batch_sample_weights).mean()

                value_loss.backward()
                torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.grad_clip_norm)
                self.value_optimizer.step()

                self.optimizer.zero_grad()
                dist = self.get_model_policy(self.network, mini_batch_states, mini_batch_rnn_states)

                if not self.discrete:
                    logp_all = dist.log_prob(mini_batch_safe_actions)
                    is_raise = (mini_batch_safe_actions[:, 0] >= Action.get_raise_threshold()).float()
                    logp = logp_all[:, 0] + (logp_all[:, 1] * is_raise)

                    entropy_all = dist.entropy()
                    entropy = entropy_all[:, 0] + (entropy_all[:, 1] * is_raise)
                else:
                    logp = dist.log_prob(mini_batch_safe_actions)
                    entropy = dist.entropy()

                with torch.no_grad():
                    logratio = logp - mini_batch_logp_old
                    ratio = torch.exp(logratio)
                    approx_kl = ((ratio - 1) - logratio).mean()

                if approx_kl.item() > 1.5 * self.target_kl:
                    break

                ratio = torch.exp(logp - mini_batch_logp_old)
                policy_loss = -torch.min(ratio * mini_batch_advantages, torch.clamp(ratio, 1.0 - self.clip_threshold,
                                                                         1.0 + self.clip_threshold) * mini_batch_advantages)
                if sample_weights is None:
                    policy_loss = policy_loss.mean()
                else:
                    policy_loss = (policy_loss * mini_batch_sample_weights).mean()

                if not torch.isfinite(policy_loss):
                    print("WARNING: loss is not finite")

                entropy_loss = entropy.mean()
                loss = policy_loss - (self.entropy_coef * entropy_loss)

                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)

                self.optimizer.step()
                avg_v_loss += value_loss.item()
                avg_loss += loss.item()
                avg_p_loss += policy_loss.item()
                avg_e_loss += entropy_loss.item()
                count += 1

        if count != 0:
            loss = avg_loss / count
            value_loss = avg_v_loss / count
            policy_loss = avg_p_loss / count
            entropy_loss = avg_e_loss / count
        else:
            loss = avg_loss
            value_loss = avg_v_loss
            policy_loss = avg_p_loss
            entropy_loss = avg_e_loss

        with torch.no_grad():
            dist = self.get_model_policy(self.network, states, batch_rnn_states)
            samples = dist.sample((1000,))
            action_hist = samples[:, :, 0]
            betting_size = samples[:, :, 1]
            alpha_tensor = None
            beta_tensor = None
            if self.mode == "normal":
                action_hist = torch.sigmoid(action_hist)
                betting_size = torch.sigmoid(betting_size)
            elif self.mode == "beta":
                alpha_tensor = dist.concentration1[0]
                beta_tensor = dist.concentration0[0]

        return {"loss": loss, "value_loss": value_loss, "policy_loss": policy_loss, "entropy_loss": entropy_loss,
                "action_hist": action_hist, "alpha_hist": alpha_tensor, "beta_hist": beta_tensor,
                "betting_size": betting_size, "rewards": batch_rewards}

    def full_batch_update(self, batch_states, batch_rewards, batch_actions, batch_rnn_states=None, sample_weights=None, *args,
               **kwargs):
        batch_size = len(batch_rewards)
        sgd_steps = max(self.sgd_steps, self.sgd_steps * batch_size // self.base_batch_size)

        if isinstance(batch_rewards[0], torch.Tensor):
            batch_rewards = torch.stack(batch_rewards).to(self.device).to(torch.float32)
        else:
            clean_rewards = [float(r) for r in batch_rewards]
            batch_rewards_np = np.array(clean_rewards, dtype=np.float32)
            batch_rewards = torch.as_tensor(batch_rewards_np, device=self.device)

        # 1. Preprocess the states outside the SGD loop
        states_list, current_actors_list = batch_states
        batched_states_dict = self.preprocess_batch(states_list, current_actors_list)
        states = (batched_states_dict,)

        if isinstance(batch_actions[0], torch.Tensor):
            actions = torch.stack(batch_actions).to(self.device).to(torch.long if self.discrete else torch.float32)
        else:
            actions = torch.as_tensor(
                np.array(batch_actions),
                device=self.device,
                dtype=torch.long if self.discrete else torch.float32,
            )
        safe_actions = torch.clamp(actions, min=1e-5, max=1.0 - 1e-5)

        if batch_rnn_states is not None:
            new_hs = []
            new_cs = []
            for (h, c) in batch_rnn_states:
                new_hs.append(h)
                new_cs.append(c)
            batch_rnn_states = (torch.cat(new_hs, dim=1), torch.cat(new_cs, dim=1))

        if sample_weights is not None:
            sample_weights = torch.tensor(sample_weights, device=self.device)
            assert sample_weights.dim() == 1
            normalized_sample_weights = sample_weights / sample_weights.mean()
            prob_sample_weights = sample_weights / sample_weights.sum()

        sign = torch.sign(batch_rewards)
        batch_rewards = sign * torch.log(batch_rewards.abs() + 1)

        with torch.no_grad():
            dist_old = self.get_model_policy(self.network, states, batch_rnn_states)
            if not self.discrete:
                logp_all = dist_old.log_prob(safe_actions)
                is_raise = (safe_actions[:, 0] >= Action.get_raise_threshold()).float()
                logp_old = logp_all[:, 0] + (logp_all[:, 1] * is_raise)
            else:
                logp_old = dist_old.log_prob(safe_actions)

            value_function = self.value_network(*states).squeeze(-1)
            advantages = batch_rewards - value_function.clone().detach()

            if sample_weights is None:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                weighted_mean = (advantages * prob_sample_weights).sum()
                weighted_var = (prob_sample_weights * ((advantages - weighted_mean) ** 2)).sum()
                weighted_std = torch.sqrt(weighted_var + 1e-8)
                advantages = (advantages - weighted_mean) / (weighted_std + 1e-8)

        avg_loss = 0
        count = 0
        avg_v_loss = 0
        avg_p_loss = 0
        avg_e_loss = 0

        for i in range(sgd_steps):
            self.value_optimizer.zero_grad()
            value_function = self.value_network(*states).squeeze(-1)

            if sample_weights is None:
                value_loss = torch.nn.functional.smooth_l1_loss(value_function, batch_rewards)
            else:
                value_loss = torch.nn.functional.smooth_l1_loss(value_function, batch_rewards, reduction="none")
                value_loss = (value_loss * normalized_sample_weights).mean()

            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.grad_clip_norm)
            self.value_optimizer.step()

            self.optimizer.zero_grad()
            dist = self.get_model_policy(self.network, states, batch_rnn_states)

            if not self.discrete:
                logp_all = dist.log_prob(safe_actions)
                is_raise = (safe_actions[:, 0] >= Action.get_raise_threshold()).float()
                logp = logp_all[:, 0] + (logp_all[:, 1] * is_raise)

                entropy_all = dist.entropy()
                entropy = entropy_all[:, 0] + (entropy_all[:, 1] * is_raise)
            else:
                logp = dist.log_prob(safe_actions)
                entropy = dist.entropy()

            with torch.no_grad():
                logratio = logp - logp_old
                ratio = torch.exp(logratio)
                approx_kl = ((ratio - 1) - logratio).mean()

            if approx_kl.item() > 1.5 * self.target_kl:
                break
            ratio = torch.exp(logp - logp_old)
            policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1.0 - self.clip_threshold,
                                                                     1.0 + self.clip_threshold) * advantages)
            if sample_weights is None:
                policy_loss = policy_loss.mean()
            else:
                policy_loss = (policy_loss * normalized_sample_weights).mean()

            if not torch.isfinite(policy_loss):
                print("WARNING: loss is not finite")

            entropy_loss = entropy.mean()
            loss = policy_loss - (self.entropy_coef * entropy_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)
            self.optimizer.step()

            avg_v_loss += value_loss.item()
            avg_loss += loss.item()
            avg_p_loss += policy_loss.item()
            avg_e_loss += entropy_loss.item()
            count += 1

        if count != 0:
            loss = avg_loss / count
            value_loss = avg_v_loss / count
            policy_loss = avg_p_loss / count
            entropy_loss = avg_e_loss / count
        else:
            loss = avg_loss
            value_loss = avg_v_loss
            policy_loss = avg_p_loss
            entropy_loss = avg_e_loss

        with torch.no_grad():
            dist = self.get_model_policy(self.network, states, batch_rnn_states)
            samples = dist.sample((1000,))
            action_hist = samples[:, :, 0]
            betting_size = samples[:, :, 1]
            alpha_tensor = None
            beta_tensor = None
            if self.mode == "normal":
                action_hist = torch.sigmoid(action_hist)
                betting_size = torch.sigmoid(betting_size)
            elif self.mode == "beta":
                alpha_tensor = dist.concentration1[0]
                beta_tensor = dist.concentration0[0]

        return {"loss": loss, "value_loss": value_loss, "policy_loss": policy_loss, "entropy_loss": entropy_loss,
                "action_hist": action_hist, "alpha_hist": alpha_tensor, "beta_hist": beta_tensor,
                "betting_size": betting_size, "rewards": batch_rewards}

    def update(self, batch_states, batch_rewards, batch_actions, batch_rnn_states=None, sample_weights=None, *args,
               **kwargs):
        if self.update_mode == "full_batch":
            return self.full_batch_update(batch_states, batch_rewards, batch_actions, batch_rnn_states, sample_weights,
                                          *args, **kwargs)
        elif self.update_mode == "mini_batch":
            return self.mini_batch_update(batch_states, batch_rewards, batch_actions, batch_rnn_states, sample_weights,
                                          *args, **kwargs)
        else:
            raise ValueError(f"Unknown update mode: {self.update_mode}")

    def get_action(self, state: (pokerkit.State, int), rnn_state = None):
        policy = self.get_model_policy(self.network, state, rnn_state=rnn_state)
        return policy.sample().cpu().squeeze(0)

    def get_model_policy(self, network, state, rnn_state = None) -> Union[Categorical, Normal]:
        # Handle live play tuples that need preprocessing
        if isinstance(state, tuple) and len(state) == 2 and not isinstance(state[0], dict):
            s, a = state
            batched_dict = self.preprocess_batch([s], [a])
            state_args = (batched_dict,)
        else:
            # Handle already batched dictionary states
            state_args = state

        if self.discrete:
            logits = network(*state_args)
            dist: Categorical = Categorical(logits)
            return dist
        else:
            dist: Normal = network(*state_args)
            return dist


class PPOInferenceWrapper:
    def __init__(self, models, discrete: bool = False):
        self.network = models[0]
        self.discrete = discrete
        self.device = next(self.network.parameters()).device

    def load_params(self, param_dicts):
        network_param_dict, _ = param_dicts
        self.network.load_state_dict(network_param_dict)

    def load_network_params(self, params):
        self.network.load_state_dict(params)

    def to(self, device):
        self.network = self.network.to(device)
        self.device = device
        return self

    def preprocess_batch(self, states_list, actors_list):
        preprocessor = StatePreprocessor()
        batch_dict = {}
        for s, a in zip(states_list, actors_list):
            processed = preprocessor.process(s, a)
            for k, v in processed.items():
                if k not in batch_dict:
                    batch_dict[k] = []
                batch_dict[k].append(v)

        tensor_dict = {}
        for k, v in batch_dict.items():
            if k in ["num_players", "rel_to_button", "player_ranks", "player_suits", "board_ranks", "board_suits"]:
                tensor_dict[k] = torch.tensor(v, dtype=torch.long, device=self.device)
            else:
                tensor_dict[k] = torch.tensor(v, dtype=torch.float32, device=self.device)
        return tensor_dict

    def get_action(self, state: (pokerkit.State, int), rnn_state = None):
        policy = self.get_model_policy(self.network, state, rnn_state=rnn_state)
        return policy.sample().cpu().squeeze(0)

    def get_model_policy(self, network, state, rnn_state = None) -> Union[Categorical, Normal]:
        if isinstance(state, tuple) and len(state) == 2 and not isinstance(state[0], dict):
            s, a = state
            batched_dict = self.preprocess_batch([s], [a])
            state_args = (batched_dict,)
        else:
            state_args = state

        if self.discrete:
            logits = network(*state_args)
            dist: Categorical = Categorical(logits)
            return dist
        else:
            dist: Normal = network(*state_args)
            return dist