from typing import Union
import numpy as np
import torch
from pip._internal import network
from torch.distributions import Categorical, Normal
import pokerkit
from src.models import get_value_model, load_dummy_model
from src.action_interpreter import Action
from . import PPOInferenceWrapper
from .alg import OnPolicyAlgorithm
from .ppo import PPOInferenceWrapper


class NeuRD(OnPolicyAlgorithm):
    default_hyperparameters = {
        "mini_batch_size": 500,
        "lr": 1e-4,
        "value_lr": 5e-4,
        "grad_clip_norm": 0.5,
        "reward_normalization_scaler": 1,
    }
    def __init__(self, lr, device, value_lr, reward_normalization_scaler, grad_clip_norm, mini_batch_size):
        super(NeuRD, self).__init__(lr, device)
        self.mini_batch_size =mini_batch_size
        self.value_lr = value_lr
        self.grad_clip_norm = grad_clip_norm
        network, value_network = self.init_networks(device, True, "logits")
        self.network = network
        self.value_network = value_network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.value_lr)
        self.reward_normalization_scaler = reward_normalization_scaler
        self.mini_batch_size = mini_batch_size

    @staticmethod
    def init_networks(device, discrete, mode):
        network = load_dummy_model(device, discrete, mode, return_logits=True)  # need logits NeuRD policy loss
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

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = self.value_lr

    def get_params(self):
        return [self.network.state_dict(), self.value_network.state_dict()]

    def get_optimizer_params(self):
        return [self.optimizer.state_dict(), self.value_optimizer.state_dict()]

    def preprocess_batch(self, states_list, actors_list):
        """Converts raw Python states/actors into a batched dictionary of PyTorch tensors."""
        from src.game_registry import get_current_game_config
        StatePreprocessor = get_current_game_config()['state_preprocessor']
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

    def update(self, batch_states, batch_rewards, batch_actions, batch_rnn_states=None, sample_weights=None, *args,
               **kwargs):
        # we only do mini_batch updates
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
            actions = torch.stack(batch_actions).to(self.device).long()  # need long for indexing
        else:
            actions = torch.as_tensor(
                np.array(batch_actions),
                device=self.device,
                dtype=torch.long,
            )

        if sample_weights is not None:
            sample_weights = torch.tensor(sample_weights, device=self.device)
            assert sample_weights.dim() == 1
            normalized_sample_weights = sample_weights / sample_weights.mean()
            prob_sample_weights = sample_weights / sample_weights.sum()

        with torch.no_grad():
            value_function = self.value_network(*states).squeeze(-1)
            advantages = batch_rewards - value_function.clone().detach()
            if sample_weights is None:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                weighted_mean = (advantages * prob_sample_weights).sum()
                weighted_var = (prob_sample_weights * ((advantages - weighted_mean) ** 2)).sum()
                weighted_std = torch.sqrt(weighted_var + 1e-8)
                advantages = (advantages - weighted_mean) / (weighted_std + 1e-8)

        count = 0
        avg_v_loss = 0
        avg_p_loss = 0

        indices = torch.randperm(batch_size, device=self.device)
        for start_idx in range(0, batch_size, self.mini_batch_size):
            mini_batch_indices = indices[start_idx:start_idx + self.mini_batch_size]
            mini_batch_dict = {k: v[mini_batch_indices] for k, v in states[0].items()}
            mini_batch_states = (mini_batch_dict,)
            mini_batch_rewards = batch_rewards[mini_batch_indices]
            mini_batch_advantages = advantages[mini_batch_indices]
            mini_batch_actions = actions[mini_batch_indices]

            if sample_weights is not None:
                mini_batch_sample_weights = normalized_sample_weights[mini_batch_indices]

            self.value_optimizer.zero_grad()
            value_function = self.value_network(*mini_batch_states).squeeze(-1)
            if sample_weights is None:
                value_loss = torch.nn.functional.smooth_l1_loss(value_function, mini_batch_rewards)
            else:
                value_loss = torch.nn.functional.smooth_l1_loss(value_function, mini_batch_rewards,
                                                                reduction="none")
                value_loss = (value_loss * mini_batch_sample_weights).mean()
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.grad_clip_norm)
            self.value_optimizer.step()

            self.optimizer.zero_grad()

            decision_logits, bet_logits = self.get_model_logits(self.network, mini_batch_states)
            decision_actions = mini_batch_actions[..., 0].unsqueeze(-1)
            bet_actions = mini_batch_actions[..., 1].unsqueeze(-1)
            # we need to gather the logits by action since the neuRD loss is y(a, \theta)*advantage
            decision_logits = torch.gather(decision_logits, -1, decision_actions)
            bet_logits = torch.gather(bet_logits, -1, bet_actions)

            logits = torch.cat((decision_logits, bet_logits), dim=-1)  # concatenate along the logit dimension, not the batch dim
            policy_loss = -logits * mini_batch_advantages.unsqueeze(-1)  # - for gradient ascent

            if sample_weights is None:
                policy_loss = policy_loss.mean()
            else:
                policy_loss = (policy_loss * mini_batch_sample_weights).mean()

            if not torch.isfinite(policy_loss):
                print("WARNING: loss is not finite")
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_clip_norm)

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
            action_logits, bet_logits = self.get_model_logits(self.network, states)

        return {"loss": policy_loss, "value_loss": value_loss, "policy_loss": policy_loss,
                "action_hist": action_logits, "betting_size": bet_logits, "rewards": batch_rewards,
                "update_count": count,}


    def get_model_logits(self, network, state, hand_hidden: torch.Tensor = None,
                   game_hidden: torch.Tensor = None) -> torch.Tensor:
        # Handle live play tuples that need preprocessing
        if isinstance(state, tuple) and len(state) == 2 and not isinstance(state[0], dict):
            s, a = state
            batched_dict = self.preprocess_batch([s], [a])
            state_args = (batched_dict,)
        else:
            # Handle already batched dictionary states
            state_args = state

        logits = network(*state_args)
        return logits


class NeuRDInferenceWrapper(PPOInferenceWrapper):

    def get_model_policy(self, network, state, hand_hidden: torch.Tensor = None,
                   game_hidden: torch.Tensor = None):
        if isinstance(state, tuple) and len(state) == 2 and not isinstance(state[0], dict):
            s, a = state
            batched_dict = self.preprocess_batch([s], [a])
            state_args = (batched_dict,)
        else:
            # Handle already batched dictionary states
            state_args = state

        decision_logits, bet_logits = network(*state_args)
        return Categorical(logits=decision_logits), Categorical(logits=bet_logits)