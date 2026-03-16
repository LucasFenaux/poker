"""
Implementation PPOClip
"""
from typing import Union
import numpy as np
import torch
from torch.distributions import Categorical, Normal
import copy
import pokerkit
from models import get_value_model, load_dummy_model


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
    default_hyperparameters = {
                # "sgd_steps": 80,  # openai implementation
                "sgd_steps": 5,
                "clip_threshold": 0.2,  # openai
                "target_kl": 0.01,  # openai
                "lr": 1e-4,
                "value_lr": 5e-4,
                "reward_normalization_scaler": 100,
                "entropy_coef": 0.05
                }
    key = "ppo"
    def __init__(self, lr, device, value_lr, sgd_steps, clip_threshold, target_kl, reward_normalization_scaler,
                 entropy_coef, discrete: bool = False):
        super(PPO, self).__init__( lr, device)
        self.sgd_steps = sgd_steps
        self.clip_threshold = clip_threshold
        self.grad_norm_clip_threshold = 0.5
        self.target_kl = target_kl
        self.value_lr = value_lr
        network, value_network = self.init_networks(device, discrete)
        self.network = network
        self.value_network = value_network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.value_lr)
        self.discrete = discrete
        self.reward_normalization_scaler = reward_normalization_scaler
        self.entropy_coef = entropy_coef

    @staticmethod
    def init_networks(device, discrete):
        network = load_dummy_model(device, discrete)
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

    def get_params(self):
        return [self.network.state_dict(), self.value_network.state_dict()]

    def update(self, batch_states, batch_rewards, batch_actions, batch_rnn_states=None, *args, **kwargs):
        if isinstance(batch_rewards[0], torch.Tensor):
            batch_rewards = torch.stack(batch_rewards).to(self.device).to(torch.float32)
        else:
            # 1. Cast every element to a standard Python float to strip away Fractions
            clean_rewards = [float(r) for r in batch_rewards]

            # 2. Now NumPy will happily create a float32 array instead of an object_ array
            batch_rewards_np = np.array(clean_rewards, dtype=np.float32)
            # print("rewards", batch_rewards_np.shape)

            # 3. Convert to PyTorch tensor
            batch_rewards = torch.as_tensor(batch_rewards_np, device=self.device)
        # print("rew", batch_rewards.shape)
        states = batch_states

        if isinstance(batch_actions[0], torch.Tensor):
            actions = torch.stack(batch_actions).to(self.device).to(torch.long if self.discrete else torch.float32)
        else:
            actions = torch.as_tensor(
                np.array(batch_actions),
                device=self.device,
                dtype=torch.long if self.discrete else torch.float32,
            )
        # print("actions", actions.shape)
        if batch_rnn_states is not None:
            # we need to reshape the rnn states to be in the proper shape: (LxBxD)
            new_hs = []
            new_cs = []
            for (h, c) in batch_rnn_states:
                new_hs.append(h)
                new_cs.append(c)
            batch_rnn_states = (torch.cat(new_hs, dim=1), torch.cat(new_cs, dim=1))

        old_network = copy.deepcopy(self.network)
        old_network.requires_grad_(False)  # get a copy of the current network
        dist_old = self.get_model_policy(old_network, states, batch_rnn_states)
        logp_old = dist_old.log_prob(actions)
        if not self.discrete:
            logp_old = logp_old.sum(dim=-1)  # Our discrete models return both which action and the bet sizing

        avg_loss = 0
        count = 0
        avg_v_loss = 0
        avg_p_loss = 0
        avg_e_loss = 0

        batch_rewards = batch_rewards / self.reward_normalization_scaler  # we scale down the rewards
        # batch_rewards = normalize_rewards(batch_rewards, self.env_name)

        # we then do multiple steps of SGD for the policy update
        for i in range(self.sgd_steps):
            # compute what we need
            self.value_optimizer.zero_grad()
            value_function = self.value_network(*states).squeeze()
            # value_function = normalize_rewards(value_function, self.env_name)
            # we scale down the predicted reward
            value_function = value_function

            # we normalize after here because we need to preserve the sign of the advantage so we cannot mean/std batch normalize it as that might shift it
            advantages = batch_rewards - value_function.clone().detach()

            # mean/std normalize the advantages
            advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)

            # update the value network
            value_loss = torch.nn.functional.mse_loss(value_function, batch_rewards)

            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.grad_norm_clip_threshold)

            value_loss.backward()
            self.value_optimizer.step()

            # we then update the policy network
            # get the policy of current and old
            self.optimizer.zero_grad()
            dist = self.get_model_policy(self.network, states, batch_rnn_states)
            logp = dist.log_prob(actions)

            # add entropy regularization
            entropy = dist.entropy()

            if not self.discrete:
                logp = logp.sum(dim=-1)
                entropy = entropy.sum(dim=-1)  # Sum entropy across action dimensions for continuous spaces

            approx_kl = logp_old - logp  # openai early_stop check
            if approx_kl.mean().item() > 1.5 * self.target_kl:
                break
            ratio = torch.exp(logp - logp_old)
            # compute the sign-dependent advantage coefficient
            policy_loss = -torch.min(ratio * advantages.unsqueeze(1),
                              torch.clamp(ratio, 1.0 - self.clip_threshold, 1.0 + self.clip_threshold) * advantages.unsqueeze(1)).mean()
            if not torch.isfinite(policy_loss):
                print("WARNING: loss is not finite")

            # add the entropy reg component
            entropy_loss = entropy.mean()

            loss = policy_loss - (self.entropy_coef * entropy_loss)

            torch.nn.utils.clip_grad_norm_(self.network.parameters(), self.grad_norm_clip_threshold)

            loss.backward()
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
        return {"loss": loss, "value_loss": value_loss, "policy_loss": policy_loss, "entropy_loss": entropy_loss}

    def get_action(self, state: (pokerkit.State, int), rnn_state = None):
        policy = self.get_model_policy(self.network, state, rnn_state=rnn_state)
        return policy.sample().cpu()

    def get_model_policy(self, network, state: (pokerkit.State, int), rnn_state = None) -> Union[Categorical, Normal]:
        if self.discrete:
            logits = network(*state)
            dist: Categorical = Categorical(logits)
            return dist
        else:
            dist: Normal = network(*state)  # already returns a Normal distribution when not deterministic
            return dist


class PPOInferenceWrapper:
    def __init__(self, models, discrete: bool = False):
        self.network = models[0]
        self.discrete = discrete

    def load_params(self, param_dicts):
        network_param_dict, _ = param_dicts
        self.network.load_state_dict(network_param_dict)

    def load_network_params(self, params):
        self.network.load_state_dict(params)

    def to(self, device):
        self.network = self.network.to(device)
        return self

    def get_action(self, state: (pokerkit.State, int), rnn_state = None):
        policy = self.get_model_policy(self.network, state, rnn_state=rnn_state)
        return policy.sample().cpu()

    def get_model_policy(self, network, state: (pokerkit.State, int), rnn_state = None) -> Union[Categorical, Normal]:
        if self.discrete:
            logits = network(*state)
            dist: Categorical = Categorical(logits)
            return dist
        else:
            dist: Normal = network(*state)  # already returns a Normal distribution when not deterministic
            return dist
