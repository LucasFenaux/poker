"""
Implementation PPOClip
"""
from typing import Union
import numpy as np
import torch
from torch.distributions import Categorical, Normal
import copy
import pokerkit
from models import get_value_model


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
                "sgd_steps": 80,  # openai implementation
                "clip_threshold": 0.2,  # openai
                "target_kl": 0.01,  # openai
                "lr": 1e-3,
                "value_lr": 1e-3,
                }
    key = "ppo"
    def __init__(self, network, lr, device, value_lr, sgd_steps, clip_threshold, target_kl,
                 discrete: bool = False):
        super(PPO, self).__init__( lr, device)
        self.sgd_steps = sgd_steps
        self.clip_threshold = clip_threshold
        self.target_kl = target_kl
        self.value_lr = value_lr
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        self.value_network = get_value_model(device)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.value_lr)
        self.discrete = discrete

        network_size = 0
        for name, param in self.network.named_parameters():
            network_size += param.numel()
        print(f"Network size: {network_size}")

        network_size = 0
        for name, param in self.network.named_parameters():
            if param.requires_grad:
                network_size += param.numel()
        print(f"Trainable network size: {network_size}")

    def set_network(self, network):
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def get_network(self):
        return self.network

    def load_params(self, param_dict):
        self.network.load_state_dict(param_dict)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)

    def update(self, batch_states, batch_rewards, batch_actions, batch_rnn_states=None, *args, **kwargs):
        batch_rewards = torch.as_tensor(np.array(batch_rewards), device=self.device, dtype=torch.float32)
        # states = torch.as_tensor(
        #     np.array(batch_states),
        #     device=self.device,
        #     dtype=torch.float32,
        # )
        states = batch_states

        actions = torch.as_tensor(
            np.array(batch_actions),
            device=self.device,
            dtype=torch.long if self.discrete else torch.float32,
        )
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

        # batch_rewards = normalize_rewards(batch_rewards, self.env_name)

        # we then do multiple steps of SGD for the policy update
        for i in range(self.sgd_steps):
            # compute what we need
            self.value_optimizer.zero_grad()
            value_function = self.value_network(*states).squeeze()
            # value_function = normalize_rewards(value_function, self.env_name)
            # we normalize after here because we need to preserve the sign of the advantage so we cannot mean/std batch normalize it as that might shift it
            advantages = batch_rewards - value_function.clone().detach()

            # update the value network
            value_loss = torch.nn.functional.mse_loss(value_function, batch_rewards)
            value_loss.backward()
            self.value_optimizer.step()

            # we then update the policy network
            # get the policy of current and old
            self.optimizer.zero_grad()
            dist = self.get_model_policy(self.network, states, batch_rnn_states)
            logp = dist.log_prob(actions)
            if not self.discrete:
                logp = logp.sum(dim=-1)

            approx_kl = logp_old - logp  # openai early_stop check
            if approx_kl.mean().item() > 1.5 * self.target_kl:
                break
            ratio = torch.exp(logp - logp_old)
            # compute the sign-dependent advantage coefficient
            loss = -torch.min(ratio * advantages.unsqueeze(1),
                              torch.clamp(ratio, 1.0 - self.clip_threshold, 1.0 + self.clip_threshold) * advantages.unsqueeze(1)).mean()
            if not torch.isfinite(loss):
                print("WARNING: loss is not finite")

            loss.backward()
            self.optimizer.step()
            avg_v_loss += value_loss.item()
            avg_loss += loss.item()
            count += 1
        if count != 0:
            loss = avg_loss / count
            value_loss = avg_v_loss / count
        else:
            loss = avg_loss
            value_loss = avg_v_loss
        return {"loss": loss, "value_loss": value_loss}

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
    def __init__(self, network, discrete: bool = False):
        self.network = network
        self.discrete = discrete

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
