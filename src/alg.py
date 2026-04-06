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
                "base_batch_size": 5000,
                "clip_threshold": 0.2,  # openai
                "target_kl": 0.01,  # openai
                "lr": 1e-4,
                "value_lr": 5e-4,
                "reward_normalization_scaler": 1,
                # "entropy_coef": 1e-4,
                "entropy_coef": 0,
                "grad_clip_norm": 0.5,
                }
    key = "ppo"
    def __init__(self, lr, device, value_lr, sgd_steps, clip_threshold, target_kl, reward_normalization_scaler,
                 grad_clip_norm, entropy_coef, base_batch_size, mode, discrete: bool = False):
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
        # self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr)

        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.value_lr)
        # self.value_optimizer = torch.optim.SGD(self.value_network.parameters(), lr=self.value_lr)

        self.discrete = discrete
        self.reward_normalization_scaler = reward_normalization_scaler
        self.entropy_coef = entropy_coef
        self.base_batch_size = base_batch_size

    @staticmethod
    def init_networks(device, discrete, mode):
        network = load_dummy_model(device, discrete, mode)
        value_network = get_value_model(device)
        return network, value_network

    def set_network(self, network):
        self.network = network
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr)

    def get_network(self):
        return self.network

    def load_params(self, param_dicts):
        network_param_dict, value_param_dict = param_dicts
        self.network.load_state_dict(network_param_dict)
        self.optimizer = torch.optim.Adam(self.network.parameters(), lr=self.lr)
        # self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr)

        self.value_network.load_state_dict(value_param_dict)
        self.value_optimizer = torch.optim.Adam(self.value_network.parameters(), lr=self.value_lr)
        # self.value_optimizer = torch.optim.SGD(self.value_network.parameters(), lr=self.value_lr)

    def load_optimizer_params(self, optimizer_params):
        network_opt_params, value_opt_params = optimizer_params
        self.optimizer.load_state_dict(network_opt_params)
        self.value_optimizer.load_state_dict(value_opt_params)

    def get_params(self):
        return [self.network.state_dict(), self.value_network.state_dict()]

    def get_optimizer_params(self):
        return [self.optimizer.state_dict(), self.value_optimizer.state_dict()]

    def update(self, batch_states, batch_rewards, batch_actions, batch_rnn_states=None, sample_weights=None, *args,
               **kwargs):
        batch_size = len(batch_states)
        sgd_steps = max(self.sgd_steps, self.sgd_steps * batch_size / self.base_batch_size)
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
        safe_actions = torch.clamp(actions, min=1e-5, max=1.0 - 1e-5)

        # print("actions", actions.shape)
        if batch_rnn_states is not None:
            # we need to reshape the rnn states to be in the proper shape: (LxBxD)
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

        # batch_rewards = batch_rewards / self.reward_normalization_scaler  # we scale down the rewards
        # we scale the rewards logarithmically to avoid explosion and to minimize the advantage distortion from
        # big all-ins
        sign = torch.sign(batch_rewards)
        batch_rewards = sign * torch.log(batch_rewards.abs() + 1)

        # pre-compute what we need
        with torch.no_grad():
        # old_network = copy.deepcopy(self.network)
        # old_network.requires_grad_(False)  # get a copy of the current network
        #     dist_old = self.get_model_policy(self.network, states, batch_rnn_states)
        #     logp_old = dist_old.log_prob(safe_actions)
        #     if not self.discrete:
        #         logp_old = logp_old.sum(dim=-1)  # Our discrete models return both which action and the bet sizing
            dist_old = self.get_model_policy(self.network, states, batch_rnn_states)
            if not self.discrete:
                logp_all = dist_old.log_prob(safe_actions)
                # logp_all[:, 0] is the action choice, logp_all[:, 1] is the bet sizing

                # Create a mask: 1.0 if it was a RAISE (>= 2/3), 0.0 otherwise
                is_raise = (safe_actions[:, 0] >= Action.get_raise_threshold()).float()

                # Only add the bet sizing log_prob if the action was actually a RAISE
                logp_old = logp_all[:, 0] + (logp_all[:, 1] * is_raise)
            else:
                logp_old = dist_old.log_prob(safe_actions)


            value_function = self.value_network(*states).squeeze(-1)

            # we normalize after here because we need to preserve the sign of the advantage so we cannot mean/std batch normalize it as that might shift it
            advantages = batch_rewards - value_function.clone().detach()

            # mean/std normalize the advantages
            if sample_weights is None:
                advantages = (advantages - advantages.mean()) / (advantages.std() + 1e-8)
            else:
                # we do weighted normalization
                weighted_mean = (advantages * prob_sample_weights).sum()
                weighted_var = (prob_sample_weights * ((advantages - weighted_mean) ** 2)).sum()
                weighted_std = torch.sqrt(weighted_var + 1e-8)
                advantages = (advantages - weighted_mean) / (weighted_std + 1e-8)

        avg_loss = 0
        count = 0
        avg_v_loss = 0
        avg_p_loss = 0
        avg_e_loss = 0
        # batch_rewards = normalize_rewards(batch_rewards, self.env_name)

        # we then do multiple steps of SGD for the policy update
        for i in range(sgd_steps):
            # compute what we need
            self.value_optimizer.zero_grad()
            value_function = self.value_network(*states).squeeze(-1)
            # value_function = normalize_rewards(value_function, self.env_name)
            # we scale down the predicted reward
            value_function = value_function

            # update the value network
            if sample_weights is None:
                # value_loss = torch.nn.functional.mse_loss(value_function, batch_rewards)
                value_loss = torch.nn.functional.smooth_l1_loss(value_function, batch_rewards)
            else:
                # can switch back to MSE since we added the log scaling
                # value_loss = torch.nn.functional.mse_loss(value_function, batch_rewards, reduction="none")
                value_loss = torch.nn.functional.smooth_l1_loss(value_function, batch_rewards, reduction="none")
                assert value_loss.dim() == normalized_sample_weights.dim()
                value_loss = (value_loss * normalized_sample_weights).mean()

            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.grad_clip_norm)

            self.value_optimizer.step()

            # we then update the policy network
            # get the policy of current and old
            self.optimizer.zero_grad()
            dist = self.get_model_policy(self.network, states, batch_rnn_states)
            # logp = dist.log_prob(safe_actions)

            # add entropy regularization
            # entropy = dist.entropy()
            #
            # if not self.discrete:
            #     logp = logp.sum(dim=-1)
            #     entropy = entropy.sum(dim=-1)  # Sum entropy across action dimensions for continuous spaces


            # masked irrelevant bet sizing
            if not self.discrete:
                logp_all = dist.log_prob(safe_actions)
                is_raise = (safe_actions[:, 0] >= Action.get_raise_threshold()).float()
                logp = logp_all[:, 0] + (logp_all[:, 1] * is_raise)

                # You also need to mask the entropy!
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
            # compute the sign-dependent advantage coefficient
            policy_loss = -torch.min(ratio * advantages, torch.clamp(ratio, 1.0 - self.clip_threshold,
                                                                     1.0 + self.clip_threshold) * advantages)
            if sample_weights is None:
                policy_loss = policy_loss.mean()
            else:
                # we do a weighted mean
                policy_loss = (policy_loss * normalized_sample_weights).mean()

            if not torch.isfinite(policy_loss):
                print("WARNING: loss is not finite")

            # add the entropy reg component
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

        # compute a histogram
        with torch.no_grad():
            dist = self.get_model_policy(self.network, states, batch_rnn_states)
            # samples = dist.sample((1000,))[:, :, 0]
            samples = dist.sample((1000,))
            action_hist = samples[:, :, 0]
            betting_size = samples[:, :, 1]
            alpha_tensor = None
            beta_tensor = None
            if self.mode == "normal":
                action_hist = torch.sigmoid(action_hist)
                betting_size = torch.sigmoid(betting_size)
            elif self.mode == "beta":
                dist: torch.distributions.Beta
                alpha_tensor = dist.concentration1[0]
                beta_tensor = dist.concentration0[0]

        # Log the actual distribution shape to TensorBoard

        return {"loss": loss, "value_loss": value_loss, "policy_loss": policy_loss, "entropy_loss": entropy_loss,
                "action_hist": action_hist, "alpha_hist": alpha_tensor, "beta_hist": beta_tensor,
                "betting_size": betting_size, "rewards": batch_rewards}

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
