import numpy as np
import torch
import copy
from torch.distributions import Categorical
from src.utils.models import load_dummy_model, get_q_model
from .alg import OnPolicyAlgorithm
from .ppo import PPOInferenceWrapper


class NeuRD(OnPolicyAlgorithm):
    default_hyperparameters = {
        "mini_batch_size": 500,
        "lr": 2e-3,
        "value_lr": 1e-2,
        "grad_clip_norm": 0.5,
        "reward_normalization_scaler": 1,
        "entropy_coef": 0.01,
        "beta": 2.0,  # openspiel uses this as default
        "critic_update_ratio": 4, # NeuRD paper uses 4 updates to Q for every 1 to Policy
        "target_network_update_freq": 100,
        "target_network_reg_weight": 0.01,
        "logit_penalty_weight": 100.0,
    }
    def __init__(self, lr, device, value_lr, reward_normalization_scaler, grad_clip_norm, mini_batch_size, entropy_coef,
                 beta, critic_update_ratio, target_network_update_freq, target_network_reg_weight, logit_penalty_weight, mode="categorical",
                 discrete=True, **kwargs):
        super(NeuRD, self).__init__(lr, device)
        self.mini_batch_size = mini_batch_size
        self.value_lr = value_lr
        self.grad_clip_norm = grad_clip_norm
        self.entropy_coef = entropy_coef
        self.beta = beta
        self.critic_update_ratio = critic_update_ratio
        self.target_network_update_freq = target_network_update_freq
        self.target_network_reg_weight = target_network_reg_weight
        self.logit_penalty_weight = logit_penalty_weight
        self.total_update_count = 0
        self.mode = mode
        self.discrete = discrete
        assert self.mode == "categorical"  # only mode supported for neurd
        assert self.discrete   # does not support continuous actions
        network, value_network, target_value_network = self.init_networks(device, discrete, mode)
        self.network = network
        self.value_network = value_network
        self.target_value_network = target_value_network
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr)
        self.value_optimizer = torch.optim.SGD(self.value_network.parameters(), lr=self.value_lr)
        self.reward_normalization_scaler = reward_normalization_scaler
        self.mini_batch_size = mini_batch_size

    @staticmethod
    def init_networks(device, discrete, mode):
        network = load_dummy_model(device, discrete, mode, return_logits=True)  # need logits NeuRD policy loss
        value_network = get_q_model(device, discrete, mode)
        target_value_network = get_q_model(device, discrete, mode)
        return network, value_network, target_value_network

    def set_network(self, network):
        self.network = network
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr)

    def get_network(self):
        return self.network

    def load_params(self, param_dicts):
        network_param_dict, value_param_dict, target_network_dict = param_dicts
        self.network.load_state_dict(network_param_dict)
        self.optimizer = torch.optim.SGD(self.network.parameters(), lr=self.lr)

        self.value_network.load_state_dict(value_param_dict)
        self.value_optimizer = torch.optim.SGD(self.value_network.parameters(), lr=self.value_lr)

        self.target_value_network.load_state_dict(value_param_dict)

    def load_optimizer_params(self, optimizer_params):
        network_opt_params, value_opt_params = optimizer_params
        self.optimizer.load_state_dict(network_opt_params)
        self.value_optimizer.load_state_dict(value_opt_params)

        for param_group in self.optimizer.param_groups:
            param_group['lr'] = self.lr
        for param_group in self.value_optimizer.param_groups:
            param_group['lr'] = self.value_lr

    def get_params(self):
        return [self.network.state_dict(), self.value_network.state_dict(), self.target_value_network.state_dict()]

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

    def target_value_network_update(self):
        self.target_value_network.load_state_dict(self.value_network.state_dict())

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

        count = 0
        avg_v_loss = 0
        avg_p_loss = 0

        indices = torch.randperm(batch_size, device=self.device)
        for start_idx in range(0, batch_size, self.mini_batch_size):
            mini_batch_indices = indices[start_idx:start_idx + self.mini_batch_size]
            mini_batch_dict = {k: v[mini_batch_indices] for k, v in states[0].items()}
            mini_batch_states = (mini_batch_dict,)
            mini_batch_rewards = batch_rewards[mini_batch_indices]
            mini_batch_actions = actions[mini_batch_indices]

            if sample_weights is not None:
                mini_batch_sample_weights = normalized_sample_weights[mini_batch_indices]

            self.value_optimizer.zero_grad()
            q_dec, q_bet = self.value_network(*mini_batch_states)

            with torch.no_grad():
                target_q_dec, target_q_bet = self.target_value_network(*mini_batch_states)

            # 1. Main loss (only on taken actions)
            action_dec_indices = mini_batch_actions[..., 0] if q_bet is not None else mini_batch_actions
            q_dec_taken = q_dec.gather(dim=-1, index=action_dec_indices.unsqueeze(-1)).squeeze(-1)
            
            if sample_weights is None:
                main_loss_dec = torch.nn.functional.smooth_l1_loss(q_dec_taken, mini_batch_rewards)
                reg_loss_dec = torch.nn.functional.smooth_l1_loss(q_dec, target_q_dec)
            else:
                main_loss_dec = torch.nn.functional.smooth_l1_loss(q_dec_taken, mini_batch_rewards, reduction="none")
                main_loss_dec = (main_loss_dec * mini_batch_sample_weights).mean()
                reg_loss_dec = torch.nn.functional.smooth_l1_loss(q_dec, target_q_dec, reduction="none")
                reg_loss_dec = (reg_loss_dec.mean(dim=-1) * mini_batch_sample_weights).mean()
                
            value_loss_dec = main_loss_dec + self.target_network_reg_weight * reg_loss_dec

            if q_bet is not None:
                q_bet_taken = q_bet.gather(dim=-1, index=mini_batch_actions[..., 1].unsqueeze(-1)).squeeze(-1)
                
                if sample_weights is None:
                    main_loss_bet = torch.nn.functional.smooth_l1_loss(q_bet_taken, mini_batch_rewards)
                    reg_loss_bet = torch.nn.functional.smooth_l1_loss(q_bet, target_q_bet)
                else:
                    main_loss_bet = torch.nn.functional.smooth_l1_loss(q_bet_taken, mini_batch_rewards, reduction="none")
                    main_loss_bet = (main_loss_bet * mini_batch_sample_weights).mean()
                    reg_loss_bet = torch.nn.functional.smooth_l1_loss(q_bet, target_q_bet, reduction="none")
                    reg_loss_bet = (reg_loss_bet.mean(dim=-1) * mini_batch_sample_weights).mean()
                    
                value_loss_bet = main_loss_bet + self.target_network_reg_weight * reg_loss_bet
                value_loss = value_loss_dec + value_loss_bet
            else:
                value_loss = value_loss_dec

            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(self.value_network.parameters(), self.grad_clip_norm)
            self.value_optimizer.step()

            self.total_update_count += 1
            if self.total_update_count % self.critic_update_ratio == 0:
                self.optimizer.zero_grad()
                
                decision_logits, bet_logits = self.get_model_logits(self.network, mini_batch_states)
                
                # Calculate V(s) = sum_a pi(a|s) Q(s,a) to compute advantages for all actions
                with torch.no_grad():
                    pi_dec = Categorical(logits=decision_logits).probs
                    # Paper Section 4: "we use entropy regularization applied to the action-values q"
                    # This prevents the policy from collapsing into deterministic strategies!
                    reg_q_dec = q_dec - self.entropy_coef * torch.log(pi_dec.clamp(min=1e-8))
                    v_dec = torch.sum(pi_dec * reg_q_dec, dim=-1)

                policy_loss_dec = -torch.sum(decision_logits * (reg_q_dec.detach() - v_dec.unsqueeze(-1).detach()), dim=-1)
                
                # L2 Penalty to bound logits to [-beta, beta] instead of Tabular zero-gradient masking
                bound_penalty_dec = torch.nn.functional.relu(decision_logits.abs() - self.beta).pow(2).sum(dim=-1)

                if bet_logits is not None:
                    with torch.no_grad():
                        pi_bet = Categorical(logits=bet_logits).probs
                        reg_q_bet = q_bet - self.entropy_coef * torch.log(pi_bet.clamp(min=1e-8))
                        v_bet = torch.sum(pi_bet * reg_q_bet, dim=-1)

                    policy_loss_bet = -torch.sum(bet_logits * (reg_q_bet.detach() - v_bet.unsqueeze(-1).detach()), dim=-1)
                    bound_penalty_bet = torch.nn.functional.relu(bet_logits.abs() - self.beta).pow(2).sum(dim=-1)

                    if sample_weights is None:
                        policy_loss = policy_loss_dec.mean() + policy_loss_bet.mean() + self.logit_penalty_weight * (bound_penalty_dec.mean() + bound_penalty_bet.mean())
                    else:
                        policy_loss = (policy_loss_dec * mini_batch_sample_weights).mean() + (policy_loss_bet * mini_batch_sample_weights).mean() + self.logit_penalty_weight * ((bound_penalty_dec * mini_batch_sample_weights).mean() + (bound_penalty_bet * mini_batch_sample_weights).mean())
                else:
                    if sample_weights is None:
                        policy_loss = policy_loss_dec.mean() + self.logit_penalty_weight * bound_penalty_dec.mean()
                    else:
                        policy_loss = (policy_loss_dec * mini_batch_sample_weights).mean() + self.logit_penalty_weight * (bound_penalty_dec * mini_batch_sample_weights).mean()

                if not torch.isfinite(policy_loss):
                    print("WARNING: loss is not finite")
                    
                policy_loss.backward()
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
            action_logits, bet_logits = self.get_model_logits(self.network, states)
            action_hist = Categorical(logits=action_logits).sample().float()
            if bet_logits is not None:
                bet_hist = Categorical(logits=bet_logits).sample().float()
            else:
                bet_hist = None

        return {"loss": policy_loss, "value_loss": value_loss, "policy_loss": policy_loss,
                "action_hist": action_hist, "betting_size": bet_hist, "rewards": batch_rewards,
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
        if bet_logits is not None:
            return Categorical(logits=decision_logits), Categorical(logits=bet_logits)
        else:
            return Categorical(logits=decision_logits), None