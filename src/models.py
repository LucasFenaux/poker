from typing import Union
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
import torch.nn.functional as F
import pokerkit

from src.state_interpreter import StateInterpreter, StateSnapshot, StatePreprocessor

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


def preprocess_raw_states(states_list, actors_list, device):
    """Helper function to intercept raw states and batch them into a tensor dict."""
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
            tensor_dict[k] = torch.tensor(v, dtype=torch.long, device=device)
        else:
            tensor_dict[k] = torch.tensor(v, dtype=torch.float32, device=device)
    return tensor_dict


class PokerModel(nn.Module):
    def __init__(self, interpreter: StateInterpreter, deterministic: bool, mode: str):
        super(PokerModel, self).__init__()
        self.interpreter = interpreter
        self.input_dim = interpreter.expected_input_size()
        self.mode = mode

        self.embed_net = nn.Sequential(
            nn.Linear(self.input_dim, 256), nn.GELU(),
            nn.Linear(256, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU()
        )
        self.deterministic = deterministic

        self.mu_net = None
        self.std_net = None
        self.alpha_net = None
        self.beta_net = None
        if self.mode == "normal":
            self.mu_net = nn.Linear(64, 2)
            if not deterministic:
                self.std_net = nn.Linear(64, 2)
        elif self.mode == "beta":
            self.alpha_net = nn.Linear(64, 2)
            nn.init.constant_(self.alpha_net.bias, 1.0)
            if not self.deterministic:
                self.beta_net = nn.Linear(64, 2)
                nn.init.constant_(self.beta_net.bias, 1.0)
        else:
            raise NotImplementedError(self.mode)

    def _forward_beta(self, feature_vector: torch.Tensor):
        if self.deterministic:
            return self.alpha_net(self.embed_net(feature_vector))
        else:
            feature_embedding = self.embed_net(feature_vector)
            alpha = F.softplus(self.alpha_net(feature_embedding)) + 1e-5
            beta = F.softplus(self.beta_net(feature_embedding)) + 1e-5
            # alpha = torch.clamp(alpha, min=0.01, max=50.0)
            # beta = torch.clamp(beta, min=0.01, max=50.0)
            alpha = torch.clamp(alpha, min=0.01, max=500.0)
            beta = torch.clamp(beta, min=0.01, max=500.0)
            dist = Beta(alpha, beta)
            return dist

    def _forward_normal(self, feature_vector: torch.Tensor):
        if self.deterministic:
            return self.mu_net(self.embed_net(feature_vector))
        else:
            feature_embedding = self.embed_net(feature_vector)
            mu = self.mu_net(feature_embedding)
            std = F.softplus(self.std_net(feature_embedding)) + 1e-5
            dist = Normal(mu, std)
            return dist

    def forward(self, state: Union[pokerkit.State, StateSnapshot, list, torch.Tensor, dict],
                current_actor: Union[int, list[int]] = None):
        # 1. Handle perfectly preprocessed dictionaries (From train.py / alg.py)
        if isinstance(state, dict):
            feature_vector = self.interpreter(state)

        # 2. Handle already embedded tensors
        elif isinstance(state, torch.Tensor) and current_actor is None:
            feature_vector = state

        # 3. Handle raw single states (From evaluate_puzzles.py)
        elif isinstance(state, (pokerkit.State, StateSnapshot)) and isinstance(current_actor, int):
            device = next(self.parameters()).device
            tensor_dict = preprocess_raw_states([state], [current_actor], device)
            feature_vector = self.interpreter(tensor_dict)

        # 4. Handle raw lists of states
        elif isinstance(state, list) and isinstance(current_actor, list):
            device = next(self.parameters()).device
            tensor_dict = preprocess_raw_states(state, current_actor, device)
            feature_vector = self.interpreter(tensor_dict)

        else:
            raise NotImplementedError(f"Unsupported input types: {type(state)}, {type(current_actor)}")

        if self.mode == "normal":
            return self._forward_normal(feature_vector)
        elif self.mode == "beta":
            return self._forward_beta(feature_vector)
        else:
            raise NotImplementedError


class ValueModel(nn.Module):
    def __init__(self, interpreter: StateInterpreter):
        super(ValueModel, self).__init__()
        self.interpreter = interpreter
        self.input_dim = interpreter.expected_input_size()

        self.net = nn.Sequential(nn.Linear(self.input_dim, 256), nn.GELU(),
                                 nn.Linear(256, 128), nn.GELU(),
                                 nn.Linear(128, 64), nn.GELU(),
                                 nn.Linear(64, 1))

    def _forward(self, feature_vector: torch.Tensor):
        return self.net(feature_vector)

    def forward(self, state: Union[pokerkit.State, StateSnapshot, list, torch.Tensor, dict],
                current_actor: Union[int, list[int]] = None):
        # Mirroring the robust routing from PokerModel
        if isinstance(state, dict):
            feature_vector = self.interpreter(state)
        elif isinstance(state, torch.Tensor) and current_actor is None:
            feature_vector = state
        elif isinstance(state, (pokerkit.State, StateSnapshot)) and isinstance(current_actor, int):
            device = next(self.parameters()).device
            tensor_dict = preprocess_raw_states([state], [current_actor], device)
            feature_vector = self.interpreter(tensor_dict)
        elif isinstance(state, list) and isinstance(current_actor, list):
            device = next(self.parameters()).device
            tensor_dict = preprocess_raw_states(state, current_actor, device)
            feature_vector = self.interpreter(tensor_dict)
        else:
            raise NotImplementedError(f"Unsupported input types: {type(state)}, {type(current_actor)}")

        return self._forward(feature_vector)


def get_value_model(device: torch.device) -> ValueModel:
    interpreter = StateInterpreter(device).to(device)
    return ValueModel(interpreter).to(device)


def load_model(player_id, device, deterministic=False, mode="beta") -> PokerModel:
    interpreter = StateInterpreter(device).to(device)
    model = PokerModel(interpreter, deterministic, mode).to(device)
    return model


def load_dummy_model(device, deterministic=False, mode="beta") -> PokerModel:
    interpreter = StateInterpreter(device).to(device)
    model = PokerModel(interpreter, deterministic, mode).to(device)
    return model