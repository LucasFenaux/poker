from typing import Union
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import torch.nn.functional as F
import pokerkit

from state_interpreter import StateInterpreter, StateSnapshot

LOG_STD_MIN = -20.0
LOG_STD_MAX = 2.0


class PokerModel(nn.Module):
    def __init__(self, interpreter: StateInterpreter, deterministic: bool):
        super(PokerModel, self).__init__()
        self.interpreter = interpreter
        self.input_dim = interpreter.expected_input_size()

        self.embed_net = nn.Sequential(nn.Linear(self.input_dim, 64), nn.GELU(),
                                    nn.Linear(64, 16), nn.GELU())

        # one dim for which action and one dim for bet sizing
        self.mu_net = nn.Linear(16, 2)
        if not deterministic:
            self.log_std_net = nn.Linear(16, 2)
        else:
            self.log_std_net = None
        self.deterministic = deterministic

    def _forward(self, feature_vector: torch.Tensor):
        if self.deterministic:
            return self.mu_net(self.embed_net(feature_vector))
        else:
            feature_embedding = self.embed_net(feature_vector)
            mu = self.mu_net(feature_embedding)
            # log_std = self.log_std_net(feature_embedding)
            # log_std = torch.clamp(log_std, LOG_STD_MIN, LOG_STD_MAX)
            # std = torch.exp(log_std)
            # treat it as straight std
            std = F.softplus(self.log_std_net(feature_embedding)) + 1e-5

            dist = Normal(mu, std)
            return dist

    def forward(self, state: Union[pokerkit.State, StateSnapshot, list], current_actor: Union[int, list[int]]):
        if isinstance(state, (pokerkit.State, StateSnapshot)) and isinstance(current_actor, int):
            feature_vector = self.interpreter(state, current_actor)
        elif isinstance(state, list) and isinstance(current_actor, list):
            feature_vector = []
            for state, c_a in zip(state, current_actor):
                feature_vector.append(self.interpreter(state, c_a))
            feature_vector = torch.stack(feature_vector)
        else:
            raise NotImplementedError

        return self._forward(feature_vector)


class ValueModel(nn.Module):
    def __init__(self, interpreter: StateInterpreter):
        super(ValueModel, self).__init__()
        self.interpreter = interpreter
        self.input_dim = interpreter.expected_input_size()

        self.net = nn.Sequential(nn.Linear(self.input_dim, 64), nn.GELU(),
                                 nn.Linear(64, 16), nn.GELU(),
                                 nn.Linear(16, 1))


    def _forward(self, feature_vector: torch.Tensor):
        return self.net(feature_vector)

    def forward(self, state: Union[pokerkit.State, StateSnapshot, list], current_actor: Union[int, list[int]]):
        if isinstance(state, (pokerkit.State, StateSnapshot)) and isinstance(current_actor, int):
            feature_vector = self.interpreter(state, current_actor)
        elif isinstance(state, list) and isinstance(current_actor, list):
            feature_vector = []
            for state, c_a in zip(state, current_actor):
                feature_vector.append(self.interpreter(state, c_a))
            feature_vector = torch.stack(feature_vector)
        else:
            raise NotImplementedError

        return self._forward(feature_vector)


def get_value_model(device: torch.device) -> ValueModel:
    interpreter = StateInterpreter(device).to(device)
    return ValueModel(interpreter).to(device)


def load_model(player_id, device, deterministic=False) -> PokerModel:
    # TODO: implement saving/loading of player models
    interpreter = StateInterpreter(device).to(device)
    model  = PokerModel(interpreter, deterministic).to(device)
    return model


def load_dummy_model(device, deterministic=False) -> PokerModel:
    interpreter = StateInterpreter(device).to(device)
    model  = PokerModel(interpreter, deterministic).to(device)
    return model