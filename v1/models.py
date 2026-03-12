from typing import Union
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
import pokerkit

from state_interpreter import StateInterpreter


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
            log_std = self.log_std_net(feature_embedding)
            dist = Normal(mu, torch.exp(log_std))
            return dist

    def forward(self, state: Union[pokerkit.State, list[pokerkit.State]], current_actor: Union[int, list[int]]):
        if isinstance(state, pokerkit.State) and isinstance(current_actor, int):
            feature_vector = self.interpreter(state, current_actor)
        elif isinstance(state, list) and isinstance(current_actor, list):
            feature_vector = []
            for state, current_actor in zip(state, current_actor):
                feature_vector.append(self.interpreter(state, current_actor))
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


    def forward(self, state: pokerkit.State, current_actor: int):
        feature_vector = self.interpreter(state, current_actor)
        return self.net(feature_vector)


def get_value_model(device: torch.device) -> ValueModel:
    interpreter = StateInterpreter(device).to(device)
    return ValueModel(interpreter).to(device)


def load_model(player_id, device, deterministic=False) -> PokerModel:
    # TODO: implement saving/loading of player models
    interpreter = StateInterpreter(device).to(device)
    model  = PokerModel(interpreter, deterministic).to(device)
    return model