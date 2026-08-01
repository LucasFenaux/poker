import torch
from abc import ABC, abstractmethod

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


class InferenceWrapper(ABC):
    @abstractmethod
    def __init__(self, models, discrete):
        pass

    @abstractmethod
    def load_params(self, param_dicts):
        pass

    @abstractmethod
    def load_network_params(self, params):
        pass

    @abstractmethod
    def to(self, device):
        pass

    @abstractmethod
    def preprocess_batch(self, states_list, actors_list):
        pass

    @abstractmethod
    def get_action(self, state: tuple, hand_hidden: torch.Tensor = None,
                   game_hidden: torch.Tensor = None):
        pass

    @abstractmethod
    def get_model_policy(self, network, state, hand_hidden: torch.Tensor = None,
                   game_hidden: torch.Tensor = None):
        pass
