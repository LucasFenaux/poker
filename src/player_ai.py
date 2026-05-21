import torch


class PlayerAI:
    def __init__(self, models, optimizer_params=None):
        self.models = models
        self.optimizer_params = optimizer_params

    def load_params(self, param_dicts):
        for model, param_dict in zip(self.models, param_dicts):
            model.load_state_dict(param_dict)

    def get_params(self):
        params = []
        for model in self.models:
            params.append(model.state_dict())
        return params

    def load_optimizers(self, optimizer_params):
        self.optimizer_params = optimizer_params

    def get_optimizer_params(self):
        return self.optimizer_params


class RNNPlayerAI(PlayerAI):
    def __init__(self, models, optimizer_params=None):
        super(RNNPlayerAI, self).__init__(models, optimizer_params)
        # Assuming the model is the first in your self.models list
        self.network = self.models[0]

    def init_hand_memory(self, batch_size: int = 1):
        """Initializes a zero-tensor for the hand-level GRU."""
        return torch.zeros(batch_size, self.network.hand_memory_size, device=next(self.network.parameters()).device)

    def init_game_memory(self, batch_size: int = 1):
        """Initializes a zero-tensor for the game-level GRU."""
        return torch.zeros(batch_size, self.network.game_memory_size, device=next(self.network.parameters()).device)