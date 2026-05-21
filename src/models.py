from typing import Union
import torch
import torch.nn as nn
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta
import torch.nn.functional as F
import pokerkit
from src.ppo_self_play.global_settings import IS_RECURRENT
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
            # alpha = torch.clamp(alpha, min=0.01, max=500.0)
            # beta = torch.clamp(beta, min=0.01, max=500.0)
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



class HierarchicalPokerModel(nn.Module):
    def __init__(self, interpreter: StateInterpreter, deterministic: bool, mode: str):
        super(HierarchicalPokerModel, self).__init__()
        self.interpreter = interpreter
        self.input_dim = interpreter.expected_input_size()
        self.mode = mode
        self.hand_memory_size = 64
        self.game_memory_size = 32
        self.hand_gru = nn.GRUCell(input_size=128, hidden_size=self.hand_memory_size)
        self.game_gru = nn.GRUCell(input_size=self.hand_memory_size, hidden_size=self.game_memory_size)

        self.embed_net = nn.Sequential(nn.Linear(self.input_dim, 256), nn.GELU(),
                                 nn.Linear(256, 128), nn.GELU(),)
                                 # nn.Linear(128, 128), nn.GELU(),)

        self.policy_mlp = nn.Sequential(
            nn.Linear(128+64+32, 128), nn.GELU(),
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

    def update_game_memory(self, final_hand_hidden: torch.Tensor, current_game_hidden: torch.Tensor):
        """Call this ONLY at the end of a hand to update the long-term session memory."""
        return self.game_gru(final_hand_hidden, current_game_hidden)

    def _get_distribution(self, policy_features: torch.Tensor):
        if self.mode == "beta":
            if self.deterministic:
                return self.alpha_net(policy_features)
            alpha = F.softplus(self.alpha_net(policy_features)) + 1e-5
            beta = F.softplus(self.beta_net(policy_features)) + 1e-5
            alpha = torch.clamp(alpha, min=0.01, max=500.0)
            beta = torch.clamp(beta, min=0.01, max=500.0)
            return Beta(alpha, beta)
        elif self.mode == "normal":
            if self.deterministic:
                return self.mu_net(policy_features)
            mu = self.mu_net(policy_features)
            std = F.softplus(self.std_net(policy_features)) + 1e-5
            return Normal(mu, std)

    def forward(self, state: Union[pokerkit.State, StateSnapshot, list, torch.Tensor, dict],
                hand_hidden: torch.Tensor,
                game_hidden: torch.Tensor,
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

        x = self.embed_net(feature_vector)

        new_hand_hidden = self.hand_gru(x, hand_hidden)

        combined = torch.cat([x, new_hand_hidden, game_hidden], dim=-1)
        policy_features = self.policy_mlp(combined)
        dist = self._get_distribution(policy_features)

        # RETURN TUPLE: The Distribution AND the new Hand State (to pass to the next step)
        return dist, new_hand_hidden


class HierarchicalValueModel(nn.Module):
    def __init__(self, interpreter: StateInterpreter):
        super(HierarchicalValueModel, self).__init__()
        self.interpreter = interpreter
        self.input_dim = interpreter.expected_input_size()

        self.hand_memory_size = 64
        self.game_memory_size = 32
        self.hand_gru = nn.GRUCell(input_size=128, hidden_size=self.hand_memory_size)
        self.game_gru = nn.GRUCell(input_size=self.hand_memory_size, hidden_size=self.game_memory_size)

        self.embed_net = nn.Sequential(nn.Linear(self.input_dim, 256), nn.GELU(),
                                 nn.Linear(256, 128), nn.GELU(),)
                                 # nn.Linear(128, 128), nn.GELU(),)

        self.value_head = nn.Sequential(
            nn.Linear(128+64+32, 128), nn.GELU(),
            nn.Linear(128, 64), nn.GELU(),
            nn.Linear(64, 1)
        )

    def update_game_memory(self, final_hand_hidden: torch.Tensor, current_game_hidden: torch.Tensor):
        return self.game_gru(final_hand_hidden, current_game_hidden)

    def forward(self, state: Union[pokerkit.State, StateSnapshot, list, torch.Tensor, dict],
                hand_hidden: torch.Tensor,
                game_hidden: torch.Tensor,
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

        x = self.embed_net(feature_vector)

        new_hand_hidden = self.hand_gru(x, hand_hidden)
        combined = torch.cat([x, new_hand_hidden, game_hidden], dim=-1)
        value = self.value_head(combined)

        return value, new_hand_hidden


def get_value_model(device: torch.device) -> Union[ValueModel, HierarchicalValueModel]:
    interpreter = StateInterpreter(device).to(device)
    if IS_RECURRENT:
        return HierarchicalValueModel(interpreter).to(device)
    else:
        return ValueModel(interpreter).to(device)


def load_model(player_id, device, deterministic=False, mode="beta") -> Union[PokerModel, HierarchicalPokerModel]:
    interpreter = StateInterpreter(device).to(device)
    if IS_RECURRENT:
        model = HierarchicalPokerModel(interpreter, deterministic, mode).to(device)
    else:
        model = PokerModel(interpreter, deterministic, mode).to(device)
    return model


def load_dummy_model(device, deterministic=False, mode="beta") -> Union[PokerModel, HierarchicalPokerModel]:
    interpreter = StateInterpreter(device).to(device)
    if IS_RECURRENT:
        model = HierarchicalPokerModel(interpreter, deterministic, mode).to(device)
    else:
        model = PokerModel(interpreter, deterministic, mode).to(device)
    return model