import ray
import torch
import pokerkit
import os
from global_settings import NUM_PLAYERS, NUM_GPUS, NUM_CPUS
from models import load_model
from alg import PPO, OnPolicyAlgorithm


@ray.remote(num_cpus=0.5*(NUM_CPUS/NUM_PLAYERS), num_gpus=0.5*(float(NUM_GPUS/NUM_PLAYERS)))
class PlayerActor:
    def __init__(self, player_id: int, save_folder: str, device: torch.device, deterministic: bool = False):
        self.player_id = player_id
        self.deterministic = deterministic
        self.save_folder = save_folder
        self.model = load_model(player_id, device, deterministic).to(device)
        self.alg = PPO(self.model, device=device, discrete=deterministic, **PPO.default_hyperparameters)
        self.buffer = None
        self.batch_states = []
        self.batch_current_actors = []
        self.batch_rewards = []
        self.batch_actions = []
        self.gamma = 0.99
        self.batch_size = 5000

    def get_id(self):
        return self.player_id

    def get_action(self, state: pokerkit.State, current_actor: int):
        return self.alg.get_action((state, current_actor), rnn_state=None)

    def store_hand(self, states, current_actors, actions, reward):
        assert len(states) == len(actions) == len(current_actors)
        if isinstance(self.alg, OnPolicyAlgorithm):
            # there is no notion of discounting rewards in a hand of poker
            # since rewards always come at a specific time and when the player gets
            # the reward is irrelevant
            rewards = [reward] * len(states)

            self.batch_states.extend(states)
            self.batch_current_actors.extend(current_actors)
            self.batch_rewards.extend(rewards)
            self.batch_actions.extend(actions)
        else:
            raise NotImplementedError

    def update(self):
        if isinstance(self.alg, OnPolicyAlgorithm):
            if len(self.batch_states) >= self.batch_size:
                # we need batch_states, batch_rewards, and batch_actions
                self.alg.update((self.batch_states, self.batch_current_actors), self.batch_rewards, self.batch_actions)
                self.batch_states, self.batch_current_actors, self.batch_rewards, self.batch_actions = [], [], [], []
            # else:
            #     if self.player_id == 0:
            #         print(f"Player 0 current batch len is {len(self.batch_states)}")

    def save(self):
        torch.save(self.model.state_dict(), os.path.join(self.save_folder, f"{self.player_id}.pt"))

    def load(self):
        self.model.load_state_dict(torch.load(os.path.join(self.save_folder, f"{self.player_id}.pt")))