import ray
from global_settings import NUM_PLAYERS
from models import load_model


@ray.remote(num_cpus=1, num_gpus=float(1/NUM_PLAYERS))
class PlayerActor:
    def __init__(self, player_id: int):
        self.player_id = player_id
        self.model = load_model(player_id)
        self.alg = None
        self.buffer = []

    def act(self, state):
        return None

    def store_transition(self, state, action, reward, done):
        self.buffer.append((state, action, reward, done))

    def update(self):
        self.alg.update(self.buffer)

