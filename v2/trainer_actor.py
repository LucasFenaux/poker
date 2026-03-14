import ray
from ray.util.queue import Queue, Empty
import time

from models import load_dummy_model
from alg import PPO


@ray.remote(num_cpus=1)
class TrainerActor:
    def __init__(self, trainer_id: int, in_queue: Queue, out_queue: Queue, device, discrete: bool):
        self.trainer_id = trainer_id
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.model = load_dummy_model(device, discrete)
        self.device = device
        self.discrete = discrete
        self.num_training_ran = 0

    def run(self, player_id, player_state_dict, data_batch):
        try:
            # load the state_dict into the model
            self.model.load_state_dict(player_state_dict)
            self.model.train()
            self.model = self.model.to(self.device)

            # create the training algorithm
            alg = PPO(self.model, device=self.device, discrete=self.discrete, **PPO.default_hyperparameters)

            # run the model update
            alg.update(data_batch["states"], data_batch["rewards"], data_batch["actions"],
                       batch_rnn_states=data_batch.get("batch_rnn_states", None))

            # send the updated model params back to the manager
            new_weights = alg.get_network().state_dict()
            self.out_queue.put((player_id, new_weights))
            self.num_training_ran += 1
        except Exception as e:
            print(f"Exception: {e} encountered in Trainer {self.trainer_id} training player: {player_id}")
            # abort training and send back the original weights
            self.out_queue.put((player_id, player_state_dict))

    def start(self):
        while True:
            try:
                data = self.in_queue.get(block=True, timeout=1)
            except Empty:
                continue
            time.sleep(1)
            if data["type"] == "message":
                terminate = data.get("terminate", True)  # by default we assume that we need to terminate in case of a malformed message
                if terminate:
                    return True

            assert data["type"] == "player"

            self.run(data["player_id"], data["state_dict"], data["data_batch"])


