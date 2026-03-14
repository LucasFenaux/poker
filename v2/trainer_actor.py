import ray
from ray.util.queue import Queue, Empty

from models import load_dummy_model
from alg import PPO


@ray.remote(num_cpus=1)
class TrainerActor:
    def __init__(self, in_queue: Queue, out_queue: Queue, device, discrete: bool):
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.model = load_dummy_model(device, discrete)
        self.device = device
        self.discrete = discrete
        self.start()

    def run(self, player_id, player_state_dict, data_batch):
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

    def start(self):
        while True:
            try:
                data = self.in_queue.get(block=True, timeout=1)
            except Empty:
                data = None

            if data["type"] == "message":
                terminate = data.get("terminate", True)  # by default we assume that we need to terminate in case of a malformed message
                if terminate:
                    return True

            assert data["type"] == "player"

            self.run(data["player_id"], data["state_dict"], data["data_batch"])


