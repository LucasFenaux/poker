import ray
from ray.util.queue import Queue, Empty
import time
import os
from torch.utils.tensorboard import SummaryWriter

from models import load_dummy_model
from alg import PPO


@ray.remote(num_cpus=1)
class TrainerActor:
    def __init__(self, trainer_id: int, in_queue: Queue, out_queue: Queue, device, discrete: bool, log_folder: str):
        self.trainer_id = trainer_id
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.models = PPO.init_networks(device, discrete)
        self.device = device
        self.discrete = discrete
        self.num_training_ran = 0
        self.log_folder = log_folder
        log_path = os.path.join(self.log_folder, "tensorboard_logs")
        self.writer = SummaryWriter(log_dir=log_path)

    def run(self, player_id, player_state_dicts, data_batch, player_training_count: int):
        try:
            alg = PPO(device=self.device, discrete=self.discrete, **PPO.default_hyperparameters)
            alg.load_params(player_state_dicts)
            # load the state_dict into the model
            # self.model.load_state_dict(player_state_dict)
            # self.model.train()
            # self.model = self.model.to(self.device)

            # create the training algorithm

            # run the model update
            metrics = alg.update(data_batch["states"], data_batch["rewards"], data_batch["actions"],
                       batch_rnn_states=data_batch.get("batch_rnn_states", None))
            # trainer metrics
            self.writer.add_scalar(f"Trainer_{self.trainer_id}/Policy_Loss", metrics["loss"], self.num_training_ran)
            self.writer.add_scalar(f"Trainer_{self.trainer_id}/Value_Loss", metrics["value_loss"], self.num_training_ran)

            self.writer.add_scalar(f"Player_{player_id}/Policy_Loss", metrics["loss"], player_training_count)
            self.writer.add_scalar(f"Player_{player_id}/Value_Loss", metrics["value_loss"], player_training_count)

            # send the updated model params back to the manager
            new_weights = alg.get_params()
            self.out_queue.put((player_id, new_weights))
            self.num_training_ran += 1
        except Exception as e:
            print(f"Exception: {e} encountered in Trainer {self.trainer_id} training player: {player_id}")
            # abort training and send back the original weights
            self.out_queue.put((player_id, player_state_dicts))

    def start(self):
        while True:
            try:
                data = self.in_queue.get(block=True, timeout=1)
            except Empty:
                continue

            if data["type"] == "message":
                terminate = data.get("terminate", True)  # by default we assume that we need to terminate in case of a malformed message
                if terminate:
                    return True

            assert data["type"] == "player"

            self.run(data["player_id"], data["state_dicts"], data["data_batch"], data["player_training_count"])


