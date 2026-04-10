import ray
from ray.util.queue import Queue, Empty
import torch
import os
from torch.utils.tensorboard import SummaryWriter

from src.alg import PPO

@ray.remote(num_cpus=1)
class TrainerActor:
    def __init__(self, trainer_id: int, in_queue: Queue, out_queue: Queue, device, discrete: bool, log_folder: str,
                 player_save_folder: str, mode: str) -> None:
        # torch.set_num_threads(8)
        self.trainer_id = trainer_id
        self.in_queue = in_queue
        self.out_queue = out_queue
        self.mode = mode
        self.models = PPO.init_networks(device, discrete, mode)
        self.device = device
        self.discrete = discrete
        self.num_training_ran = 0
        self.log_folder = log_folder
        self.player_save_folder = player_save_folder
        log_path = os.path.join(self.log_folder, "tensorboard_logs")
        self.writer = SummaryWriter(log_dir=log_path)

    def save_player(self, player_id, params):
        torch.save(params, os.path.join(self.player_save_folder, f"{player_id}.pt"))

    def run(self, player_id, player_state_dicts, data_batch, player_training_count: int, optimizer_state_dict = None):
        try:
            alg = PPO(device=self.device, mode=self.mode, discrete=self.discrete, **PPO.default_hyperparameters)
            alg.load_params(player_state_dicts)
            if optimizer_state_dict is not None:
                alg.load_optimizer_params(optimizer_state_dict)
            # run the model update
            metrics = alg.update(data_batch["states"], data_batch["rewards"], data_batch["actions"],
                                 batch_rnn_states=data_batch.get("batch_rnn_states", None),
                                 sample_weights=data_batch.get("sample_weights", None))
            batch_size = len(data_batch["rewards"])
            # trainer metrics
            self.writer.add_scalar(f"Trainer_{self.trainer_id}/Loss", metrics["loss"], self.num_training_ran)
            self.writer.add_scalar(f"Trainer_{self.trainer_id}/Entropy_Loss", metrics["entropy_loss"], self.num_training_ran)
            self.writer.add_scalar(f"Trainer_{self.trainer_id}/Policy_Loss", metrics["policy_loss"], self.num_training_ran)
            self.writer.add_scalar(f"Trainer_{self.trainer_id}/Value_Loss", metrics["value_loss"], self.num_training_ran)
            self.writer.add_scalar(f"Trainer_{self.trainer_id}/Batch_Size", batch_size, self.num_training_ran)

            self.writer.add_scalar(f"Player_{player_id}/Policy_Loss", metrics["policy_loss"], player_training_count)
            self.writer.add_scalar(f"Player_{player_id}/Value_Loss", metrics["value_loss"], player_training_count)
            self.writer.add_scalar(f"Player_{player_id}/Loss", metrics["loss"], player_training_count)
            self.writer.add_scalar(f"Player_{player_id}/Entropy_Loss", metrics["entropy_loss"], player_training_count)
            self.writer.add_scalar(f"Player_{player_id}/Batch_Size", batch_size, player_training_count)
            self.writer.add_histogram(f"Player_{player_id}/Action_Dist", metrics["action_hist"], player_training_count)
            self.writer.add_histogram(f"Player_{player_id}/Betting_Size", metrics["betting_size"], player_training_count)
            self.writer.add_histogram(f"Player_{player_id}/Rewards", metrics["rewards"], player_training_count)

            if self.mode == "beta":
                # we grab the alpha and beta values
                alpha, beta = metrics["alpha_hist"], metrics["beta_hist"]
                if alpha is not None:
                    self.writer.add_histogram(f"Player_{player_id}/Alpha_Dist", alpha,
                                              player_training_count)
                if beta is not None:
                    self.writer.add_histogram(f"Player_{player_id}/Beta_Dist", beta,
                                              player_training_count)

            # send the updated model params back to the manager
            new_weights = alg.get_params()
            new_optimizer_params = alg.get_optimizer_params()
            message = {
                "type": "player",
                "player_id": player_id,
                "new_weights": new_weights,
                "new_optimizer_params": new_optimizer_params,
            }
            self.out_queue.put(message)
            self.num_training_ran += 1
            return new_weights, new_optimizer_params
        except Exception as e:
            print(f"Exception: {e} encountered in Trainer {self.trainer_id} training player: {player_id}")
            # abort training and send back the original weights
            message = {
                "type": "player",
                "player_id": player_id,
                "new_weights": player_state_dicts,
                "new_optimizer_params": optimizer_state_dict,
            }
            self.out_queue.put(message)
            return None

    def start(self):
        while True:
            try:
                data = self.in_queue.get(block=True, timeout=1)
            except Empty:
                continue

            if data["type"] == "message":
                terminate = data.get("terminate", False)  # by default we assume that we need to terminate in case of a malformed message
                if terminate:
                    # we need to send a message to the manager to alert him that we are terminating
                    message = {
                        "type": "termination",
                        "trainer_id": self.trainer_id,
                    }
                    self.out_queue.put(message)
                    return True

            assert data["type"] == "player"

            player_id = data["player_id"]

            try:
                player = ray.get(data["player_ref"])
                num_samples = data["num_samples"]

                states = []
                rewards = []
                actions = []
                sample_weights = []
                current_actors = []

                # parse the batch data
                sub_batches = ray.get(data["batch_ref"])
                for sub_batch in sub_batches:
                    states.extend(sub_batch["states"])
                    rewards.extend(sub_batch["rewards"])
                    actions.extend(sub_batch["actions"])
                    sample_weights.extend(sub_batch["sample_weights"])
                    current_actors.extend(sub_batch["current_actors"])

                batch = {
                    "states": (states, current_actors),
                    "rewards": rewards,
                    "actions": actions,
                    "sample_weights": sample_weights,
                }
                # batch = ray.get(data["batch_ref"])
                player_training_count = data["player_training_count"]

                params = self.run(player_id, player.get_params(), batch, player_training_count, player.get_optimizer_params())
                if params is not None:
                    self.save_player(player_id, params)
            except Exception as e:
                print(f"CRITICAL TRAINER CRASH! Rescuing player {player_id}: {e}")

                # If we couldn't even get the player ref, we have to fake the weights
                # to prevent the manager from crashing on receipt.
                fallback_weights = player.get_params() if 'player' in locals() else None
                fallback_optim = player.get_optimizer_params() if 'player' in locals() else None

                message = {
                    "type": "player",
                    "player_id": player_id,
                    "new_weights": fallback_weights,
                    "new_optimizer_params": fallback_optim,
                }
                self.out_queue.put(message)
