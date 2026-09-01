import os
import traceback

import ray
from ray.util.queue import Queue, Empty
import torch
from torch.utils.tensorboard import SummaryWriter

from src.global_settings import IS_RECURRENT
from src.game_registry import get_current_game_config
from src.game_registry import get_current_game_hyperparameters
from src.utils.historical_sampling import HistoricalSampling


class Trainer:
    def __init__(self, historical_sampling_send_queue: Queue, device, discrete: bool, log_folder: str,
                 player_save_folder: str, mode: str) -> None:
        self.historical_sampling_send_queue = historical_sampling_send_queue
        self.mode = mode
        self.alg_class = get_current_game_config()["alg"]
        self.inference_wrapper_class = get_current_game_config()["inference_wrapper"]

        if IS_RECURRENT:
            self.models = self.alg_class.init_networks(device, discrete, mode)
        else:
            self.models = self.alg_class.init_networks(device, discrete, mode)
        self.device = device
        self.discrete = discrete
        self.num_training_ran = 0
        self.log_folder = log_folder
        self.player_save_folder = player_save_folder
        log_path = os.path.join(self.log_folder, "tensorboard_logs")
        self.writer = SummaryWriter(log_dir=log_path)

    def save_player(self, player_id, params, player_training_count):
        new_weights, new_optimizer_params = params
        save_data = (new_weights, new_optimizer_params, player_training_count)
        temp_path = os.path.join(self.player_save_folder, f"{player_id}.pt.tmp")
        final_path = os.path.join(self.player_save_folder, f"{player_id}.pt")
        torch.save(save_data, temp_path)
        os.replace(temp_path, final_path)

    def run(self, player_state_dicts, data_batch, player_training_count: int, optimizer_state_dict = None):
        try:
            if len(data_batch["rewards"]) == 0:
                # The player took zero actions in this batch.
                # Return the original weights immediately without running SGD.
                return player_state_dicts, optimizer_state_dict

            hyperparameters = get_current_game_hyperparameters()
            # fill in any missing values with the default hyperparameters

            default_hyperparameters = self.alg_class.default_hyperparameters

            for key, val in default_hyperparameters.items():
                if not key in hyperparameters:
                    hyperparameters[key] = val

            alg = self.alg_class(device=self.device, mode=self.mode, discrete=self.discrete, **hyperparameters)
            alg.load_params(player_state_dicts)
            if optimizer_state_dict is not None:
                alg.load_optimizer_params(optimizer_state_dict)
            # run the model update
            metrics = alg.update(data_batch["states"], data_batch["rewards"], data_batch["actions"],
                                 batch_rnn_states=data_batch.get("batch_rnn_states", None),
                                 sample_weights=data_batch.get("sample_weights", None),
                                 new_hands=data_batch.get("new_hands", []))

            batch_size = len(data_batch["rewards"])  # NOTE: in the case of RNN model, this is the number of games, no number of transitions. [GAMES[Transitions]]
            # trainer metrics
            if self.writer is not None:
                self.writer.add_scalar(f"Loss", metrics["loss"], self.num_training_ran)
                if "entropy_loss" in metrics:
                    self.writer.add_scalar(f"Entropy_Loss", metrics["entropy_loss"], self.num_training_ran)
                self.writer.add_scalar(f"Policy_Loss", metrics["policy_loss"], self.num_training_ran)
                self.writer.add_scalar(f"Value_Loss", metrics["value_loss"], self.num_training_ran)
                self.writer.add_scalar(f"Batch_Size", batch_size, self.num_training_ran)
                if "weight_l2_change" in metrics:
                    self.writer.add_scalar(f"Weight_L2_Change", metrics["weight_l2_change"], self.num_training_ran)

                action_hist = metrics["action_hist"]
                if action_hist is not None and len(action_hist) > 0:
                    unique_actions, counts = torch.unique(action_hist, return_counts=True)
                    total_actions = counts.sum().item()
                    for action_idx, count in zip(unique_actions, counts):
                        freq = (count.item() / total_actions) * 100.0
                        self.writer.add_scalar(f"Action_{int(action_idx.item())}_Freq_%", freq, player_training_count)
                if metrics.get("betting_size") is not None:
                    self.writer.add_histogram(f"Betting_Size", metrics["betting_size"], player_training_count)
                self.writer.add_histogram(f"Rewards", metrics["rewards"], player_training_count)

                if self.mode == "beta":
                    # we grab the alpha and beta values
                    alpha, beta = metrics["alpha_hist"], metrics["beta_hist"]
                    if alpha is not None:
                        self.writer.add_histogram(f"Alpha_Dist", alpha,
                                                  player_training_count)
                    if beta is not None:
                        self.writer.add_histogram(f"Beta_Dist", beta,
                                                  player_training_count)

            # send the updated model params back to the manager
            new_weights = alg.get_params()
            new_optimizer_params = alg.get_optimizer_params()

            self.num_training_ran += 1
            return new_weights, new_optimizer_params
        except Exception as e:
            print(f"Exception: {e} encountered in Trainer ")
            traceback.print_exc()
            # abort training and send back the original weights
            return player_state_dicts, optimizer_state_dict


    def start(self, player_id, player, num_samples, sub_batches,
              training_count):
        try:
            states = []
            rewards = []
            actions = []
            sample_weights = []
            current_actors = []
            new_hands = []
            # parse the batch data
            for sub_batch in sub_batches:
                if IS_RECURRENT:
                    # We MUST filter out empty games to prevent padding crashes
                    for i in range(len(sub_batch["rewards"])):
                        if len(sub_batch["rewards"][i]) > 0:
                            states.append(sub_batch["states"][i])
                            rewards.append(sub_batch["rewards"][i])
                            actions.append(sub_batch["actions"][i])
                            sample_weights.append(sub_batch["sample_weights"][i])
                            current_actors.append(sub_batch["current_actors"][i])
                            if "new_hands" in sub_batch:
                                new_hands.append(sub_batch["new_hands"][i])
                else:
                    states.extend(sub_batch["states"])
                    rewards.extend(sub_batch["rewards"])
                    actions.extend(sub_batch["actions"])
                    sample_weights.extend(sub_batch["sample_weights"])
                    current_actors.extend(sub_batch["current_actors"])
                    if "new_hands" in sub_batch:
                        new_hands.extend(sub_batch["new_hands"])

            batch = {
                "states": (states, current_actors),
                "rewards": rewards,
                "actions": actions,
                "sample_weights": sample_weights,
                "new_hands": new_hands
            }
            # batch = ray.get(data["batch_ref"])

            params = self.run(player.get_params(), batch, training_count, player.get_optimizer_params())
            if params is not None:
                self.save_player(player_id, params, training_count + 1)

            # check if we need to add to historical players saved
            new_player_version = training_count + 1
            if HistoricalSampling.should_add(new_player_version):
                data = {
                    "player_id": player_id,
                    "player_state_dicts": params,
                    "player_version": new_player_version,
                }
                self.historical_sampling_send_queue.put_nowait(data)
            return params[0], params[1], new_player_version

        except Exception as e:
            print(f"CRITICAL TRAINER CRASH! Rescuing player {player_id}: {e}")
            traceback.print_exc()
            # If we couldn't even get the player ref, we have to fake the weights
            # to prevent the manager from crashing on receipt.
            fallback_weights = player.get_params() if 'player' in locals() else None
            fallback_optim = player.get_optimizer_params() if 'player' in locals() else None
            return fallback_weights, fallback_optim, training_count
