import os
os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] = "1"
import glob
import math
import pickle
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
from src.action_interpreter import Action
from src.models import load_model, get_value_model
from src.state_interpreter import StatePreprocessor

# Hyperparameters
num_epochs = 30
lr = 1e-4
value_lr = 2e-3
entropy_coef = 0.001  # Adjust this to control how "wide" the pre-trained distribution stays
batch_size = 512
data_folder = "./data/"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


class PokerBehaviorCloningDataset(Dataset):
    """
    Loads saved poker games, calculates inverse action bounds, extracts rewards,
    and preprocesses all states into raw tensor dictionaries.
    """

    def __init__(self, data_dir: str):
        self.preprocessed_states = []
        self.targets = []
        self.rewards = []

        preprocessor = StatePreprocessor()

        print("Loading dataset chunks and preprocessing states into tensors...")
        chunk_files = glob.glob(os.path.join(data_dir, "chunk_*.pkl"))
        files = tqdm(chunk_files)

        for file_path in files:
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            # We now also zip in the rewards!
            for state_snapshot, actor, (action, amt), reward in zip(data['states'], data['current_actors'],
                                                                    data['actions'], data['rewards']):

                # ==========================================
                # 1. Compute [0, 1] Targets
                # ==========================================
                if action == Action.CHECK_OR_FOLD:
                    act_target = 0.166
                elif action == Action.CHECK_OR_CALL:
                    act_target = 0.500
                elif action == Action.RAISE:
                    act_target = 0.833
                else:
                    raise ValueError(f"Unknown action: {action}")

                min_bet = state_snapshot.min_bet
                if min_bet is None:
                    min_bet = max(state_snapshot.bets) if state_snapshot.bets else 0.0

                max_bet = state_snapshot.max_bet
                if max_bet is None:
                    max_bet = min_bet

                safe_min = max(float(min_bet), 1e-5)
                safe_max = max(float(max_bet), safe_min)

                if safe_max <= safe_min:
                    bet_target = 0.0
                else:
                    log_min = math.log(safe_min)
                    log_max = math.log(safe_max)
                    clamped_amt = min(max(float(amt), safe_min), safe_max)
                    bet_target = (math.log(clamped_amt) - log_min) / (log_max - log_min)

                self.targets.append([act_target, bet_target])

                # ==========================================
                # 2. Extract Reward
                # ==========================================
                # Apply the same log-scaling to the rewards that you use in your PPO script
                sign = 1 if reward >= 0 else -1
                scaled_reward = sign * math.log(abs(reward) + 1)
                self.rewards.append(scaled_reward)

                # ==========================================
                # 3. Preprocess State to Tensor Dictionary
                # ==========================================
                processed_dict = preprocessor.process(state_snapshot, actor)

                tensor_dict = {}
                for k, v in processed_dict.items():
                    if k in ["num_players", "rel_to_button", "player_ranks", "player_suits", "board_ranks",
                             "board_suits"]:
                        tensor_dict[k] = torch.tensor(v, dtype=torch.long)
                    else:
                        tensor_dict[k] = torch.tensor(v, dtype=torch.float32)

                self.preprocessed_states.append(tensor_dict)

        print(f"Dataset fully loaded and processed! Total training steps: {len(self.targets)}")

    def __len__(self):
        return len(self.targets)

    def __getitem__(self, idx):
        # We now return the state dict, action targets, AND the reward
        return self.preprocessed_states[idx], torch.tensor(self.targets[idx], dtype=torch.float32), torch.tensor(
            self.rewards[idx], dtype=torch.float32)


def train_for_one_epoch(model, value_model, optimizer, value_optimizer, train_loader, epoch):
    model.train()
    value_model.train()

    total_p_loss = 0.0
    total_v_loss = 0.0
    total_entropy = 0.0
    pbar = tqdm(train_loader)

    for batch_idx, (states_dict, targets, rewards) in enumerate(pbar):
        # Move everything to GPU
        states_dict = {k: v.to(device) for k, v in states_dict.items()}
        targets = targets.to(device)
        rewards = rewards.to(device)

        # Apply the log-scaling dynamically on the GPU
        sign = torch.sign(rewards)
        rewards = sign * torch.log(rewards.abs() + 1)

        safe_targets = torch.clamp(targets, min=1e-5, max=1.0 - 1e-5)

        # --------------------------------------------
        # 1. Train Value Model (Critic)
        # --------------------------------------------
        value_optimizer.zero_grad()

        value_preds = value_model(states_dict).squeeze(-1)
        v_loss = F.smooth_l1_loss(value_preds, rewards)

        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=0.5)
        value_optimizer.step()

        # --------------------------------------------
        # 2. Train Policy Model (Actor) with Entropy
        # --------------------------------------------
        if epoch <= 20:
            optimizer.zero_grad()
            dist = model(states_dict)

            logp_all = dist.log_prob(safe_targets)
            entropy_all = dist.entropy()

            # Masking: Only penalize bet sizing log_prob and entropy if the action was a RAISE
            is_raise = (targets[:, 0] >= Action.get_raise_threshold()).float()

            logp = logp_all[:, 0] + (logp_all[:, 1] * is_raise)
            entropy = entropy_all[:, 0] + (entropy_all[:, 1] * is_raise)

            # NLL loss minus the entropy regularization
            p_loss = -logp.mean()
            e_loss = entropy.mean()

            loss = p_loss - (entropy_coef * e_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_p_loss += p_loss.item()
            total_entropy += e_loss.item()
        total_v_loss += v_loss.item()

        pbar.set_description(f"Epoch [{epoch}/{num_epochs}] | Policy Loss: {total_p_loss / (batch_idx+1):.4f} | "
                             f"Value Loss: {total_v_loss / (batch_idx+1):.4f} | Entropy: {total_entropy / (batch_idx+1):.4f}")


def evaluate_and_plot(model, data_loader, device, num_batches=10):
    """
    Samples actions and bet sizes from the trained model and compares
    them to the baseline bot's targets visually.
    """
    print("Evaluating model distributions...")
    model.eval()

    true_actions = []
    pred_actions = []
    true_bets = []
    pred_bets = []

    with torch.no_grad():
        for i, (states_dict, targets, rewards) in enumerate(data_loader):
            if i >= num_batches:
                break

            states_dict = {k: v.to(device) for k, v in states_dict.items()}

            # Get the model's distribution and sample from it
            dist = model(states_dict)
            samples = dist.sample().cpu().numpy()

            targets_np = targets.cpu().numpy()

            # 0th index is the action choice
            true_actions.extend(targets_np[:, 0])
            pred_actions.extend(samples[:, 0])

            # 1st index is the bet size. We ONLY care about bet sizing if it was a RAISE.
            is_raise = targets_np[:, 0] >= Action.get_raise_threshold()

            true_bets.extend(targets_np[is_raise, 1])
            pred_bets.extend(samples[is_raise, 1])

    # Helper to map raw [0, 1] numbers back to Action Strings
    def categorize_action(val):
        if val < Action.get_call_threshold():
            return "Fold"
        elif val < Action.get_raise_threshold():
            return "Call"
        else:
            return "Raise"

    true_act_labels = [categorize_action(a) for a in true_actions]
    pred_act_labels = [categorize_action(a) for a in pred_actions]

    # --- Plotting ---
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Plot 1: Action Choice Bar Chart
    sns.histplot(x=true_act_labels, color="blue", alpha=0.5, label="Baseline Bot (Target)", ax=axes[0],
                 stat="proportion", discrete=True, shrink=0.8)
    sns.histplot(x=pred_act_labels, color="orange", alpha=0.5, label="Trained Model", ax=axes[0], stat="proportion",
                 discrete=True, shrink=0.8)
    axes[0].set_title("Action Decision Distribution")
    axes[0].set_ylabel("Proportion of Actions")
    axes[0].legend()

    # Plot 2: Bet Sizing Kernel Density Estimate (KDE)
    sns.kdeplot(true_bets, color="blue", fill=True, alpha=0.3, label="Baseline Bot", ax=axes[1], clip=(0, 1))
    sns.kdeplot(pred_bets, color="orange", fill=True, alpha=0.3, label="Trained Model", ax=axes[1], clip=(0, 1))
    axes[1].set_title("Bet Sizing Distribution (When Raising)")
    axes[1].set_xlabel("Scaled Bet Size [0, 1]")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("bc_distribution_comparison.png", dpi=300)
    print("Saved distribution plot to 'bc_distribution_comparison.png'")


if __name__ == '__main__':
    print(f"Initializing training on device: {device}")

    # 1. Setup Dataset & DataLoader
    dataset = PokerBehaviorCloningDataset(data_dir=data_folder)
    train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    # 2. Initialize Models and Optimizers
    model = load_model(player_id=0, device=device, deterministic=False, mode="beta")
    value_model = get_value_model(device=device)

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=value_lr)

    # 3. Run the Training Loop
    print("Starting Behavior Cloning...")
    for epoch in range(1, num_epochs + 1):
        train_for_one_epoch(model, value_model, optimizer, value_optimizer, train_loader, epoch)

    print("Pre-training complete! Ready for Self-Play.")

    # Save the models so PPO can load them
    torch.save([model.state_dict(), value_model.state_dict()], "bc_pretrained_model.pt")

    evaluate_and_plot(model, train_loader, device, num_batches=10)
