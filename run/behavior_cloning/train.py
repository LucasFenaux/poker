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
from torch.nn.utils.rnn import pad_sequence

from src.action_interpreter import Action
from src.models import load_model, get_value_model
from src.state_interpreter import StatePreprocessor
from src.ppo_self_play.global_settings import IS_RECURRENT
from src.ppo_self_play.alg import RNNPPO
from torch.distributions.normal import Normal
from torch.distributions.beta import Beta


# Hyperparameters
num_epochs = 50 if IS_RECURRENT else 20
lr = 1e-4
value_lr = 2e-3
entropy_coef = 0.001
batch_size = 512
data_folder = f"./data/{'rnn' if IS_RECURRENT else 'no_mem'}/"

if torch.cuda.is_available():
    device = torch.device("cuda")
elif torch.mps.is_available():
    device = torch.device("mps")
else:
    device = torch.device("cpu")


# ==========================================
# DATASET AND COLLATE FUNCTIONS
# ==========================================
class PokerBehaviorCloningDataset(Dataset):
    def __init__(self, data_dir: str):
        self.data_items = []
        preprocessor = StatePreprocessor()

        print(f"Loading dataset chunks (IS_RECURRENT={IS_RECURRENT})...")
        chunk_files = glob.glob(os.path.join(data_dir, "chunk_*.pkl"))

        for file_path in tqdm(chunk_files):
            with open(file_path, 'rb') as f:
                data = pickle.load(f)

            if IS_RECURRENT:
                # Process as Sequences
                for game_states, game_actors, game_actions, game_rewards in zip(
                        data['states'], data['current_actors'], data['actions'], data['rewards']):

                    seq_targets, seq_rewards, seq_states = [], [], []
                    for state_snapshot, actor, (action, amt), reward in zip(game_states, game_actors, game_actions,
                                                                            game_rewards):
                        targets, processed_state = self._process_step(state_snapshot, actor, action, amt, preprocessor)
                        seq_targets.append(targets)
                        seq_rewards.append(reward)
                        seq_states.append(processed_state)

                    if len(seq_targets) == 0:
                        continue

                    self.data_items.append({
                        "states": seq_states,
                        "targets": torch.tensor(seq_targets, dtype=torch.float32),
                        "rewards": torch.tensor(seq_rewards, dtype=torch.float32)
                    })
            else:
                # Process as Flat (Original Logic)
                for state_snapshot, actor, (action, amt), reward in zip(
                        data['states'], data['current_actors'], data['actions'], data['rewards']):
                    targets, processed_state = self._process_step(state_snapshot, actor, action, amt, preprocessor)
                    self.data_items.append({
                        "state": processed_state,
                        "target": torch.tensor(targets, dtype=torch.float32),
                        "reward": torch.tensor(reward, dtype=torch.float32)
                    })

    def _process_step(self, state_snapshot, actor, action, amt, preprocessor):
        # 1. Compute [0, 1] Targets
        if action == Action.CHECK_OR_FOLD:
            act_target = 0.166
        elif action == Action.CHECK_OR_CALL:
            act_target = 0.500
        elif action == Action.RAISE:
            act_target = 0.833
        else:
            raise ValueError(f"Unknown action: {action}")

        min_bet = state_snapshot.min_bet or (max(state_snapshot.bets) if state_snapshot.bets else 0.0)
        max_bet = state_snapshot.max_bet or min_bet
        safe_min = max(float(min_bet), 1e-5)
        safe_max = max(float(max_bet), safe_min)

        if safe_max <= safe_min:
            bet_target = 0.0
        else:
            log_min, log_max = math.log(safe_min), math.log(safe_max)
            clamped_amt = min(max(float(amt), safe_min), safe_max)
            bet_target = (math.log(clamped_amt) - log_min) / (log_max - log_min)

        # 2. Preprocess State
        processed_dict = preprocessor.process(state_snapshot, actor)
        tensor_dict = {
            k: torch.tensor(v, dtype=torch.long if k in ["num_players", "rel_to_button", "player_ranks", "player_suits",
                                                         "board_ranks", "board_suits"] else torch.float32)
            for k, v in processed_dict.items()
        }
        return [act_target, bet_target], tensor_dict

    def __len__(self):
        return len(self.data_items)

    def __getitem__(self, idx):
        if IS_RECURRENT:
            return self.data_items[idx]
        else:
            item = self.data_items[idx]
            return item["state"], item["target"], item["reward"]


def rnn_collate_fn(batch):
    targets = [item["targets"] for item in batch]
    rewards = [item["rewards"] for item in batch]

    seq_lengths = [len(t) for t in targets]
    max_len = max(seq_lengths)
    batch_size = len(batch)

    padded_targets = pad_sequence(targets, batch_first=True, padding_value=0.0)
    padded_rewards = pad_sequence(rewards, batch_first=True, padding_value=0.0)
    mask = torch.arange(max_len).expand(batch_size, max_len) < torch.tensor(seq_lengths).unsqueeze(1)

    padded_states = {}
    for key in batch[0]["states"][0].keys():
        key_seqs = [torch.stack([step[key] for step in item["states"]]) for item in batch]
        padded_states[key] = pad_sequence(key_seqs, batch_first=True, padding_value=0.0)

    return padded_states, padded_targets, padded_rewards, mask


# ==========================================
# TRAINING LOOPS
# ==========================================
def _unroll_policy_bc(model, states_dict, h_0, g_0):
    """Unrolls the 3D sequence step-by-step to prevent GRUCell dimensionality crashes."""
    max_seq_len = states_dict['num_players'].size(1)
    all_dist_params = []
    h, g = h_0, g_0

    for t in range(max_seq_len):
        step_dict = {k: v[:, t] for k, v in states_dict.items()}
        dist, h = model(step_dict, hand_hidden=h, game_hidden=g)
        if model.mode == "normal":
            all_dist_params.append((dist.loc, dist.scale))
        elif model.mode == "beta":
            all_dist_params.append((dist.concentration1, dist.concentration0))

    if model.mode == "normal":
        mu = torch.stack([p[0] for p in all_dist_params], dim=1)
        std = torch.stack([p[1] for p in all_dist_params], dim=1)
        return Normal(mu, std)
    elif model.mode == "beta":
        alpha = torch.stack([p[0] for p in all_dist_params], dim=1)
        beta = torch.stack([p[1] for p in all_dist_params], dim=1)
        return Beta(alpha, beta)


def _unroll_value_bc(value_model, states_dict, h_0, g_0):
    """Unrolls the 3D sequence step-by-step for the Value model."""
    max_seq_len = states_dict['num_players'].size(1)
    all_values = []
    h, g = h_0, g_0

    for t in range(max_seq_len):
        step_dict = {k: v[:, t] for k, v in states_dict.items()}
        val, h = value_model(step_dict, hand_hidden=h, game_hidden=g)
        all_values.append(val)

    return torch.stack(all_values, dim=1)


def train_flat_epoch(model, value_model, optimizer, value_optimizer, train_loader, epoch):
    model.train()
    value_model.train()
    total_p_loss, total_v_loss, total_entropy = 0.0, 0.0, 0.0
    pbar = tqdm(train_loader)

    for batch_idx, (states_dict, targets, rewards) in enumerate(pbar):
        states_dict = {k: v.to(device) for k, v in states_dict.items()}
        targets = targets.to(device)
        rewards = rewards.to(device)
        safe_targets = torch.clamp(targets, min=1e-5, max=1.0 - 1e-5)

        # Value Network
        value_optimizer.zero_grad()
        value_preds = value_model(states_dict).squeeze(-1)
        v_loss = F.smooth_l1_loss(value_preds, rewards)
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=0.5)
        value_optimizer.step()

        # Policy Network
        if epoch <= 4*num_epochs/5:
            optimizer.zero_grad()
            dist = model(states_dict)
            logp_all = dist.log_prob(safe_targets)
            entropy_all = dist.entropy()

            is_raise = (targets[:, 0] >= Action.get_raise_threshold()).float()
            logp = logp_all[:, 0] + (logp_all[:, 1] * is_raise)
            entropy = entropy_all[:, 0] + (entropy_all[:, 1] * is_raise)

            p_loss = -logp.mean()
            e_loss = entropy.mean()
            loss = p_loss - (entropy_coef * e_loss)

            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_p_loss += p_loss.item()
            total_entropy += e_loss.item()

        total_v_loss += v_loss.item()
        pbar.set_description(
            f"Flat Epoch [{epoch}/{num_epochs}] | Policy: {total_p_loss / (batch_idx + 1):.4f} | Value: {total_v_loss / (batch_idx + 1):.4f}")


def train_rnn_epoch(model, value_model, optimizer, value_optimizer, train_loader, epoch):
    model.train()
    value_model.train()
    total_p_loss, total_v_loss, total_entropy = 0.0, 0.0, 0.0
    pbar = tqdm(train_loader)

    for batch_idx, (states_dict, targets, rewards, mask) in enumerate(pbar):
        states_dict = {k: v.to(device) for k, v in states_dict.items()}
        targets, rewards, mask = targets.to(device), rewards.to(device), mask.to(device)
        safe_targets = torch.clamp(targets, min=1e-5, max=1.0 - 1e-5)
        batch_size = targets.size(0)

        # Explicitly initialize hidden states to zero tensors to pass into the unrollers
        h_p = torch.zeros(batch_size, model.hand_memory_size, device=device)
        g_p = torch.zeros(batch_size, model.game_memory_size, device=device)
        h_v = torch.zeros(batch_size, value_model.hand_memory_size, device=device)
        g_v = torch.zeros(batch_size, value_model.game_memory_size, device=device)

        # Value Network
        value_optimizer.zero_grad()
        value_preds = _unroll_value_bc(value_model, states_dict, h_v, g_v).squeeze(-1)
        v_loss_unreduced = F.smooth_l1_loss(value_preds, rewards, reduction="none")
        v_loss = (v_loss_unreduced * mask).sum() / mask.sum()
        v_loss.backward()
        torch.nn.utils.clip_grad_norm_(value_model.parameters(), max_norm=0.5)
        value_optimizer.step()

        # Policy Network
        if epoch <= 4*num_epochs/5:
            optimizer.zero_grad()
            dist = _unroll_policy_bc(model, states_dict, h_p, g_p)

            logp_all = dist.log_prob(safe_targets)
            entropy_all = dist.entropy()

            is_raise = (targets[:, :, 0] >= Action.get_raise_threshold()).float()
            logp = logp_all[:, :, 0] + (logp_all[:, :, 1] * is_raise)
            entropy = entropy_all[:, :, 0] + (entropy_all[:, :, 1] * is_raise)

            p_loss_unreduced = -logp
            p_loss = (p_loss_unreduced * mask).sum() / mask.sum()
            e_loss = (entropy * mask).sum() / mask.sum()

            loss = p_loss - (entropy_coef * e_loss)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.5)
            optimizer.step()

            total_p_loss += p_loss.item()
            total_entropy += e_loss.item()

        total_v_loss += v_loss.item()
        pbar.set_description(
            f"RNN Epoch [{epoch}/{num_epochs}] | Policy: {total_p_loss / (batch_idx + 1):.4f} | Value: {total_v_loss / (batch_idx + 1):.4f}")


# ==========================================
# EVALUATION & PLOTTING
# ==========================================
def evaluate_and_plot(model, data_loader, device, num_batches=10):
    print("Evaluating model distributions...")
    model.eval()
    true_actions, pred_actions, true_bets, pred_bets = [], [], [], []

    with torch.no_grad():
        if IS_RECURRENT:
            for i, (states_dict, targets, rewards, mask) in enumerate(data_loader):
                if i >= num_batches: break
                states_dict = {k: v.to(device) for k, v in states_dict.items()}

                h_p = torch.zeros(targets.size(0), model.hand_memory_size, device=device)
                g_p = torch.zeros(targets.size(0), model.game_memory_size, device=device)
                dist = _unroll_policy_bc(model, states_dict, h_p, g_p)

                samples = dist.sample().cpu().numpy()
                targets_np = targets.cpu().numpy()
                mask_np = mask.cpu().numpy().astype(bool)

                # Flatten based on mask to drop padding zeros
                valid_targets = targets_np[mask_np]
                valid_samples = samples[mask_np]

                true_actions.extend(valid_targets[:, 0])
                pred_actions.extend(valid_samples[:, 0])
                is_raise = valid_targets[:, 0] >= Action.get_raise_threshold()
                true_bets.extend(valid_targets[is_raise, 1])
                pred_bets.extend(valid_samples[is_raise, 1])
        else:
            for i, (states_dict, targets, rewards) in enumerate(data_loader):
                if i >= num_batches: break
                states_dict = {k: v.to(device) for k, v in states_dict.items()}

                dist = model(states_dict)
                samples = dist.sample().cpu().numpy()
                targets_np = targets.cpu().numpy()

                true_actions.extend(targets_np[:, 0])
                pred_actions.extend(samples[:, 0])
                is_raise = targets_np[:, 0] >= Action.get_raise_threshold()
                true_bets.extend(targets_np[is_raise, 1])
                pred_bets.extend(samples[is_raise, 1])

    def categorize_action(val):
        if val < Action.get_call_threshold():
            return "Fold"
        elif val < Action.get_raise_threshold():
            return "Call"
        else:
            return "Raise"

    true_act_labels = [categorize_action(a) for a in true_actions]
    pred_act_labels = [categorize_action(a) for a in pred_actions]

    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    sns.histplot(x=true_act_labels, color="blue", alpha=0.5, label="Baseline Bot (Target)", ax=axes[0],
                 stat="proportion", discrete=True, shrink=0.8)
    sns.histplot(x=pred_act_labels, color="orange", alpha=0.5, label="Trained Model", ax=axes[0], stat="proportion",
                 discrete=True, shrink=0.8)
    axes[0].set_title("Action Decision Distribution")
    axes[0].set_ylabel("Proportion of Actions")
    axes[0].legend()

    sns.kdeplot(true_bets, color="blue", fill=True, alpha=0.3, label="Baseline Bot", ax=axes[1], clip=(0, 1))
    sns.kdeplot(pred_bets, color="orange", fill=True, alpha=0.3, label="Trained Model", ax=axes[1], clip=(0, 1))
    axes[1].set_title("Bet Sizing Distribution (When Raising)")
    axes[1].set_xlabel("Scaled Bet Size [0, 1]")
    axes[1].set_ylabel("Density")
    axes[1].legend()

    plt.tight_layout()
    plt.savefig("bc_distribution_comparison.png", dpi=300)
    print("Saved distribution plot to 'bc_distribution_comparison.png'")

# ==========================================
# MAIN EXECUTION
# ==========================================
if __name__ == '__main__':
    print(f"Initializing training on device: {device}")

    dataset = PokerBehaviorCloningDataset(data_dir=data_folder)

    if IS_RECURRENT:
        # rnn_batch_size = max(1, batch_size // 10)
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, collate_fn=rnn_collate_fn)
        model, value_model = RNNPPO.init_networks(device, discrete=False, mode="beta")
    else:
        train_loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
        model = load_model(player_id=0, device=device, deterministic=False, mode="beta")
        value_model = get_value_model(device=device)

    if IS_RECURRENT:
        print("Freezing game_gru parameters to prevent Learned Amnesia during BC...")
        # Freeze policy game memory
        for param in model.game_gru.parameters():
            param.requires_grad = False

        # Freeze value game memory
        for param in value_model.game_gru.parameters():
            param.requires_grad = False

    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    value_optimizer = torch.optim.Adam(value_model.parameters(), lr=value_lr)

    print("Starting Behavior Cloning...")
    for epoch in range(1, num_epochs + 1):
        if IS_RECURRENT:
            train_rnn_epoch(model, value_model, optimizer, value_optimizer, train_loader, epoch)
        else:
            train_flat_epoch(model, value_model, optimizer, value_optimizer, train_loader, epoch)

    print("Pre-training complete! Ready for Self-Play.")
    torch.save([model.state_dict(), value_model.state_dict()], f"bc_pretrained_model_no_log_{'rnn' if IS_RECURRENT else 'no_mem'}.pt")

    evaluate_and_plot(model, train_loader, device, num_batches=10)