import os
import sys
import glob
import torch
# Ensure we can import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.state_interpreter import StateSnapshot
from src.alg import PPO, RNNPPO, PPOInferenceWrapper, RNNPPOInferenceWrapper
from src.global_settings import IS_RECURRENT

from src.vanilla_cfr.kuhn_poker_solve import Node, Player, cfr_traversal

_RESET = "\033[0m"

def color_prob(ppo_prob: float, cfr_prob: float) -> str:
    """Return an ANSI-colored string for ppo_prob relative to cfr_prob.
    - Above CFR  -> white-to-vivid-green gradient (brighter = further above)
    - Below CFR  -> white-to-vivid-red   gradient (brighter = further below)
    - Very close -> default terminal colour (within 0.5 pp)
    Uses 24-bit true-color escape codes: \\033[38;2;R;G;Bm
    """
    diff = ppo_prob - cfr_prob          # positive = above CFR
    label = f"{ppo_prob*100:>6.2f}%"

    if abs(diff) < 0.005:               # within 0.5 pp -> plain white
        return label

    # t: 0.0 (barely off) -> 1.0 (50 pp or more off)
    t = min(1.0, abs(diff) / 0.50)

    if diff > 0:                        # above CFR -> green
        # (255, 255, 255) -> (80, 255, 80)
        r = int(255 - t * (255 - 80))
        g = 255
        b = int(255 - t * (255 - 80))
    else:                               # below CFR -> red
        # (255, 255, 255) -> (255, 80, 80)
        r = 255
        g = int(255 - t * (255 - 80))
        b = int(255 - t * (255 - 80))

    return f"\033[38;2;{r};{g};{b}m{label}{_RESET}"

def load_eval_models(model_path, device):
    if IS_RECURRENT:
        policy_net, value_net = RNNPPO.init_networks(device, mode="beta", discrete=False)
        wrapper = RNNPPOInferenceWrapper((policy_net,), discrete=False)
    else:
        policy_net, value_net = PPO.init_networks(device, mode="beta", discrete=False)
        wrapper = PPOInferenceWrapper((policy_net,), discrete=False)

    loaded_data = torch.load(model_path, map_location=device, weights_only=True)

    version = 1
    if isinstance(loaded_data, tuple) and len(loaded_data) == 3:
        checkpoint = loaded_data[0]
        version = loaded_data[2]
    elif isinstance(loaded_data, tuple) and len(loaded_data) == 2:
        checkpoint = loaded_data[0]
    else:
        checkpoint = loaded_data
        
    if isinstance(version, torch.Tensor):
        version = version.item()
    try:
        version = max(1, int(version))
    except (ValueError, TypeError):
        version = 1
        
    if version == 1 and "_" in os.path.basename(model_path):
        try:
            version = max(1, int(os.path.basename(model_path).split('_')[1].split('.')[0]))
        except ValueError:
            pass

    if IS_RECURRENT:
        wrapper.load_params(checkpoint)
    else:
        wrapper.load_params((checkpoint[0],))
    wrapper.to(device)

    value_net.load_state_dict(checkpoint[1])
    value_net.to(device)
    value_net.eval()

    return wrapper, value_net, version

def train_cfr(iterations=100000):
    all_info_states_player_one = Node.generate_all_info_states(player=1)
    all_info_states_player_two = Node.generate_all_info_states(player=2)

    player_one = Player(player=1, all_info_states=all_info_states_player_one)
    player_two = Player(player=2, all_info_states=all_info_states_player_two)

    for i in range(iterations):
        for j in range(1, 3):
            root_node = Node()
            cfr_traversal(root_node, j, player_one, player_two, 1, 1, 1)

    return player_one, player_two

def infoset_to_snapshot(infoset_str, player):
    card_char = infoset_str[0]
    actions = infoset_str[1:]
    
    card_map = {'j': "Js", 'q': "Qs", 'k': "Ks"}
    hole_cards = card_map[card_char]
    
    bets = [0.0, 0.0]
    actor = 0
    for a in actions:
        if a == 'h':
            pass
        elif a == 'b':
            bets[actor] = 1.0
        elif a == 'c':
            bets[actor] = bets[1 - actor]
        actor = 1 - actor
        
    assert actor == player - 1
    
    snapshot = StateSnapshot(
        hole_cards=hole_cards,
        board_cards="??????????",
        player_count=2,
        blinds_or_straddles=(1, 1),
        bets=bets,
        stacks=[10.0, 10.0],
        in_hand=[True, True],
        pots=[2.0],
        min_bet=1.0,
        max_bet=1.0
    )
    return snapshot, actor

def get_latest_run_folder():
    runs = glob.glob(os.path.join(project_root, "results", "run_KUHN_*"))
    if not runs:
        return None
    runs.sort(key=os.path.getmtime)
    return runs[-1]

def evaluate():
    device = torch.device("cpu")
    run_folder = get_latest_run_folder()
    if not run_folder:
        print("No KUHN run folder found in results/")
        return
        
    players_dir = os.path.join(run_folder, "players")
    historical_dir = os.path.join(run_folder, "historical_checkpoints")
    
    current_model_files = glob.glob(os.path.join(players_dir, "*.pt"))
    historical_model_files = glob.glob(os.path.join(historical_dir, "*.pt"))
    
    model_files = current_model_files + historical_model_files
    
    if not model_files:
        print("No .pt files found in players or historical_checkpoints dir.")
        return
        
    print(f"Found {len(current_model_files)} current models and {len(historical_model_files)} historical models in {run_folder}")
    
    # print("\nComputing distances between networks...")
    # def get_flat_weights(wrapper):
    #     return torch.cat([p.view(-1) for p in wrapper.network.parameters() if p.requires_grad])
    #
    # if IS_RECURRENT:
    #     dummy_policy, _ = RNNPPO.init_networks(device, mode="beta", discrete=False)
    #     dummy_wrapper = RNNPPOInferenceWrapper((dummy_policy,), discrete=False)
    # else:
    #     dummy_policy, _ = PPO.init_networks(device, mode="beta", discrete=False)
    #     dummy_wrapper = PPOInferenceWrapper((dummy_policy,), discrete=False)
    #
    # dummy_weights = get_flat_weights(dummy_wrapper)
    #
    # model_weights_dict = {}
    # for model_path in model_files:
    #     is_historical = "historical_checkpoints" in model_path
    #     filename = os.path.basename(model_path).replace(".pt", "")
    #     name = f"Hist_{filename}" if is_historical else f"Curr_{filename}"
    #
    #     wrapper, _ = load_eval_models(model_path, device)
    #     model_weights_dict[name] = get_flat_weights(wrapper)
    #
    # dummy_distances = []
    # for name, w in model_weights_dict.items():
    #     dist = torch.norm(dummy_weights - w).item()
    #     dummy_distances.append(dist)
    #
    # print(f"Avg L2 Distance (Dummy vs Trained): {sum(dummy_distances)/len(dummy_distances):.4f} (Min: {min(dummy_distances):.4f}, Max: {max(dummy_distances):.4f})")
    #
    # pairwise_distances = []
    # names = list(model_weights_dict.keys())
    # for i in range(len(names)):
    #     for j in range(i + 1, len(names)):
    #         dist = torch.norm(model_weights_dict[names[i]] - model_weights_dict[names[j]]).item()
    #         pairwise_distances.append(dist)
    #
    # if pairwise_distances:
    #     print(f"Avg L2 Distance (Trained vs Trained): {sum(pairwise_distances)/len(pairwise_distances):.4f} (Min: {min(pairwise_distances):.4f}, Max: {max(pairwise_distances):.4f})")
    # else:
    #     print("Not enough models for Pairwise Trained distances.")
        
    print("=========================================\n")
    
    print("Training CFR bot for 100,000 iterations...")
    cfr_p1, cfr_p2 = train_cfr(100_000)
    
    model_evaluations = []
    from src.global_settings import GAME_TYPE
    from src.action_interpreter import Action

    for model_path in model_files:
        policy_net, value_net, version = load_eval_models(model_path, device)
        is_current = "historical_checkpoints" not in model_path
        model_id = os.path.basename(model_path).replace(".pt", "")
        
        model_probs = {}
        for player_obj in [cfr_p1, cfr_p2]:
            strategy = player_obj.get_average_strategy_probs()
            
            for info_state in strategy.keys():
                snapshot, actor_index = infoset_to_snapshot(str(info_state), player_obj.order)
                
                with torch.no_grad():
                    if IS_RECURRENT:
                        (action_dist, bet_sizing_dist), _ = policy_net.get_model_policy(policy_net.network, (snapshot, actor_index))
                    else:
                        action_dist, bet_sizing_dist = policy_net.get_model_policy(policy_net.network, (snapshot, actor_index))
                    
                    probs = action_dist.probs.squeeze(0)
                    
                    if GAME_TYPE == "KUHN" or len(probs) == 2:
                        prob_raise = probs[1].item()
                        prob_check_fold = probs[0].item()
                    else:
                        prob_raise = probs[Action.RAISE.value].item()
                        prob_check_fold = probs[Action.CHECK_OR_FOLD.value].item() + probs[Action.CHECK_OR_CALL.value].item()
                
                model_probs[str(info_state)] = {"raise": prob_raise, "check_fold": prob_check_fold}
                
        model_evaluations.append((is_current, version, model_id, model_probs))

    specific_model_id = "46"  # Set this to any current model ID to add a dedicated column for it

    weighting_schemes = {
        "Equal": lambda is_curr, ver, m_id: 1.0,
        "Linear": lambda is_curr, ver, m_id: float(ver),
        "Quad": lambda is_curr, ver, m_id: float(ver) ** 2,
        "Current": lambda is_curr, ver, m_id: 1.0 if is_curr else 0.0,
    }
    
    if specific_model_id is not None:
        weighting_schemes[f"P{specific_model_id}"] = lambda is_curr, ver, m_id: 1.0 if is_curr and m_id == specific_model_id else 0.0

    scheme_probs = {}
    for scheme_name, weight_fn in weighting_schemes.items():
        total_weight = 0.0
        agg = {}
        for is_current, version, model_id, m_probs in model_evaluations:
            w = weight_fn(is_current, version, model_id)
            if w <= 0:
                continue
            total_weight += w
            for info_state, probs_dict in m_probs.items():
                if info_state not in agg:
                    agg[info_state] = {"raise": 0.0, "check_fold": 0.0}
                agg[info_state]["raise"] += probs_dict["raise"] * w
                agg[info_state]["check_fold"] += probs_dict["check_fold"] * w
        
        if total_weight > 0:
            for info_state in agg:
                agg[info_state]["raise"] /= total_weight
                agg[info_state]["check_fold"] /= total_weight
        scheme_probs[scheme_name] = agg

    num_models = len(model_evaluations)
    print("\nWeighting Schemes:")
    print("  - Equal:   All models (current and historical) weighted equally (w = 1)")
    print("  - Linear:  Weighted proportionally to training step / recency (w = version)")
    print("  - Quad:    Weighted proportionally to squared training step (w = version²)")
    print("  - Current: Only current pool models (historical checkpoints excluded)")
    if specific_model_id is not None:
        print(f"  - P{specific_model_id}:      Only evaluates current model ID {specific_model_id}")

    header = f"{'InfoSet':<8} | {'Action':<10} | {'CFR':<8}"
    for scheme_name in weighting_schemes:
        header += f" | {scheme_name[:8]:<8}"
        
    divider = "-" * len(header)

    print("\n" + "=" * len(header))
    print(f"{'DECISION MATRIX COMPARISON':^{len(header)}}")
    print(f"{f'(Evaluated on {num_models} models)':^{len(header)}}")
    print("=" * len(header))
    print(header)
    print(divider)
    
    mae_totals = {scheme: 0.0 for scheme in weighting_schemes}
    num_decisions = 0

    for player_obj in [cfr_p1, cfr_p2]:
        strategy = player_obj.get_average_strategy_probs()
        
        for info_state in sorted(list(strategy.keys()), key=lambda x: str(x)):
            cfr_probs = strategy[info_state]
            
            for action_str, cfr_prob in cfr_probs.items():
                if action_str in ('b', 'c'):
                    action_name = "Bet/Call"
                    prob_key = "raise"
                else:
                    action_name = "Check/Fold"
                    prob_key = "check_fold"
                
                row_str = f"{str(info_state):<8} | {action_name:<10} | {cfr_prob*100:>6.2f}%"
                for scheme_name in weighting_schemes:
                    ppo_prob = scheme_probs[scheme_name][str(info_state)][prob_key]
                    colored = color_prob(ppo_prob, cfr_prob)
                    row_str += f" | {colored}"
                    mae_totals[scheme_name] += abs(cfr_prob - ppo_prob)
                
                num_decisions += 1
                print(row_str)
        print(divider)

    if num_decisions > 0:
        mae_str = f"{'MAE vs':<8} | {'CFR':<10} | {'0.00%':<8}"
        for scheme_name in weighting_schemes:
            mae_val = (mae_totals[scheme_name] / num_decisions) * 100
            mae_str += f" | {mae_val:>6.2f}%"
        print(mae_str)
        print("=" * len(header) + "\n")

if __name__ == "__main__":
    evaluate()
