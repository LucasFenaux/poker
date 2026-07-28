import os
import sys
import glob
import torch
import random

# Ensure we can import from src
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(project_root)

from src.state_interpreter import StateSnapshot
from src.ppo_self_play.alg import PPO, RNNPPO, PPOInferenceWrapper, RNNPPOInferenceWrapper
from src.ppo_self_play.global_settings import IS_RECURRENT

from src.vanilla_cfr.kuhn_poker_solve import Node, Player, cfr_traversal

def load_eval_models(model_path, device):
    if IS_RECURRENT:
        policy_net, value_net = RNNPPO.init_networks(device, mode="beta", discrete=False)
        wrapper = RNNPPOInferenceWrapper((policy_net,), discrete=False)
    else:
        policy_net, value_net = PPO.init_networks(device, mode="beta", discrete=False)
        wrapper = PPOInferenceWrapper((policy_net,), discrete=False)

    loaded_data = torch.load(model_path, map_location=device, weights_only=True)

    if isinstance(loaded_data, tuple) and len(loaded_data) == 3:
        checkpoint = loaded_data[0]
    elif isinstance(loaded_data, tuple) and len(loaded_data) == 2:
        checkpoint = loaded_data[0]
    else:
        checkpoint = loaded_data
        
    if IS_RECURRENT:
        wrapper.load_params(checkpoint)
    else:
        wrapper.load_params((checkpoint[0],))
    wrapper.to(device)

    value_net.load_state_dict(checkpoint[1])
    value_net.to(device)
    value_net.eval()

    return wrapper, value_net

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
    
    ppo_probs_agg = {}
    from src.ppo_self_play.global_settings import GAME_TYPE
    from src.action_interpreter import Action

    for model_path in model_files:
        policy_net, value_net = load_eval_models(model_path, device)
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
                
                if str(info_state) not in ppo_probs_agg:
                    ppo_probs_agg[str(info_state)] = {"raise": 0.0, "check_fold": 0.0}
                ppo_probs_agg[str(info_state)]["raise"] += prob_raise
                ppo_probs_agg[str(info_state)]["check_fold"] += prob_check_fold

    num_models = len(model_files)
    for info_state in ppo_probs_agg:
        ppo_probs_agg[info_state]["raise"] /= num_models
        ppo_probs_agg[info_state]["check_fold"] /= num_models
    
    print("\n=========================================================================")
    print("                      DECISION MATRIX COMPARISON                         ")
    print(f"                       (Averaged over {num_models} models)                        ")
    print("=========================================================================")
    print(f"{'InfoSet':<10} | {'Action':<12} | {'CFR Prob':<10} | {'PPO Prob':<10}")
    print("-" * 73)
    
    for player_obj in [cfr_p1, cfr_p2]:
        strategy = player_obj.get_average_strategy_probs()
        
        for info_state in sorted(list(strategy.keys()), key=lambda x: str(x)):
            cfr_probs = strategy[info_state]
            prob_raise = ppo_probs_agg[str(info_state)]["raise"]
            prob_check_fold = ppo_probs_agg[str(info_state)]["check_fold"]
            
            for action_str, cfr_prob in cfr_probs.items():
                if action_str in ('b', 'c'):
                    ppo_prob = prob_raise
                    action_name = "Bet/Call"
                else:
                    ppo_prob = prob_check_fold
                    action_name = "Check/Fold"
                    
                print(f"{str(info_state):<10} | {action_name:<12} | {cfr_prob*100:>6.2f}%   | {ppo_prob*100:>6.2f}%")
        print("-" * 73)

if __name__ == "__main__":
    evaluate()
