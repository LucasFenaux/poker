import importlib
from pokerkit import NoLimitTexasHoldem, KuhnPoker, Automation
from src.global_settings import GAME_TYPE, IS_RECURRENT


def get_holdem_table_params(table_size, **kwargs):
    # kwargs can contain small_blind, big_blind, bb_starting_stacks
    small_blind = kwargs.get('small_blind', 1)
    big_blind = kwargs.get('big_blind', 2)
    bb_starting_stacks = kwargs.get('bb_starting_stacks', 100)
    starting_stacks = [bb_starting_stacks * big_blind] * table_size
    
    return {
        "raw_blinds_or_straddles": (small_blind, big_blind),
        "min_bet": big_blind,
        "raw_starting_stacks": starting_stacks,
        "player_count": table_size
    }

def get_kuhn_table_params(table_size, **kwargs):
    # kwargs can contain small_blind, big_blind, bb_starting_stacks
    small_blind = kwargs.get('small_blind', 1)
    big_blind = kwargs.get('big_blind', 2)
    bb_starting_stacks = kwargs.get('bb_starting_stacks', 100)
    starting_stacks = [bb_starting_stacks * big_blind] * table_size
    
    return {
        "raw_blinds_or_straddles": (small_blind, big_blind),
        "min_bet": big_blind,
        "raw_starting_stacks": starting_stacks,
        "player_count": table_size
    }

GAME_REGISTRY = {
    "HOLDEM": {
        "action_size": 2,
        "num_decisions": 3,
        "min_stack": 100,
        "max_stack": 100,
        "min_bb_ratio": 2,
        "max_bb_ratio": 2,
        "min_allowed_start_bb": 10,
        # "action_interpreter_path": "src.action_interpreter.HoldemActionInterpreter",
        # "state_preprocessor_path": "src.state_interpreter.HoldemStatePreprocessor",
        # "state_interpreter_path": "src.state_interpreter.HoldemStateInterpreter",
        # "table_actor_path": "src.table_actor.holdem.HoldemTableActor",
        "pokerkit_game": NoLimitTexasHoldem,
        "pokerkit_automations": (
            Automation.ANTE_POSTING, Automation.BET_COLLECTION, Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING, Automation.HOLE_DEALING, Automation.BOARD_DEALING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
        ),
        "table_param_generator": get_holdem_table_params
    },
    "KUHN": {
        "action_size": 1,
        "num_decisions": 2,
        "min_stack": 10,
        "max_stack": 10,
        "min_bb_ratio": 1,
        "max_bb_ratio": 1,
        "min_allowed_start_bb": 1,
        # "action_interpreter_path": "src.action_interpreter.KuhnActionInterpreter",
        # "state_preprocessor_path": "src.state_interpreter.KuhnStatePreprocessor",
        # "state_interpreter_path": "src.state_interpreter.KuhnStateInterpreter",
        # "table_actor_path": "src.table_actor.kuhn.KuhnTableActor",
        "pokerkit_game": KuhnPoker,
        "pokerkit_automations": (
            Automation.ANTE_POSTING, Automation.BET_COLLECTION, Automation.BLIND_OR_STRADDLE_POSTING,
            Automation.CARD_BURNING, Automation.HOLE_DEALING, Automation.BOARD_DEALING,
            Automation.HOLE_CARDS_SHOWING_OR_MUCKING, Automation.HAND_KILLING,
            Automation.CHIPS_PUSHING, Automation.CHIPS_PULLING,
        ),
        "table_param_generator": get_kuhn_table_params
    }
}


HYPERPARAMETER_REGISTRY = {
    "HOLDEM": {
        False: {   # is recurrent
        },
        True: {
        }
    },
    "KUHN": {
        False: {  # is recurrent
        },
        True: {
            "entropy_coef": 1e-2
            # "lr": 1e-3,  # higher learning rate to boost learning
            # "value_lr": 5e-3,   # adjusting the value lr accordingly
        }
    }
}

def get_class_from_path(path_str):
    module_path, class_name = path_str.rsplit('.', 1)
    module = importlib.import_module(module_path)
    return getattr(module, class_name)

def get_current_game_config():
    if GAME_TYPE not in GAME_REGISTRY:
        raise ValueError(f"Unknown GAME_TYPE: {GAME_TYPE}. Please define it in GAME_REGISTRY.")
    
    config = GAME_REGISTRY[GAME_TYPE]
    
    from src.state_interpreter import get_state_preprocessor_class, get_state_interpreter_class
    from src.action_interpreter import get_action_interpreter_class
    from src.table_actor import get_table_actor_class

    return {
        "action_size": config["action_size"],
        "num_decisions": config["num_decisions"],
        # "action_interpreter": get_class_from_path(config["action_interpreter_path"]),
        "action_interpreter": get_action_interpreter_class(),
        # "state_preprocessor": get_class_from_path(config["state_preprocessor_path"]),
        "state_preprocessor": get_state_preprocessor_class(),
        # "state_interpreter": get_class_from_path(config["state_interpreter_path"]),
        "state_interpreter": get_state_interpreter_class(),
        # "table_actor": get_class_from_path(config["table_actor_path"]),
        "table_actor": get_table_actor_class(),
        "pokerkit_game": config["pokerkit_game"],
        "pokerkit_automations": config["pokerkit_automations"],
        "table_param_generator": config["table_param_generator"],
        "action_map": config["action_map"],
        "min_stack": config["min_stack"],
        "max_stack": config["max_stack"],
        "min_bb_ratio": config["min_bb_ratio"],
        "max_bb_ratio": config["max_bb_ratio"],
        "min_allowed_start_bb": config["min_allowed_start_bb"],
    }


def get_current_game_hyperparameters():
    if GAME_TYPE not in GAME_REGISTRY:
        raise ValueError(f"Unknown GAME_TYPE: {GAME_TYPE}. Please define it in GAME_REGISTRY.")

    hyperparameters = HYPERPARAMETER_REGISTRY[GAME_TYPE][IS_RECURRENT]
    return hyperparameters