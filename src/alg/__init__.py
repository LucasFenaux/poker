from .alg import BaseAlgorithm, OnPolicyAlgorithm
from .ppo import PPO, PPOInferenceWrapper
from .rnn_ppo import RNNPPO, RNNPPOInferenceWrapper
from .neurd import NeuRD, NeuRDInferenceWrapper
from src.global_settings import ALG, IS_RECURRENT


def get_alg_class():
    if ALG == "PPO" and IS_RECURRENT:
        return RNNPPO
    elif ALG == "PPO" and not IS_RECURRENT:
        return PPO
    elif ALG == "NEURD":
        return NeuRD
    else:
        raise NotImplementedError


def get_inference_wrapper_class():
    if ALG == "PPO" and IS_RECURRENT:
        return RNNPPOInferenceWrapper
    elif ALG == "PPO" and not IS_RECURRENT:
        return PPOInferenceWrapper
    elif ALG == "NEURD":
        return NeuRDInferenceWrapper
    else:
        raise NotImplementedError