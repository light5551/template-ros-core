from run import get_env
import torch 
import numpy as np
from wrappers.general_wrappers import DTPytorchWrapper, get_wrappers, print_all_wrappers
from wrappers.action_wpappers import Heading2WheelVelsWrapper, LeftRightBraking2WheelVelsWrapper

obs_wrappers = get_wrappers(get_env())

def solution(obs, model):
    #print(obs.shape)
    for idx, obs_wrap in enumerate(obs_wrappers):
        obs = obs_wrap.observation(obs)
    
    action = model.compute_single_action(obs, explore=False)
    action = np.clip(np.array([1 + action, 1 - action]), 0., 1.) * 0.6
    return action

    return [0.1, 1]
