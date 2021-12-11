import gym
import logging
import cv2
#from gym_duckietown.simulator import Simulator

try:
    from gym_duckietown.simulator import Simulator
except:
    pass

logger = logging.getLogger(__name__)
import numpy as np


class DTPytorchWrapper:
    def __init__(self, shape=(480, 640, 3)):
        self.shape = shape
        self.transposed_shape = (shape[2], shape[0], shape[1])

    def preprocess(self, obs):
        # from PIL import Image
        # return np.array(Image.fromarray(obs).resize(self.shape[0:2])).transpose(2, 0, 1)
        obs = cv2.resize(obs, self.shape[0:2])
        # NOTICE: OpenCV changes the order of the channels !!!
        obs = cv2.cvtColor(obs, cv2.COLOR_BGR2RGB)
        return obs


class InconvenientSpawnFixingWrapper(gym.Wrapper):
    """
    Fixes the "Exception: Could not find a valid starting pose after 5000 attempts" in duckietown-gym-daffy 5.0.13
    The robot is first placed in a random drivable tile, than a configuration is sampled on this tile. If the later
    step fails, it is repeated (up to 5000 times). If a tile has too many obstacles on it, it might not have any
    convenient (collision risking) configurations, so another tile should be selected. Instead of selecting a new tile,
    Duckietown gym just raises the above exception.
    This wrapper calls reset() again and again if a new tile has to be sampled.
    .. note::
        ``gym_duckietown.simulator.Simulator.reset()`` is called in ``gym_duckietown.simulator.Simulator.__init__(...)``.
        **Simulator instantiation should also be wrapped in a similar while loop!!!**
    """

    def reset(self, **kwargs):
        spawn_successful = False
        spawn_attempts = 1
        while not spawn_successful:
            try:
                logger.debug(self.unwrapped.user_tile_start)
                logger.debug("+" * 1000)
                print(self.unwrapped.seed_value)
                ret = self.env.reset(**kwargs)
                spawn_successful = True
            except Exception as e:
                self.unwrapped.seed_value += 1  # Otherwise it selects the same tile in the next attempt
                logger.debug(self.unwrapped.user_tile_start)
                self.unwrapped.seed(self.unwrapped.seed_value)
                logger.error("{}; Retrying with new seed: {}".format(e, self.unwrapped.seed_value))
                spawn_attempts += 1
        logger.debug("Reset and Spawn successful after {} attempts".format(spawn_attempts))
        return ret


class DummyDuckietownGymLikeEnv(gym.Env):
    def __init__(self):
        self.observation_space = gym.spaces.Box(
            low=0,
            high=255,
            shape=(480, 640, 3),
            dtype=np.uint8
        )
        self.action_space = gym.spaces.Box(
            low=-1,
            high=1,
            shape=(2,),
            dtype=np.float32
        )
        self.road_tile_size = 0.585

    def reset(self):
        logger.warning("Dummy Duckietown Gym reset() called!")
        return np.zeros((480, 640, 3), dtype=np.uint8)

    def step(self, action):
        logger.warning("Dummy Duckietown Gym step() called!")
        obs = np.zeros((480, 640, 3))
        reward = 0.0
        done = True
        info = {}
        return obs, reward, done, info


def get_wrappers(wrapped_env):
    obs_wrappers = []
    action_wrappers = []
    reward_wrappers = []
    orig_env = wrapped_env
    while not isinstance(orig_env, DummyDuckietownGymLikeEnv):# and not isinstance(orig_env, Simulator):
        if isinstance(orig_env, gym.ObservationWrapper):
            obs_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.ActionWrapper):
            action_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.RewardWrapper):
            reward_wrappers.append(orig_env)
        elif isinstance(orig_env, gym.Wrapper):
            None
        else:
            print("Strange wrapper")
        #if not isinstance(orig_env, DummyDuckietownGymLikeEnv) and isinstance(orig_env, ):
        #    break
        orig_env = orig_env.env

    return obs_wrappers[::-1], action_wrappers[::-1], reward_wrappers[::-1]


def print_all_wrappers(dummy_env):
    obs_wrappers, action_wrappers, reward_wrappers = get_wrappers(dummy_env)
    print("Observation wrappers")
    print(*obs_wrappers, sep="\n")
    print("\nAction wrappers")
    print(*action_wrappers, sep="\n")
    print("\nReward wrappers")
    print(*reward_wrappers, sep="\n")
