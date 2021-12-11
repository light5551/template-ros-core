from wrappers.action_wpappers import LeftRightBraking2WheelVelsWrapper
from wrappers.observe_wrappers import NormalizeWrapper
try:
    from pyglet.window import key
except Exception:
    pass
from ray import tune
from env import launch_env
from ray.tune import register_env
from ray.rllib.agents.ppo import ppo, PPOTrainer
from configs.api import load_config, get_rllib_config, env_config
from wrappers.general_wrappers import get_wrappers, print_all_wrappers

CONFIG_PATH = "./configs/main.yaml"


def registy(evaluate=False):#642-my
    checkpoint_path = "./models/checkpoint-391"

    ENV_NAME = 'Duckietown'
    register_env(ENV_NAME, launch_env)

    rllib_config = get_rllib_config(CONFIG_PATH)
    rllib_config.update({
        "env": ENV_NAME,
        "framework": "torch",
        "num_workers": 1,  # 32,
        "lr": 0.0001,
    })

    trainer = PPOTrainer(config=rllib_config)
    trainer.restore(checkpoint_path)
    return trainer


def get_env():
    return launch_env(env_config(CONFIG_PATH))

