from typing import Dict

import numpy as np
from ray.rllib import RolloutWorker, Policy, BaseEnv, SampleBatch
from ray.rllib.agents import DefaultCallbacks, MultiCallbacks


class MyCallbacks(DefaultCallbacks):
    
    def on_episode_start(self, *, worker: RolloutWorker, base_env: BaseEnv,
                        policies: Dict[str, Policy], episode,
                        env_index: int, **kwargs):
        print("work")



def get_callbacks():
    return MultiCallbacks([MyCallbacks])
