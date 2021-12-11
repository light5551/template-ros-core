import logging
import gym
import numpy as np

logger = logging.getLogger(__name__)


class DtRewardWrapperDistanceTravelled(gym.RewardWrapper):
    def __init__(self, env):
        super(DtRewardWrapperDistanceTravelled, self).__init__(env)
        # gym_duckietown.simulator.Simulator):
        self.prev_pos = None

    def reward(self, reward):
        # Baseline reward is a for each step
        my_reward = 0

        # Get current position and store it for the next step
        pos = self.unwrapped.cur_pos
        prev_pos = self.prev_pos
        self.prev_pos = pos
        if prev_pos is None:
            return 0

        # Get the closest point on the curve at the current and previous position
        angle = self.unwrapped.cur_angle
        curve_point, tangent = self.unwrapped.closest_curve_point(pos, angle)
        prev_curve_point, prev_tangent = self.unwrapped.closest_curve_point(prev_pos, angle)
        if curve_point is None or prev_curve_point is None:
            logger.error("self.unwrapped.closest_curve_point(pos, angle) returned None!!!")
            return my_reward
        # Calculate the distance between these points (chord of the curve), curve length would be more accurate
        diff = curve_point - prev_curve_point
        dist = np.linalg.norm(diff)

        try:
            lp = self.unwrapped.get_lane_pos2(pos, self.unwrapped.cur_angle)
            # print("Dist: {:3.2f} | DotDir: {:3.2f} | Angle_deg: {:3.2f}".format(lp.dist, lp.dot_dir, lp.angle_deg))
        except Exception:
            return my_reward

        # Dist is negative on the left side of the rignt lane center and is -0.1 on the lane center.
        # The robot is 0.13 (m) wide, to keep the whole vehicle in the right lane, dist should be > -0.1+0.13/2)=0.035
        # 0.05 is a little less conservative
        if lp.dist < -0.05:
            return my_reward
        # Check if the agent moved in the correct direction
        if np.dot(tangent, diff) < 0:
            return my_reward

        # Reward is proportional to the distance travelled at each step
        my_reward = 50 * dist
        if np.isnan(my_reward):
            my_reward = 0.
            logger.error("Reward is nan!!!")
        return my_reward



class DtRewardClipperWrapper(gym.RewardWrapper):
    def __init__(self, env, clip_high=1000, clip_low=-100):
        super(DtRewardClipperWrapper, self).__init__(env)
        self.clip_high = clip_high
        self.clip_low = clip_low

    def reward(self, reward):
        if np.isnan(reward):
            reward = 0.
            logger.error("Reward is nan!!!")
        return np.clip(reward, self.clip_low, self.clip_high)


class DtRewardVelocity(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardVelocity, self).__init__(env)
        self.velocity_reward = 0.

    def reward(self, reward):
        self.velocity_reward = np.max(self.unwrapped.wheelVels) * 0.25
        if np.isnan(self.velocity_reward):
            self.velocity_reward = 0.
            logger.error("Velocity reward is nan, likely because the action was [nan, nan]!")
        return reward + self.velocity_reward

    def reset(self, **kwargs):
        self.velocity_reward = 0.
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['velocity'] = self.velocity_reward
        return observation, self.reward(reward), done, info


class DtRewardCollisionAvoidance(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardCollisionAvoidance, self).__init__(env)
            # gym_duckietown.simulator.Simulator
        self.prev_proximity_penalty = 0.
        self.proximity_reward = 0.

    def reward(self, reward):
        # Proximity reward is proportional to the change of proximity penalty. Range is ~ 0 - +1.5 (empirical)
        # Moving away from an obstacle is promoted, if the robot and the obstacle are close to each other.
        proximity_penalty = self.unwrapped.proximity_penalty2(self.unwrapped.cur_pos, self.unwrapped.cur_angle)
        self.proximity_reward = -(self.prev_proximity_penalty - proximity_penalty) * 50
        if self.proximity_reward < 0.:
            self.proximity_reward = 0.
        logger.debug("Proximity reward: {:.3f}".format(self.proximity_reward))
        self.prev_proximity_penalty = proximity_penalty
        return reward + self.proximity_reward

    def reset(self, **kwargs):
        self.prev_proximity_penalty = 0.
        self.proximity_reward = 0.
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['collision_avoidance'] = self.proximity_reward
        return observation, self.reward(reward), done, info


class DtRewardProximityPenalty(gym.RewardWrapper):
    def __init__(self, env):
        if env is not None:
            super(DtRewardProximityPenalty, self).__init__(env)
        self.proximity_reward = 0.

    def reward(self, reward):
        proximity_penalty = self.unwrapped.proximity_penalty2(self.unwrapped.cur_pos, self.unwrapped.cur_angle)
        self.proximity_reward = proximity_penalty * 2.5
        logger.debug("Proximity reward: {:.3f}".format(self.proximity_reward))
        return reward + self.proximity_reward

    def reset(self, **kwargs):
        self.proximity_reward = 0.
        return self.env.reset(**kwargs)

    def step(self, action):
        observation, reward, done, info = self.env.step(action)
        if 'custom_rewards' not in info.keys():
            info['custom_rewards'] = {}
        info['custom_rewards']['collision_avoidance'] = self.proximity_reward
        return observation, self.reward(reward), done, info

