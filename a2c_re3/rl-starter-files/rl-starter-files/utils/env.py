import gym
import gym_minigrid

from gym_minigrid.wrappers import *
def make_env(env_key, seed=None):
    env = gym.make(env_key)
    # env = RGBImgPartialObsWrapper(env) # Get pixel observations
    # env = ImgObsWrapper(env) # Get rid of the 'mission' field
    # obs = env.reset() # This now produces an RGB tensor only
    env.seed(seed)
    return env

