import numpy as np
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__
from config import GlobalConfig

class AlgorithmConfig:
    preserve = ''

class RandomController(object):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.n_action = GlobalConfig.ScenarioConfig.n_actions

    def interact_with_env(self, StateRecall):
        obs = StateRecall['Latest-Obs']
        P = StateRecall['ENV-PAUSE']
        active_thread_obs = obs[~P]
        actions = np.random.randint(low=0,high=self.n_action, size=(self.n_thread, self.n_agent, 1))
        StateRecall['_hook_'] = None
        return actions, StateRecall 
