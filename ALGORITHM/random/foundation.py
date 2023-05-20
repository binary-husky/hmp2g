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


class RandomControllerWithActionSetV2(object):

    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from .actionset import ActionConvertV2
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.actions_set = ActionConvertV2(
            SELF_TEAM_ASSUME=team, 
            OPP_TEAM_ASSUME=(1-team), 
            OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[1-team]
        )
        self.n_action = self.actions_set.n_act

    def interact_with_env(self, StateRecall):
        obs = StateRecall['Latest-Obs']
        P = StateRecall['ENV-PAUSE']

        active_thread_obs = obs[~P]
        actions = np.random.randint(low=0,high=self.n_action, size=(self.n_thread, self.n_agent, 1))

        act_converted = np.array(
                            list(map(lambda x: self.actions_set.convert_act_arr(None, x), 
                            actions.flatten()))).reshape(self.n_thread, self.n_agent, self.actions_set.ActDigitLen)

        StateRecall['_hook_'] = None
        return act_converted, StateRecall 


class RandomControllerWithActionSetV4(object):

    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from .actionset import ActionConvertMovingV4
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.actions_set = ActionConvertMovingV4(
            SELF_TEAM_ASSUME=team, 
            OPP_TEAM_ASSUME=(1-team), 
            OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[1-team]
        )
        self.n_action = self.actions_set.n_act

    def interact_with_env(self, StateRecall):
        obs = StateRecall['Latest-Obs']
        P = StateRecall['ENV-PAUSE']

        active_thread_obs = obs[~P]
        actions = np.random.randint(low=0,high=self.n_action, size=(self.n_thread, self.n_agent, 1))

        act_converted = np.array(
                            list(map(lambda x: self.actions_set.convert_act_arr(None, x), 
                            actions.flatten()))).reshape(self.n_thread, self.n_agent, self.actions_set.ActDigitLen)

        StateRecall['_hook_'] = None
        return act_converted, StateRecall 
