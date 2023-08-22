import numpy as np
from UTIL.config_args import ChainVar

def usual_id_arrangment(N_AGENT_EACH_TEAM):
    """
        e.g., 
        input [5, 3]
        output [range(0,5), range(5,8)]
    """
    AGENT_ID_EACH_TEAM = []
    p = 0
    for team_agent_num in N_AGENT_EACH_TEAM:
        AGENT_ID_EACH_TEAM.append(range(p, p + team_agent_num))
        p += team_agent_num
    return AGENT_ID_EACH_TEAM

class ScenarioConfig(object):  
    '''
        ScenarioConfig: This config class will be 'injected' with new settings from JSONC.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (As the name indicated, ChainVars will change WITH vars it 'chained_with' during config injection)
        (please see UTIL.config_args to find out how this advanced trick works out.)
    '''
    # <Part 1> Needed by the hmp core #
    N_AGENT_EACH_TEAM = [1, ]
    AGENT_ID_EACH_TEAM = usual_id_arrangment(N_AGENT_EACH_TEAM)
    N_TEAM = len(N_AGENT_EACH_TEAM)

    # chained parameters, will change along with 'N_AGENT_EACH_TEAM'
    AGENT_ID_EACH_TEAM_cv = ChainVar(lambda N_AGENT_EACH_TEAM: usual_id_arrangment(N_AGENT_EACH_TEAM), chained_with=['N_AGENT_EACH_TEAM'])
    N_TEAM_cv = ChainVar(lambda N_AGENT_EACH_TEAM: len(N_AGENT_EACH_TEAM), chained_with=['N_AGENT_EACH_TEAM'])
    
    # algorithm selection
    TEAM_NAMES = ['ALGORITHM.None->None',]

    '''
        ## If the length of action array == the number of teams, set ActAsUnity to True
        ## If the length of action array == the number of agents, set ActAsUnity to False
    '''
    ActAsUnity = False

    '''
        ## If the length of reward array == the number of agents, set RewardAsUnity to False
        ## If the length of reward array == 1, set RewardAsUnity to True
    '''
    RewardAsUnity = True

    '''
        ## If the length of obs array == the number of agents, set ObsAsUnity to False
        ## If the length of obs array == the number of teams, set ObsAsUnity to True
    '''
    ObsAsUnity = False

    # <Part 2> Needed by env itself #
    MaxEpisodeStep = 100

from ..common.base_env import BaseEnv

class LLM_Trainer(BaseEnv):
    def __init__(self, rank) -> None:
        self.observation_space = None
        self.action_space = None
        self.rank = rank

    def step(self, act):
        obs = np.zeros(shape=(1, 1))
        reward = np.zeros(shape=(1))
        done = False
        info = {}
        return (obs, reward,  done, info)  # choose this if RewardAsUnity

    def reset(self):
        # obs: a Tensor with shape (n_agent, ...)
        # info: a dict
        obs = np.zeros(shape=(1, 1))
        info = {}
        return obs, info


def make_llm_env(env_id, rank):
    return LLM_Trainer(rank)
