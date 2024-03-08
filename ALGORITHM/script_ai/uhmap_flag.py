import copy
import numpy as np
from UTIL.tensor_ops import distance_mat_between
from scipy.optimize import linear_sum_assignment
from MISSION.uhmap.actset_lookup import encode_action_as_digits
from MISSION.uhmap.actionset_v3 import strActionToDigits, ActDigitLen
from config import GlobalConfig

class DummyAlgConfig():
    reserve = ""
    yield_step = 0

class DummyAlgorithmBase():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.team = team
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        self.attack_order = {}
        self.team_agent_uid = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[team]

    def forward(self, inp, state, mask=None):
        raise NotImplementedError

    def to(self, device):
        return self
    
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')
        
        n_active_thread = sum(ENV_ACTIVE)

        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, 8))

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = actions if GlobalConfig.mt_act_order == 'new_method' else np.swapaxes(actions, 0, 1)
        return actions, {}





class DummyAlgorithmCaptureFlag(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')

        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE
        RST = State_Recall['Env-Suffered-Reset']


        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')

        n_active_thread = sum(ENV_ACTIVE)
        AirCarrierUID = 2

        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen ))


        if not hasattr(self, 'yield_step'):
            self.yield_step = np.ceil(np.random.rand(self.n_thread) * DummyAlgConfig.yield_step * 2)
            self.yield_step = np.random.rand(self.n_thread)*0 + DummyAlgConfig.yield_step

        if all(RST):
            self.yield_step = np.ceil(np.random.rand(self.n_thread) * DummyAlgConfig.yield_step * 2)
            self.yield_step = np.random.rand(self.n_thread)*0 + DummyAlgConfig.yield_step


        for thread in range(self.n_thread):
            if ENV_PAUSE[thread]: 
                # 如果,该线程停止，不做任何处理
                continue

            if State_Recall['Current-Obs-Step'][thread] < self.yield_step[thread]:
                actions[thread, :] = strActionToDigits(f"ActionSet1::Idle;Null")
                continue

            # get the location of the flag
            info = State_Recall['Latest-Team-Info'][thread]
            flag = [obj for obj in info['dataGlobal']['keyObjArr'] if obj['className'] == "FlagToCapture_C"]
            assert len(flag) == 1
            flag = flag[0]
            flag_location = flag['location']
            flag_location_arr = (flag_location['x'], flag_location['y'], flag_location['z'])    # (716.6385, 3901.616, 157.034)

            # get alive agents 
            alive_agents        = [a for a in info['dataArr'] if a['agentTeam']==self.team and a['agentAlive']]
            non_flag_holders    = [a for a in alive_agents if np.abs(flag_location_arr[0]-a['agentLocationArr'][0]) + np.abs(flag_location_arr[1]-a['agentLocationArr'][1]) > 50]
            if len(alive_agents) == len(non_flag_holders):
                non_flag_holders.pop(0)
            # current step 
            st = State_Recall['Current-Obs-Step'][thread]

            # go to capture the flag
            actions[thread, :] = strActionToDigits(f"ActionSet1::MoveToLocation;X={flag_location['x']} Y={flag_location['y']} Z={flag_location['z']}")
            for i, agent in enumerate(non_flag_holders):
                theta = (st / 40 + i / 7) * 3.14159 * 2
                offset = 2000
                offset_x = offset*np.sin(theta)
                offset_y = offset*np.cos(theta)
                actions[thread, agent['indexInTeam']] = strActionToDigits(f"ActionSet1::MoveToLocation;X={flag_location['x']+offset_x} Y={flag_location['y']+offset_y} Z={flag_location['z']}")
            


        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = actions if GlobalConfig.mt_act_order == 'new_method' else np.swapaxes(actions, 0, 1)
        return actions, {}








def vector_shift_towards(pos, toward_pos, offset):
    delta = toward_pos - pos 
    delta = delta / (np.linalg.norm(delta) + 1e-10)
    return pos + delta * offset
