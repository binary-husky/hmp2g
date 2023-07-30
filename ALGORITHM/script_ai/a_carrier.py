import numpy as np
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__
from config import GlobalConfig
from MISSION.uhmap.actionset_v3 import strActionToDigits, ActDigitLen
from MISSION.uhmap.actset_lookup import encode_action_as_digits

class AlgorithmConfig:
    preserve = ''

class DummyAlgorithmBase():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        self.attack_order = {}
        self.team = team

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
        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen))

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan

        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = actions if GlobalConfig.mt_act_order == 'new_method' else np.swapaxes(actions, 0, 1)
        return actions, {}

class AgentsWithCarrier(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')

        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')

        n_active_thread = sum(ENV_ACTIVE)
        AirCarrierUID = 2

        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen ))

        for thread in range(self.n_thread):
            if ENV_PAUSE[thread]: 
                # 如果,该线程停止，不做任何处理
                continue

            self_team_carriers = [a for a in State_Recall['Latest-Team-Info'][thread]['dataArr']            if a['agentAlive'] and a['agentTeam']==self.team and a['type']=='Carrier']
            opp_team_noncarroer_agents = [a for a in State_Recall['Latest-Team-Info'][thread]['dataArr']    if a['agentAlive'] and a['agentTeam']==(self.team) and a['type']!='Carrier']
            opp_team_noncarroer_agents_tid = np.array([a['indexInTeam'] for a in opp_team_noncarroer_agents])

            opp_team_carriers = [a for a in State_Recall['Latest-Team-Info'][thread]['dataArr']             if a['agentAlive'] and a['agentTeam']==(1-self.team) and a['type']=='Carrier']
            opp_team_agents = [a for a in State_Recall['Latest-Team-Info'][thread]['dataArr']               if a['agentAlive'] and a['agentTeam']==(1-self.team)]

            if len(opp_team_carriers) > 0:
                # attack enemy carrier if there is any
                a = opp_team_carriers[0]
            else:
                # otherwise attack enemy agent
                a = opp_team_agents[0]
                
            p = a['agentLocationArr']
            actions[thread, :] = strActionToDigits('ActionSet1::PatrolMoving;X=%f Y=%f Z=%f'%(p[0],p[1],p[2]))
            for x in self_team_carriers:
                actions[thread, x['indexInTeam']] = strActionToDigits('ActionSet2::SpecificAttacking;T%d-%d'%(1-self.team, a['indexInTeam']))

            if State_Recall['Current-Obs-Step'] == 0:
                actions[thread, :] = strActionToDigits('ActionSet1::Idle;AggressivePersue')

            for a in range(self.n_agent):
                if (a//2) == State_Recall['Current-Obs-Step'][thread]//5:
                    actions[thread, a] = strActionToDigits('ActionSet1::Special;Detach')

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = actions if GlobalConfig.mt_act_order == 'new_method' else np.swapaxes(actions, 0, 1)
        return actions, {}