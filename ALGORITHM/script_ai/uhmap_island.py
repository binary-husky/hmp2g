from cmath import pi
import numpy as np
from MISSION.uhmap.actionset_v3 import strActionToDigits, ActDigitLen
from config import GlobalConfig

class DummyAlgConfig():
    reserve = ""

class DummyAlgorithmBase():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.team = team
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        self.attack_order = {}
        self.team_agent_uid = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[team]
        self.is_reaching_height = np.zeros(self.n_agent)

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
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}

class DummyAlgorithmIdle(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')

        n_active_thread = sum(ENV_ACTIVE)

        # assert len(State_Recall['Latest-Obs']) == n_active_thread, ('make sure we have the right batch of obs')

        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen))
        


        for thread in range(self.n_thread):
            if ENV_PAUSE[thread]: 
                # 如果,该线程停止，不做任何处理
                continue

            # 此处代码仅做demo用
            # 指挥部坐标1：(66270,0,2550)  坐标2：(43040,102630,2540)
            # 注意：0°对应x轴正方向,90°对应y轴正方向
            # 先改变方向，后改变高度，之后加速前往，快接近目标后降低高度，发射导弹（Version1）

            fire_range = 2000
            has_action = np.zeros(self.n_agent)
            for id in range(self.n_agent-5):
                if id < self.n_agent/2:
                    target_location = np.array((66270,0,2550))
                else:
                    target_location = np.array((43040,102630,2540))
                cruise_height = 15000
                cruise_speed = 600
                plane_location = State_Recall['Latest-Team-Info'][thread]['dataArr'][id]['agentLocationArr']
                plane_rotaion = State_Recall['Latest-Team-Info'][thread]['dataArr'][id]['agentRotationArr']
                # 1.先判断方向是否正确
                delta_location =  target_location - plane_location
                target_pitch = self.DeltaLocation2Angle(delta_location[0], delta_location[1]) 
                if np.abs(target_pitch - plane_rotaion[0]) > 1 and has_action[id] == 0:
                    actions[thread, id] = strActionToDigits('ActionSet3::ChangeDirection;{}'.format(target_pitch))
                    # print("CHange Direction")
                    has_action[id] = 1
                # 2.升高至巡航高度(非并行编码！)(注意，在上升高度的时候需要判断当前是否处于俯仰角姿态)
                if  np.abs(cruise_height - plane_location[2]) > 50 and has_action[id] == 0:
                    actions[thread, id] = strActionToDigits('ActionSet3::ChangeHeight;{}'.format(cruise_height))
                    # print("CHange Height")
                    has_action[id] = 1
                # 3.提速至巡航速度
                if np.abs(cruise_speed - plane_location[2]) > 0.001 and has_action[id] == 0:
                    actions[thread, id] = strActionToDigits('ActionSet3::ChangeSpeed;Positive')
                    # print("CHange Speed")
                    has_action[id] = 1
                # 4.接近目标，对目标发射导弹
                # dist_2D = np.sqrt(np.sum(np.square((delta_location[0], delta_location[1]))))
                # if dist_2D < 10000:
                #     actions[thread, id] = strActionToDigits('ActionSet3::LaunchMissile;NONE')


            # print(target_pitch)
            # actions[thread, :] = strActionToDigits('ActionSet3::ChangeDirection;{}'.format(target_pitch))
            # print(State_Recall['Latest-Team-Info'][thread]['dataArr'][0]['agentRotationArr']) 


        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}
    def DeltaLocation2Angle(self, delta_x, delta_y):
        # 此处为角度制
        # assert len(delta_location) == 2 or 3
        # delta_x = delta_location[0]
        # delta_y = delta_location[1]
        if delta_x == 0 and delta_y != 0:
            theta = 90 if delta_y > 0 else -90
        else:
            abs_theta = np.arctan(np.abs(delta_y) / np.abs(delta_x)) * 180 / pi
            if delta_x > 0 and delta_y >= 0:
                theta = abs_theta
            elif delta_x < 0 and delta_y >= 0:
                theta = 180 - abs_theta
            elif delta_x > 0 and delta_y < 0:
                theta = - abs_theta
            elif delta_x < 0 and delta_y < 0:    
                theta = abs_theta - 180
        return theta
        


class DummyAlgorithmIdleTarget(DummyAlgorithmBase):
    def interact_with_env(self, State_Recall):
        assert State_Recall['Latest-Obs'] is not None, ('make sure obs is ok')
        ENV_PAUSE = State_Recall['ENV-PAUSE']
        ENV_ACTIVE = ~ENV_PAUSE

        assert self.n_thread == len(ENV_ACTIVE), ('the number of thread is wrong?')

        n_active_thread = sum(ENV_ACTIVE)

        actions = np.zeros(shape=(self.n_thread, self.n_agent, ActDigitLen))
        
        for thread in range(self.n_thread):
            if ENV_PAUSE[thread]: 
                # 如果,该线程停止，不做任何处理
                continue
            actions[thread, :] = strActionToDigits('ActionSet3::N/A;N/A')

        # set actions of in-active threads to NaN (will be done again in multi_team.py, this line is not necessary)
        actions[ENV_PAUSE] = np.nan
        # swap (self.n_thread, self.n_agent) -> (self.n_agent, self.n_thread) 
        actions = np.swapaxes(actions, 0, 1)
        return actions, {}
    def DeltaLocation2Angle(self, delta_x, delta_y):
        # 此处为角度制
        # assert len(delta_location) == 2 or 3
        # delta_x = delta_location[0]
        # delta_y = delta_location[1]
        if delta_x == 0 and delta_y != 0:
            theta = 90 if delta_y > 0 else -90
        else:
            abs_theta = np.arctan(np.abs(delta_y) / np.abs(delta_x)) * 180 / pi
            if delta_x > 0 and delta_y >= 0:
                theta = abs_theta
            elif delta_x < 0 and delta_y >= 0:
                theta = 180 - abs_theta
            elif delta_x > 0 and delta_y < 0:
                theta = - abs_theta
            elif delta_x < 0 and delta_y < 0:    
                theta = abs_theta - 180
        return theta
        




