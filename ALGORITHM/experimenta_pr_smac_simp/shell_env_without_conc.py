import numpy as np
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__, repeat_at, np_one_hot
from .cython_func import roll_hisory

class ShellEnvWrapper(object):
    def __init__(self, n_agent, n_thread, space, mcv, rl_functional, alg_config, ScenarioConfig):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.n_action = ScenarioConfig.n_actions
        self.space = space
        self.mcv = mcv
        self.rl_functional = rl_functional
        self.ScenarioConfig = ScenarioConfig
        self.alg_config = alg_config
        assert not self.ScenarioConfig.EntityOriented
        self.rawobs_dim = self.core_dim = space['obs_space']['obs_shape']
        if alg_config.shell_obs_add_id:
            self.core_dim = self.core_dim + self.n_agent
        if alg_config.shell_obs_add_previous_act:
            self.core_dim = self.core_dim + self.n_action

        # whether to use avail_act to block forbiden actions
        self.AvailActProvided = False
        if hasattr(self.ScenarioConfig, 'AvailActProvided'):
            self.AvailActProvided = self.ScenarioConfig.AvailActProvided 

        # whether to load previously saved checkpoint
        self.load_checkpoint = alg_config.load_checkpoint
        self.use_policy_resonance = alg_config.use_policy_resonance
        self.cold_start = True

    def interact_with_env(self, State_Recall):
        obs = State_Recall['Latest-Obs']
        alive = ~((obs==0).all(-1))
        P = State_Recall['ENV-PAUSE']
        RST = State_Recall['Env-Suffered-Reset']

        if RST.all():
            if self.use_policy_resonance: self.rl_functional.stage_planner.uprate_eprsn(self.n_thread)
            previous_act_onehot = np.zeros((self.n_thread, self.n_agent, self.n_action), dtype=float)
        else:
            previous_act_onehot = State_Recall['_Previous_Act_Onehot_'] # 利用State_Recall的回环特性，读取上次决策的状态

        # shell_obs_add_id
        if self.alg_config.shell_obs_add_id:
            obs = np.concatenate((obs, repeat_at(np.eye(self.n_agent,dtype=obs.dtype), insert_dim=0, n_times=obs.shape[0])), -1)
        if self.alg_config.shell_obs_add_previous_act:
            obs = np.concatenate((obs, previous_act_onehot), -1)
        obs[~alive] = np.nan

        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=np.int) - 1 # 初始化全部为 -1
        state = np.array([info['state'] for info in State_Recall['Latest-Team-Info']])

        state_feed = state[~P]
        alive_feed = alive[~P]
        obs_feed_in  = obs[~P] #, his_pool_next = self.solve_duplicate(obs_feed.copy(), his_pool_obs_feed.copy(), alive_feed.copy())
        eprsn = self.rl_functional.stage_planner.eprsn[~P] if self.use_policy_resonance else None
        randl = self.rl_functional.stage_planner.randl[~P] if self.use_policy_resonance else None

        I_State_Recall = {
            'obs':obs_feed_in,
            'alive':alive_feed,
            'state':state_feed,
            'randl':randl,
            'eprsn':eprsn,
            'Test-Flag':State_Recall['Test-Flag'],
            'threads_active_flag':~P,
            'Latest-Team-Info':State_Recall['Latest-Team-Info'][~P],
        }
        if self.AvailActProvided:
            avail_act = np.array([info['avail-act'] for info in np.array(State_Recall['Latest-Team-Info'][~P], dtype=object)])
            I_State_Recall.update({'avail_act':avail_act})

        act_active, internal_recall = self.rl_functional.interact_with_env_genuine(I_State_Recall)

        act[~P] = act_active
        actions_list = np.swapaxes(act, 0, 1) # swap thread(batch) axis and agent axis

        # return necessary handles to main platform
        if self.cold_start: self.cold_start = False

        # <2> call a empty frame to gather reward
        # State_Recall['_Previous_Obs_'] = obs
        State_Recall['_Previous_Act_Onehot_'] = np_one_hot(act, n=self.n_action)
        if not State_Recall['Test-Flag']:
            State_Recall['_hook_'] = internal_recall['_hook_'] 
            assert State_Recall['_hook_'] is not None
        return actions_list, State_Recall 
