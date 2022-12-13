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
        if self.ScenarioConfig.EntityOriented:
            self.rawobs_dim = self.core_dim = self.ScenarioConfig.obs_vec_length
        else:
            self.rawobs_dim = self.core_dim = space['obs_space']['obs_shape']
        if alg_config.shell_obs_add_id:
            self.core_dim = self.core_dim + self.n_agent
        if alg_config.shell_obs_add_previous_act:
            self.core_dim = self.core_dim + self.n_action
        self.n_entity_placeholder = alg_config.n_entity_placeholder
        assert self.n_entity_placeholder >= 4

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

        if not self.ScenarioConfig.EntityOriented:
            obs = repeat_at(obs, insert_dim=-2, n_times=self.n_entity_placeholder//2, copy_mem=True)
            obs[:,:,2:] = np.nan    # 0 is self; 1 is repeated self; 2,3,... is NaN

        if RST.all():
            if self.use_policy_resonance: self.rl_functional.stage_planner.uprate_eprsn(self.n_thread)
            previous_act_onehot = np.zeros((self.n_thread, self.n_agent, self.n_action), dtype=float)
        else:
            previous_act_onehot = State_Recall['_Previous_Act_Onehot_'] # 利用State_Recall的回环特性，读取上次决策的状态

        # shell_obs_add_id
        if self.alg_config.shell_obs_add_id:
            obs = np.concatenate((obs, repeat_at(repeat_at(np.eye(10,dtype=obs.dtype), -2, obs.shape[-2]), 0, obs.shape[0])), -1)
        if self.alg_config.shell_obs_add_previous_act:
            obs = np.concatenate((obs, repeat_at(previous_act_onehot, -2, obs.shape[-2])), -1)
        obs[~alive] = np.nan

        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=np.int) - 1 # 初始化全部为 -1
        his_pool_obs = State_Recall['_Histpool_Obs_'] if '_Histpool_Obs_' in State_Recall \
            else my_view(np.zeros_like(obs),[0, 0, -1, self.core_dim])
        his_pool_obs[RST] = 0
        state = np.array([info['state'] for info in State_Recall['Latest-Team-Info']])

        obs_feed = obs[~P]
        state_feed = state[~P]
        alive_feed = alive[~P]
        his_pool_obs_feed = his_pool_obs[~P]
        obs_feed_in, his_pool_next = self.solve_duplicate(obs_feed.copy(), his_pool_obs_feed.copy(), alive_feed.copy())
        his_pool_obs[~P] = his_pool_next
        his_pool_obs[P] = 0
        eprsn = self.rl_functional.stage_planner.eprsn[~P] if self.use_policy_resonance else None

        I_State_Recall = {
            'obs':obs_feed_in, 
            'alive':alive,
            'state':state_feed, 
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
        State_Recall['_Histpool_Obs_'] = his_pool_obs
        State_Recall['_Previous_Act_Onehot_'] = np_one_hot(act, n=self.n_action)
        if not State_Recall['Test-Flag']:
            State_Recall['_hook_'] = internal_recall['_hook_'] 
            assert State_Recall['_hook_'] is not None
        return actions_list, State_Recall 

    def solve_duplicate(self, obs_feed_new, prev_his_pool, alive):
        # input might be (n_thread, n_agent, n_entity, basic_dim), or (n_thread, n_agent, n_entity*basic_dim)
        # both can be converted to (n_thread, n_agent, n_entity, basic_dim)
        obs_feed_new = my_view(obs_feed_new,[0, 0, -1, self.core_dim])
        prev_obs_feed = my_view(prev_his_pool,[0, 0, -1, self.core_dim])

        # turn history into more entities
        obs_feed = np.concatenate((obs_feed_new, prev_obs_feed), axis=-2)

        # turning all zero padding to NaN, which is excluded in normalization
        obs_feed[(obs_feed==0).all(-1)] = np.nan
        obs_feed_new[(obs_feed_new==0).all(-1)] = np.nan
        valid_mask = ~np.isnan(obs_feed_new).any(-1)

        # set self as not valid to avoid buffering self obs! valid_mask
        valid_mask[:,:,0] = False

        # N valid: how many subjects (entities) needs to be buffered
        N_valid = valid_mask.sum(-1)

        # alloc mem for next_his_pool
        next_his_pool = np.zeros_like(prev_obs_feed) # twice size  ( threads,  agents,  subjects)

        # fill next_his_pool
        next_his_pool = roll_hisory(obs_feed_new, prev_obs_feed, valid_mask, N_valid, next_his_pool)

        # a very important assumption: if an agent observe itself as NaN When it is not alive
        obs_feed[~alive] = np.nan

        return obs_feed, next_his_pool
