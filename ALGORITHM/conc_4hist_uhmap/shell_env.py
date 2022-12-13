import numpy as np
from config import GlobalConfig
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__, repeat_at
from MISSION.uhmap.actset_lookup import encode_action_as_digits
from .cython_func import roll_hisory

class ActionConvertLegacy():

    # (main_cmd, sub_cmd, x=None, y=None, z=None, UID=None, T=None, T_index=None)
    dictionary_args = [
        ('N/A',         'N/A',              None, None, None, None, None, None),   # 0
        ('Idle',        'DynamicGuard',     None, None, None, None, None, None),   # 1
        ('Idle',        'StaticAlert',      None, None, None, None, None, None),   # 2
        ('Idle',        'AggressivePersue', None, None, None, None, None, None),   # 3
        ('Idle',        'AsFarAsPossible',              None, None, None, None, None, None),   # 1
        ('Idle',        'StayWhenTargetInRange',        None, None, None, None, None, None),   # 2
        ('Idle',        'StayWhenTargetInHalfRange',    None, None, None, None, None, None),   # 3
        ('SpecificMoving',      'Dir+X',    None, None, None, None, None, None),   # 4
        ('SpecificMoving',      'Dir+Y',    None, None, None, None, None, None),   # 5
        ('SpecificMoving',      'Dir-X',    None, None, None, None, None, None),   # 6
        ('SpecificMoving',      'Dir-Y',    None, None, None, None, None, None),   # 7
        ('SpecificAttacking',   'N/A',      None, None, None, None, 1,    0),      # 8
        ('SpecificAttacking',   'N/A',      None, None, None, None, 1,    1),      # 9
        ('SpecificAttacking',   'N/A',      None, None, None, None, 1,    2),      # 10
        ('SpecificAttacking',   'N/A',      None, None, None, None, 1,    3),      # 11
        ('SpecificAttacking',   'N/A',      None, None, None, None, 1,    4),      # 12
        ('SpecificAttacking',   'N/A',      None, None, None, None, 0,    0),      # 13
        ('SpecificAttacking',   'N/A',      None, None, None, None, 0,    1),      # 14
        ('SpecificAttacking',   'N/A',      None, None, None, None, 0,    2),      # 15
        ('SpecificAttacking',   'N/A',      None, None, None, None, 0,    3),      # 16
        ('SpecificAttacking',   'N/A',      None, None, None, None, 0,    4),      # 17
        ('PatrolMoving',        'Dir+X',    None, None, None, None, None, None),   # 19
        ('PatrolMoving',        'Dir+Y',    None, None, None, None, None, None),   # 20
        ('PatrolMoving',        'Dir-X',    None, None, None, None, None, None),   # 21
        ('PatrolMoving',        'Dir-Y',    None, None, None, None, None, None),   # 22
    ]


    @staticmethod
    def convert_act_arr(type, a):
        if 'RLA_UAV' in type:
            args = ActionConvertLegacy.dictionary_args[a]
            # override wrong actions
            if args[0] == 'SpecificAttacking':
                return encode_action_as_digits('N/A',         'N/A',              None, None, None, None, None, None)
            # override incorrect actions
            if args[0] == 'Idle':
                return encode_action_as_digits('Idle',        'StaticAlert',      None, None, None, None, None, None)
            return encode_action_as_digits(*args)
        else:
            return encode_action_as_digits(*ActionConvertLegacy.dictionary_args[a])
 

class ShellEnvWrapper(object):
    def __init__(self, n_agent, n_thread, space, mcv, RL_functional, alg_config, ScenarioConfig, team):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.team = team
        self.space = space
        self.mcv = mcv
        self.RL_functional = RL_functional
        self.core_dim = GlobalConfig.ScenarioConfig.obs_vec_length

        # whether to use avail_act to block forbiden actions
        self.AvailActProvided = False
        if hasattr(ScenarioConfig, 'AvailActProvided'):
            self.AvailActProvided = ScenarioConfig.AvailActProvided 

        # whether to load previously saved checkpoint
        self.load_checkpoint = alg_config.load_checkpoint
        self.cold_start = True

    def interact_with_env(self, State_Recall):
        if not hasattr(self, 'agent_type'):
            self.agent_uid = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[self.team]
            self.agent_type = [agent_meta['type'] 
                for agent_meta in State_Recall['Latest-Team-Info'][0]['dataArr']
                if agent_meta['uId'] in self.agent_uid]

        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=np.int) - 1 # 初始化全部为 -1
        # read internal coop graph info
        obs = State_Recall['Latest-Obs']
        P = State_Recall['ENV-PAUSE']
        RST = State_Recall['Env-Suffered-Reset']
        his_pool_obs = State_Recall['_Histpool_Obs_'] if '_Histpool_Obs_' in State_Recall \
            else my_view(np.zeros_like(obs),[0, 0, -1, self.core_dim])
        his_pool_obs[RST] = 0

        obs_feed = obs[~P]
        his_pool_obs_feed = his_pool_obs[~P]
        obs_feed_in, his_pool_next = self.solve_duplicate(obs_feed.copy(), his_pool_obs_feed.copy())
        his_pool_obs[~P] = his_pool_next
        his_pool_obs[P] = 0

        I_State_Recall = {'obs':obs_feed_in, 
            'Current-Obs-Step': State_Recall['Current-Obs-Step'],
            'Test-Flag':State_Recall['Test-Flag'], 
            'threads_active_flag':~P, 
            'Latest-Team-Info':State_Recall['Latest-Team-Info'][~P],
            }
        if self.AvailActProvided:
            avail_act = np.array([info['avail-act'] for info in np.array(State_Recall['Latest-Team-Info'][~P], dtype=object)])
            I_State_Recall.update({'avail_act':avail_act})

        act_active, internal_recall = self.RL_functional.interact_with_env_genuine(I_State_Recall)

        act[~P] = act_active

        act_converted = np.array([
            [
                ActionConvertLegacy.convert_act_arr(self.agent_type[agentid], act)  for agentid, act in enumerate(th) 
            ] for th in act])
        actions_list = np.swapaxes(act_converted, 0, 1) # swap thread(batch) axis and agent axis

        # return necessary handles to main platform
        if self.cold_start: self.cold_start = False

        # <2> call a empty frame to gather reward
        # State_Recall['_Previous_Obs_'] = obs
        State_Recall['_Histpool_Obs_'] = his_pool_obs
        
        State_Recall['_hook_'] = internal_recall['_hook_']
        assert State_Recall['_hook_'] is not None
        return actions_list, State_Recall 

    def solve_duplicate(self, obs_feed_new, prev_his_pool):
        #  input might be (n_thread, n_agent, n_entity, basic_dim), or (n_thread, n_agent, n_entity*basic_dim)
        # both can be converted to (n_thread, n_agent, n_entity, basic_dim)
        obs_feed_new = my_view(obs_feed_new,[0, 0, -1, self.core_dim])
        prev_obs_feed = my_view(prev_his_pool,[0, 0, -1, self.core_dim])

        # turn history into more entities
        obs_feed = np.concatenate((obs_feed_new, prev_obs_feed), axis=-2)

        # turning all zero padding to NaN, which is excluded in normalization
        obs_feed[(obs_feed==0).all(-1)] = np.nan
        obs_feed_new[(obs_feed_new==0).all(-1)] = np.nan
        valid_mask = ~np.isnan(obs_feed_new).any(-1)    #

        # set self as not valid to avoid buffering self obs! valid_mask
        valid_mask[:,:,0] = False

        # N valid: how many subjects (entities) needs to be buffered
        N_valid = valid_mask.sum(-1)

        # alloc mem for next_his_pool
        next_his_pool = np.zeros_like(prev_obs_feed) # twice size  ( threads,  agents,  subjects)

        # fill next_his_pool
        next_his_pool = roll_hisory(obs_feed_new, prev_obs_feed, valid_mask, N_valid, next_his_pool)

        # a very important assumption: if an agent observe itself as NaN *When and Only When* it is dead
        alive_mask = ~np.isnan(obs_feed_new[:,:,0]).any(-1) 
        if (~alive_mask).any():
            obs_feed[~alive_mask] = np.nan
        return obs_feed, next_his_pool

