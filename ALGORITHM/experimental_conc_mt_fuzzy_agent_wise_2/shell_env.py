import numpy as np
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__



class ShellEnvWrapper(object):
    def __init__(self, n_agent, n_thread, space, mcv, rl_functional,
                                          alg_config, ScenarioConfig):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.space = space
        self.mcv = mcv
        self.rl_functional = rl_functional
        self.n_basic_dim = ScenarioConfig.obs_vec_length

        # whether to use avail_act to block forbiden actions
        self.AvailActProvided = False
        if hasattr(ScenarioConfig, 'AvailActProvided'):
            self.AvailActProvided = ScenarioConfig.AvailActProvided

        self.use_policy_resonance = alg_config.use_policy_resonance
        # whether to load previously saved checkpoint
        self.load_checkpoint = alg_config.load_checkpoint
        self.cold_start = True

    def interact_with_env(self, StateRecall):
        act = np.zeros(shape=(self.n_thread, self.n_agent), dtype=int) - 1 # 初始化全部为 -1
        # read internal coop graph info
        obs = StateRecall['Latest-Obs']
        RST = StateRecall['Env-Suffered-Reset']
        alive = ~((obs==0).all(-1))
        n_thread = obs.shape[0]
        if RST.all():
            if self.use_policy_resonance: self.rl_functional.stage_planner.uprate_eprsn(self.n_thread)

        previous_obs = StateRecall['_Previous_Obs_'] if '_Previous_Obs_' in StateRecall else np.zeros_like(obs)

        P = StateRecall['ENV-PAUSE']
        obs_feed = obs[~P]
        alive_feed = alive[~P]
        prev_obs_feed = previous_obs[~P]
        eprsn = self.rl_functional.stage_planner.eprsn[~P] if self.use_policy_resonance else None
        randl = self.rl_functional.stage_planner.randl[~P] if self.use_policy_resonance else None

        obs_feed_in = self.solve_duplicate(obs_feed, prev_obs_feed)

        I_StateRecall = {
            'obs':obs_feed_in,
            'alive':alive_feed,
            'randl':randl,
            'eprsn':eprsn,
            'Test-Flag':StateRecall['Test-Flag'],
            'threads_active_flag':~P,
            'Latest-Team-Info':StateRecall['Latest-Team-Info'][~P],
            }
        if self.AvailActProvided:
            avail_act = np.array([info['avail-act'] for info in np.array(StateRecall['Latest-Team-Info'][~P], dtype=object)])
            I_StateRecall.update({'avail_act':avail_act})

        act_active, internal_recall = self.rl_functional.interact_with_env_genuine(I_StateRecall)

        act[~P] = act_active
        actions_list = np.expand_dims(act, -1)

        # return necessary handles to main platform
        if self.cold_start: self.cold_start = False

        # <2> call a empty frame to gather reward
        StateRecall['_Previous_Obs_'] = obs
        if not StateRecall['Test-Flag']:
            StateRecall['_hook_'] = internal_recall['_hook_']
            assert StateRecall['_hook_'] is not None
        return actions_list, StateRecall

    def solve_duplicate(self, obs_feed, prev_obs_feed):
        #  input might be (n_thread, n_agent, n_entity, basic_dim), or (n_thread, n_agent, n_entity*basic_dim)
        # both can be converted to (n_thread, n_agent, n_entity, basic_dim)
        obs_feed = my_view(obs_feed,[0, 0, -1, self.n_basic_dim])

        # turning all zero padding to NaN, used for normalization
        obs_feed[(obs_feed==0).all(-1)] = np.nan

        return obs_feed
