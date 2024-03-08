import numpy as np
import importlib
from config import GlobalConfig
from UTIL.colorful import *
from UTIL.tensor_ops import my_view, __hash__, repeat_at
from .pymarl2_compat import AlgorithmConfig
from ALGORITHM.common.norm import DynamicNormFix
class ShellEnv(object):
    def __init__(self, rl_foundation, n_agent, n_thread, space, mcv, team):
        self.n_agent = n_agent
        self.n_thread = n_thread
        self.team = team
        self.rl_foundation = rl_foundation
        self.n_entity = GlobalConfig.ScenarioConfig.obs_n_entity
        self.obssize = GlobalConfig.ScenarioConfig.obs_vec_length

        if AlgorithmConfig.state_compat == 'pad':
            self.state_size = 4
        elif AlgorithmConfig.state_compat == 'obs_cat':
            self.state_size = self.n_entity * self.obssize * self.n_agent
        elif AlgorithmConfig.state_compat == 'obs_mean':
            self.state_size = self.n_entity * self.obssize
        else:
            assert False, 'compat method error'

        # init action converter
        module_, class_ = AlgorithmConfig.action_converter.split('->')
        init_f = getattr(importlib.import_module(module_), class_)
        self.action_converter = init_f(
                SELF_TEAM_ASSUME=team, 
                OPP_TEAM_ASSUME=(1-team), 
                OPP_NUM_ASSUME=GlobalConfig.ScenarioConfig.N_AGENT_EACH_TEAM[1-team]
        )

        self.rl_foundation.space = {
            'act_space':{
                'n_actions': len(self.action_converter.dictionary_args),
            },
            'obs_space':{
                'obs_shape': self.n_entity * self.obssize,
                'state_shape': self.state_size,
            }
        }
        self.rl_foundation.interact_with_env_real = self.rl_foundation.interact_with_env
        self.rl_foundation.interact_with_env = self.interact_with_env
        self.patience = 2000
        
        opp_type_list = [a['type'] for a in GlobalConfig.ScenarioConfig.SubTaskConfig.agent_list if a['team']!=self.team]
        self.action_converter.confirm_parameters_are_correct(team, self.n_agent, len(opp_type_list))
        
        if AlgorithmConfig.use_shell_normalization:
            # if shell normalization is enabled
            self.obs_norm = DynamicNormFix(input_size=self.n_entity * self.obssize, only_for_last_dim=True, exclude_nan=True).to(GlobalConfig.device)
            self.state_norm = DynamicNormFix(input_size=self.state_size, only_for_last_dim=True, exclude_nan=True).to(GlobalConfig.device)
            self.obs_norm_fn = self.obs_norm.np_forward
            self.state_norm_fn = self.state_norm.np_forward
        else:
            # if shell normalization is disabled, then leave a dummy function here
            self.obs_norm_fn = self.state_norm_fn = lambda x: x

    def interact_with_env(self, StateRecall):
        if not hasattr(self, 'agent_type'):
            self.agent_uid = GlobalConfig.ScenarioConfig.AGENT_ID_EACH_TEAM[self.team]
            self.agent_type = [agent_meta['type'] for agent_meta in StateRecall['Latest-Team-Info'][0]['dataArr']
                if agent_meta['uId'] in self.agent_uid]
            self.avail_act = np.stack(tuple(self.action_converter.get_tp_avail_act(tp) for tp in self.agent_type))
            self.avail_act = repeat_at(self.avail_act, insert_dim=0, n_times=self.n_thread)
        StateRecall['Latest-Obs'] = np.nan_to_num(
                self.obs_norm_fn(my_view(StateRecall['Latest-Obs'], [0,0,-1]))
            , 0)
        StateRecall['Terminal-Obs-Echo'] = [np.nan_to_num(my_view(np.array(t, dtype=float), [0,-1]), 0)   if t is not None else None for t in StateRecall['Terminal-Obs-Echo']]
        for i, d in enumerate(StateRecall['Latest-Team-Info']):
            if AlgorithmConfig.state_compat == 'pad':
                d['state']      = np.zeros(self.state_size)
                d['state-echo'] = np.zeros(self.state_size)
            elif AlgorithmConfig.state_compat == 'obs_cat':
                d['state']      = self.state_norm_fn(StateRecall['Latest-Obs'][i].flatten())
                d['state-echo'] = np.zeros_like(d['state'])
                if StateRecall['Terminal-Obs-Echo'][i] is not None: 
                    d['state-echo'] = self.state_norm_fn(StateRecall['Terminal-Obs-Echo'][i].flatten())
            elif AlgorithmConfig.state_compat == 'obs_mean':
                d['state']      = self.state_norm_fn(StateRecall['Latest-Obs'][i].mean(0))
                d['state-echo'] = np.zeros_like(d['state'])
                if StateRecall['Terminal-Obs-Echo'][i] is not None: 
                    d['state-echo'] = self.state_norm_fn(StateRecall['Terminal-Obs-Echo'][i].mean(0))
            else:
                assert False, 'compat method error'
            d['avail-act'] = self.avail_act[i]
            
        ret_action_list, team_intel = self.rl_foundation.interact_with_env_real(StateRecall)
        ret_action_list = np.swapaxes(ret_action_list, 0, 1)

        R  = ~StateRecall['ENV-PAUSE']
        
        act_converted = np.array([
            [ self.action_converter.convert_act_arr(self.agent_type[agentid], int(act)) 
                    if not np.isnan(act) else 
                    self.action_converter.convert_act_arr(self.agent_type[agentid], 0) + np.nan
                    for agentid, act    in enumerate(th)    ] 
                    for th              in ret_action_list  ])
        
        # act_converted[mask] = np.nan
        if GlobalConfig.mt_act_order != 'new_method':
            act_converted = np.swapaxes(act_converted, 0, 1)
        return act_converted, team_intel
    
    