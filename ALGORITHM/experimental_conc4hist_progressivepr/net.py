import torch,time,random
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
from torch.nn.modules.linear import Linear
from ALGORITHM.common.norm import DynamicNormFix
from ALGORITHM.common.conc import Concentration
from ALGORITHM.common.mlp import LinearFinal
from ALGORITHM.common.net_manifest import weights_init
from ALGORITHM.common.logit2act import Logit2Act
from .ccategorical import CCategorical
from UTIL.tensor_ops import repeat_at, Args2tensor_Return2numpy, Args2tensor, __hash__, __hashn__, _2tensor, gather_righthand



"""
    network initialize
"""
class Net(Logit2Act, nn.Module):
    def __init__(self,
                rawob_dim,
                state_dim,
                n_action,
                n_agent,
                stage_planner):
        super().__init__()

        from .foundation import AlgorithmConfig

        self.use_normalization = AlgorithmConfig.use_normalization
        self.n_focus_on = AlgorithmConfig.n_focus_on
        self.n_entity_placeholder = 24
        self.stage_planner = stage_planner
        self.ccategorical = CCategorical(stage_planner)
        self.use_policy_resonance = AlgorithmConfig.use_policy_resonance
        self.distribution_precision = AlgorithmConfig.distribution_precision
        h_dim = 16
        self.state_dim = state_dim

        self.n_action = n_action

        # observation normalization
        if self.use_normalization:
            self._batch_norm = DynamicNormFix(rawob_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)
            self._state_batch_norm = DynamicNormFix(state_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

        self.AT_obs_encoder = nn.Sequential(nn.Linear(rawob_dim, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, h_dim))

        self.MIX_conc_core_f = Concentration(
                        n_focus_on=self.n_focus_on-1, h_dim=h_dim, 
                        skip_connect=True, 
                        skip_connect_dim=rawob_dim)
        self.MIX_conc_core_h = Concentration(
                        n_focus_on=self.n_focus_on, h_dim=h_dim, 
                        skip_connect=True, 
                        skip_connect_dim=rawob_dim)

        tmp_dim = h_dim*2
        self.CT_get_value = nn.Sequential(
            Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),
            Linear(h_dim, self.distribution_precision)
        )
        self.CT_get_threat = nn.Sequential(
            Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),
            Linear(h_dim, 1)
        )

        # part
        self.check_n = self.n_focus_on*2

        self.AT_get_logit_db = nn.Sequential(  
            nn.Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim//2), nn.ReLU(inplace=True),
            LinearFinal(h_dim//2, self.n_action))
        
        self.is_recurrent = False
        self.apply(weights_init)
        return



    # div entity for DualConc models, distincting friend or hostile (present or history)
    def div_entity(self, mat, n):
        assert n == self.n_entity_placeholder and n % 2 == 0
        type=[  (0,),                      # self
                tuple(range(1, n//2)),     # current
                tuple(range(n//2, n)),]    # history
        
        if mat.dtype is torch.float:
            tmp = (mat[..., t, :] for t in type)
        elif mat.dtype is torch.bool:
            tmp = (mat[..., t] for t in type)
        else:
            assert False
        return tmp

    def _act(self, obs, state, test_mode, eval_mode=False, eval_actions=None, avail_act=None, eprsn=None):
        eval_act = eval_actions if eval_mode else None
        others = {}
        if self.use_normalization:
            obs = self._batch_norm(obs)
        mask_dead = torch.isnan(obs).any(-1)    # find dead agents
        obs = torch.nan_to_num_(obs, 0)         # replace dead agents' obs, from NaN to 0
        v = self.AT_obs_encoder(obs)

        zs, ze_f, ze_h          = self.div_entity(obs,       n=self.n_entity_placeholder)
        vs, ve_f, ve_h          = self.div_entity(v,         n=self.n_entity_placeholder)
        _, ve_f_dead, ve_h_dead = self.div_entity(mask_dead, n=self.n_entity_placeholder)

        # concentration module
        vh_C, vh_M = self.MIX_conc_core_h(vs=vs, ve=ve_h, ve_dead=ve_h_dead, skip_connect_ze=ze_h, skip_connect_zs=zs)
        vf_C, vf_M = self.MIX_conc_core_f(vs=vs, ve=ve_f, ve_dead=ve_f_dead, skip_connect_ze=ze_f, skip_connect_zs=zs)

        # fuse forward path
        v_C_fuse = torch.cat((vf_C, vh_C), dim=-1)  # (vs + vs + check_n + check_n)
        logits = self.AT_get_logit_db(v_C_fuse)

        # motivation encoding fusion
        n_agent = vh_M.shape[-2]

        # eprsn_cp = repeat_at(eprsn.sum(-1, keepdim=True)/n_agent, -2, n_agent)
        v_M_fuse = torch.cat((vf_M, vh_M), dim=-1)

        # motivation objectives
        BAL_value_all_level = self.CT_get_value(v_M_fuse)   # BAL
        threat = self.CT_get_threat(v_M_fuse)

        # choose action selector
        logit2act = self._logit2act_rsn if self.use_policy_resonance and self.stage_planner.is_resonance_active() else self._logit2act
        act, actLogProbs, distEntropy, probs = logit2act(logits, eval_mode=eval_mode, greedy=test_mode, eprsn=eprsn,
                                                                eval_actions=eval_act, avail_act=avail_act)

        def re_scale(t):
            SAFE_LIMIT = 8
            r = 1. /2. * SAFE_LIMIT
            return (torch.tanh_(t/r) + 1.) * r

        others['threat'] = re_scale(threat)
        value = self.select_value_level(BAL_value_all_level, eprsn, n_agent)

        # in this mode, value is used for advantage calculation
        if not eval_mode: return act, BAL_value_all_level, actLogProbs
        # in this mode, value is used for critic regression
        else:             return value, actLogProbs, distEntropy, probs, others

        
    @Args2tensor_Return2numpy
    def act(self, *args, **kargs):
        return self._act(*args, **kargs)

    @Args2tensor
    def evaluate_actions(self, *args, **kargs):
        return self._act(*args, **kargs, eval_mode=True)

    def select_value_level(self, BAL_value_all_level, eprsn, n_agent):
        def get_lv(eprsn):
            with torch.no_grad():
                asum = eprsn.sum(-1).cpu().numpy()
                lv = np.array(list(map(self.stage_planner.mapPrNum2Level, asum)))
                lv = _2tensor(lv).long()
                return repeat_at(lv, -1, n_agent).unsqueeze(-1)
        lv = get_lv(eprsn)
        value = gather_righthand(src=BAL_value_all_level, index=lv, check=False)
        return value
    