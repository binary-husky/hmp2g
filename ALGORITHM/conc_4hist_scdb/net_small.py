import torch, math, copy
import numpy as np
import torch.nn as nn
from torch.distributions.categorical import Categorical
from UTIL.tensor_ops import Args2tensor_Return2numpy, Args2tensor, __hashn__, repeat_at
from UTIL.tensor_ops import gather_righthand, _2tensor
from UTIL.exp_helper import changed
from .ccategorical import CCategorical
from .foundation import AlgorithmConfig
from ALGORITHM.commom.attention import SimpleAttention
from ALGORITHM.commom.norm import DynamicNormFix
from ALGORITHM.commom.net_manifest import weights_init
from ALGORITHM.commom.logit2act import Logit2Act



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
        self.use_normalization = AlgorithmConfig.use_normalization
        self.use_policy_resonance = AlgorithmConfig.use_policy_resonance
        self.n_action = n_action
        self.stage_planner = stage_planner
        self.ccategorical = CCategorical(stage_planner)

        h_dim = AlgorithmConfig.net_hdim
        self.state_dim = state_dim
        self.n_action = n_action

        # observation normalization
        if self.use_normalization:
            self.at_batch_norm = DynamicNormFix(rawob_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)
            self.at_state_batch_norm = DynamicNormFix(state_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

        
        # # # # # # # # # #  actor-critic share # # # # # # # # # # # #
        self.at_obs_encoder = nn.Sequential(
            nn.Linear(rawob_dim, h_dim), nn.ReLU(inplace=True), 
            nn.Linear(h_dim, h_dim)
        )
        # # # # # # # # # #        actor        # # # # # # # # # # # #
        self.at_policy_head = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, self.n_action)
        )
        assert self.n_action <= h_dim

        # # # # # # # # # # critic # # # # # # # # # # # #
        self.ct_encoder = nn.Sequential(
            nn.Linear(h_dim+state_dim, h_dim), nn.ReLU(inplace=True), 
            nn.Linear(h_dim, h_dim)
        )
        self.ct_attention_layer = SimpleAttention(h_dim=h_dim)
        self.ct_get_value = nn.Sequential(
            nn.Linear(h_dim, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, AlgorithmConfig.distribution_precision)
        )


        self.is_recurrent = False
        self.apply(weights_init)
        return

    def _act(self, obs, state, test_mode, eval_mode=False, eval_actions=None, avail_act=None, eprsn=None):
        eval_act = eval_actions if eval_mode else None
        others = {}
        if self.use_normalization:
            obs = self.at_batch_norm(obs, freeze=(eval_mode or test_mode))
            state = self.at_state_batch_norm(state, freeze=(eval_mode or test_mode))

        obs = torch.nan_to_num_(obs, 0)         # replace dead agents' obs, from NaN to 0
        
        # # # # # # # # # # actor-critic share # # # # # # # # # # # #
        bac = self.at_obs_encoder(obs)

        # # # # # # # # # # actor # # # # # # # # # # # #
        logits = self.at_policy_head(bac)
        
        # choose action selector
        logit2act = self._logit2act_rsn if self.use_policy_resonance and self.stage_planner.is_resonance_active() else self._logit2act
        
        # apply action selector
        act, actLogProbs, distEntropy, probs = logit2act(   logits,
                                                            eval_mode=eval_mode,
                                                            greedy=test_mode,
                                                            eval_actions=eval_act,
                                                            avail_act=avail_act,
                                                            eprsn=eprsn)
        
        # # # # # # # # # # critic # # # # # # # # # # # #
        n_agent = bac.shape[-2]

        state_cp = repeat_at(state, -2, n_agent)
        ct_bac = torch.cat((bac, state_cp), dim=-1)
        ct_bac = self.ct_encoder(ct_bac)
        ct_bac = self.ct_attention_layer(k=ct_bac,q=ct_bac,v=ct_bac)
        BAL_value_all_level = self.ct_get_value(ct_bac)
        value = self.select_value_level(BAL_value_all_level, eprsn, n_agent)

        # in this mode, value is used for advantage calculation
        if not eval_mode: return act, BAL_value_all_level, actLogProbs
        # in this mode, value is used for critic regression
        else:             return value, actLogProbs, distEntropy, probs, others

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
    
    @Args2tensor_Return2numpy
    def act(self, *args, **kargs):
        return self._act(*args, **kargs)

    @Args2tensor
    def evaluate_actions(self, *args, **kargs):
        return self._act(*args, **kargs, eval_mode=True)

