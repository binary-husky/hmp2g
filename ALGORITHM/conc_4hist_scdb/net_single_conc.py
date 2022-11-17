import math
import torch,time,random
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from torch.distributions.multivariate_normal import MultivariateNormal
from torch.nn.modules.linear import Linear
from ..commom.attention import MultiHeadAttention
from ..commom.norm import DynamicNormFix
from ..commom.mlp import LinearFinal, SimpleMLP, ResLinear
from .ccategorical import CCategorical
from UTIL.tensor_ops import my_view, Args2tensor_Return2numpy, Args2tensor, __hash__, __hashn__, pad_at_dim
from UTIL.tensor_ops import repeat_at, one_hot_with_nan, gather_righthand, pt_inf


def weights_init(m):
    def init_Linear(m, final_layer=False):
        nn.init.orthogonal_(m.weight.data)
        if final_layer:nn.init.orthogonal_(m.weight.data, gain=0.01)
        if m.bias is not None: nn.init.uniform_(m.bias.data, a=-0.02, b=0.02)

    initial_fn_dict = {
        'Net': None, 'DataParallel':None, 'BatchNorm1d':None, 'Concentration':None,
        'Pnet':None,'Sequential':None,'DataParallel':None,'Tanh':None,
        'ModuleList':None,'ModuleDict':None,'MultiHeadAttention':None,
        'SimpleMLP':None,'Extraction_Module':None,'SelfAttention_Module':None,
        'ReLU':None,'Softmax':None,'DynamicNormFix':None,'EXTRACT':None,
        'LinearFinal':lambda m:init_Linear(m, final_layer=True),
        'Linear':init_Linear, 'ResLinear':None, 'LeakyReLU':None,'SimpleAttention':None
    }

    classname = m.__class__.__name__
    assert classname in initial_fn_dict.keys(), ('how to handle the initialization of this class? ', classname)
    init_fn = initial_fn_dict[classname]
    if init_fn is None: return
    init_fn(m)

class Concentration(nn.Module):
    def __init__(self, n_focus_on, h_dim, skip_connect=False, skip_connect_dim=0, adopt_selfattn=False):
        super().__init__()
        self.n_focus_on = n_focus_on
        self.skip_connect = skip_connect
        self.skip_dim = h_dim+skip_connect_dim
        self.CT_W_query = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.CT_W_key = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.CT_W_val = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.CT_motivate_mlp = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.ReLU(inplace=True))
        self.AT_forward_mlp = nn.Sequential(nn.Linear((n_focus_on+1)*self.skip_dim, h_dim), nn.ReLU(inplace=True))
        self.adopt_selfattn = adopt_selfattn
        if self.adopt_selfattn:
            self.AT_Attention = Extraction_Module(hidden_dim=self.skip_dim, activate_output=True)
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, vs, ve, ve_dead, skip_connect_ze=None, skip_connect_zs=None):
        mask = ve_dead
        Q = torch.matmul(vs, self.CT_W_query) 
        K = torch.matmul(ve, self.CT_W_key) 

        norm_factor = 1 / math.sqrt(Q.shape[-1])
        compat = norm_factor * torch.matmul(Q, K.transpose(2, 3)) 
        assert compat.shape[-2] == 1
        compat = compat.squeeze(-2)
        compat[mask.bool()] = -math.inf
        score = F.softmax(compat, dim=-1)
        # nodes with no neighbours were softmax into nan, fix them to 0
        score = torch.nan_to_num(score, 0)
        # ----------- motivational brach -------------
        Va = torch.matmul(score.unsqueeze(-2), torch.matmul(ve, self.CT_W_val)) 
        v_M = torch.cat((vs, Va), -1).squeeze(-2) 
        v_M_final = self.CT_motivate_mlp(v_M)
        # -----------   forward branch   -------------
        score_sort_index = torch.argsort(score, dim=-1, descending=True)
        score_sort_drop_index = score_sort_index[..., :self.n_focus_on]
        if self.skip_connect:
            ve = torch.cat((ve, skip_connect_ze), -1)
            vs = torch.cat((vs, skip_connect_zs), -1)
        ve_C = gather_righthand(src=ve,  index=score_sort_drop_index, check=False)
        need_padding = (score_sort_drop_index.shape[-1] != self.n_focus_on)
        if need_padding:
            print('the n_focus param is large than input, advise: pad observation instead of pad here')
            ve_C = pad_at_dim(ve_C, dim=-2, n=self.n_focus_on)
        v_C_stack = torch.cat((vs, ve_C), dim=-2)
        if self.adopt_selfattn:
            v_C_stack = self.AT_Attention(v_C_stack, mask=None)

        v_C_flat = my_view(v_C_stack, [0, 0, -1]); assert v_C_stack.dim()==4
        v_C_final = self.AT_forward_mlp(v_C_flat)
        return v_C_final, v_M_final


class SimpleAttention(nn.Module):
    def __init__(self, h_dim):
        super().__init__()
        self.W_query = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.W_key = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.W_val = nn.Parameter(torch.Tensor(h_dim, h_dim))
        self.init_parameters()

    def init_parameters(self):
        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, k, q, v, mask=None):
        Q = torch.matmul(q, self.W_query) 
        K = torch.matmul(k, self.W_key) 
        V = torch.matmul(v, self.W_val)

        norm_factor = 1 / math.sqrt(Q.shape[-1])
        compat = norm_factor * torch.matmul(Q, K.transpose(2, 3)) 
        if mask is not None: compat[mask.bool()] = -math.inf
        score = torch.nan_to_num(F.softmax(compat, dim=-1), 0)
        # ----------- motivational brach -------------
        return torch.matmul(score, V) 


class Extraction_Module(nn.Module): # merge by MLP version
    def __init__(self, hidden_dim=128, activate_output=False):
        super().__init__()
        h_dim = hidden_dim
        from .foundation import AlgorithmConfig
        if AlgorithmConfig.use_my_attn:
            self.attn = SimpleAttention(h_dim=h_dim)
            print('use my attn')

        if activate_output:
            self.MLP = nn.Sequential(nn.Linear(h_dim * 2, h_dim), nn.ReLU(inplace=True))
            print("activate_output")
        else:
            self.MLP = nn.Sequential(nn.Linear(h_dim * 2, h_dim))
            print("no activate_output")

    def forward(self, agent_enc, mask=None):
        attn_out = self.attn(q=agent_enc, k=agent_enc, v=agent_enc, mask=mask)
        concated_attn_result = torch.cat(tensors=(agent_enc, attn_out), dim=-1)
        return self.MLP(concated_attn_result)



"""
    network initialize
"""
class Net(nn.Module):
    def __init__(self, 
                rawob_dim,
                state_dim,
                n_action,
                stage_planner,
                ):
        super().__init__()

        from .foundation import AlgorithmConfig

        self.use_normalization = AlgorithmConfig.use_normalization
        self.n_focus_on = AlgorithmConfig.n_focus_on
        self.actor_attn_mod = AlgorithmConfig.actor_attn_mod
        self.dual_conc = AlgorithmConfig.dual_conc
        self.n_entity_placeholder = AlgorithmConfig.n_entity_placeholder
        self.stage_planner = stage_planner
        self.ccategorical = CCategorical(stage_planner)
        self.use_policy_resonance = AlgorithmConfig.use_policy_resonance
        h_dim = AlgorithmConfig.net_hdim
        state_dim = state_dim

        self.skip_connect = True
        self.n_action = n_action
        self.alternative_critic = AlgorithmConfig.alternative_critic
        
        # observation normalization
        if self.use_normalization:
            self._batch_norm = DynamicNormFix(rawob_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)
            self._state_batch_norm = DynamicNormFix(state_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True)

        self.AT_obs_encoder = nn.Sequential(nn.Linear(rawob_dim, h_dim), nn.ReLU(inplace=True), nn.Linear(h_dim, h_dim))

        if self.dual_conc:
            self.MIX_conc_core_f = Concentration(
                            n_focus_on=self.n_focus_on-1, h_dim=h_dim, 
                            skip_connect=self.skip_connect, 
                            skip_connect_dim=rawob_dim, 
                            adopt_selfattn=self.actor_attn_mod)
            self.MIX_conc_core_h = Concentration(
                            n_focus_on=self.n_focus_on, h_dim=h_dim, 
                            skip_connect=self.skip_connect, 
                            skip_connect_dim=rawob_dim, 
                            adopt_selfattn=self.actor_attn_mod)
        else:
            self.MIX_conc_core = Concentration(
                            n_focus_on=self.n_focus_on, h_dim=h_dim, 
                            skip_connect=self.skip_connect, 
                            skip_connect_dim=rawob_dim, 
                            adopt_selfattn=self.actor_attn_mod)
 
        tmp_dim = h_dim if not self.dual_conc else h_dim*2
        self.CT_get_value = nn.Sequential(
            Linear(tmp_dim+state_dim, h_dim), nn.ReLU(inplace=True),
            Linear(h_dim, 1)
        )
        self.CT_get_threat = nn.Sequential(
            Linear(tmp_dim+state_dim, h_dim), nn.ReLU(inplace=True),
            Linear(h_dim, 1)
        )

        if self.alternative_critic:
            self.CT_get_value_alternative_critic = nn.Sequential(Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),Linear(h_dim, 1))

        # part
        self.check_n = self.n_focus_on*2
        self.AT_get_logit_db = nn.Sequential(  
            nn.Linear(tmp_dim, h_dim), nn.ReLU(inplace=True),
            nn.Linear(h_dim, h_dim//2), nn.ReLU(inplace=True),
            LinearFinal(h_dim//2, self.n_action))
            

        self.is_recurrent = False
        self.apply(weights_init)
        return

    def _logit2act_rsn(self, logits_agent_cluster, eval_mode, greedy, eval_actions=None, avail_act=None, eprsn=None):
        if avail_act is not None: logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())
        act_dist = self.ccategorical.feed_logits(logits_agent_cluster)
        
        if not greedy:    act = self.ccategorical.sample(act_dist, eprsn) if not eval_mode else eval_actions
        else:             act = torch.argmax(act_dist.probs, axis=2)
        # the policy gradient loss will feedback from here
        actLogProbs = self._get_act_log_probs(act_dist, act) 
        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs

    def _logit2act(self, logits_agent_cluster, eval_mode, greedy, eval_actions=None, avail_act=None, **kwargs):
        if avail_act is not None: logits_agent_cluster = torch.where(avail_act>0, logits_agent_cluster, -pt_inf())
        act_dist = Categorical(logits = logits_agent_cluster)
        if not greedy:     act = act_dist.sample() if not eval_mode else eval_actions
        else:              act = torch.argmax(act_dist.probs, axis=2)
        actLogProbs = self._get_act_log_probs(act_dist, act) # the policy gradient loss will feedback from here
        # sum up the log prob of all agents
        distEntropy = act_dist.entropy().mean(-1) if eval_mode else None
        return act, actLogProbs, distEntropy, act_dist.probs

    @staticmethod
    def _get_act_log_probs(distribution, action):
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)
        
    @Args2tensor_Return2numpy
    def act(self, *args, **kargs):
        act = self._act if self.dual_conc else self._act_singlec
        return act(*args, **kargs)

    @Args2tensor
    def evaluate_actions(self, *args, **kargs):
        act = self._act if self.dual_conc else self._act_singlec
        return act(*args, **kargs, eval_mode=True)

    # div entity for DualConc models, distincting friend or hostile (present or history)
    def div_entity(self, mat, n):
        assert n == self.n_entity_placeholder and n % 2 == 0
        type=[  (0,),                   # self
                tuple(range(1, n)),]    # history
        
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
            state = self._state_batch_norm(state)
        mask_dead = torch.isnan(obs).any(-1)    # find dead agents
        obs = torch.nan_to_num_(obs, 0)         # replace dead agents' obs, from NaN to 0
        v = self.AT_obs_encoder(obs)

        n_entity = obs.shape[-2]
        zs, ze     = self.div_entity(obs,       n=n_entity)
        vs, ve     = self.div_entity(v,         n=n_entity)
        _, ve_dead = self.div_entity(mask_dead, n=n_entity)

        # concentration module, ve_dead is just a mask filtering out invalid or padding entities
        v_C, v_M = self.MIX_conc_core(vs=vs, ve=ve, ve_dead=ve_dead, skip_connect_ze=ze, skip_connect_zs=zs)
        # fuse forward path
        logits = self.AT_get_logit_db(v_C) # diverge here
        # motivation encoding fusion
        n_agent = v_M.shape[-2]
        state_cp = repeat_at(state, -2, n_agent)
        v_M_fuse = torch.cat((v_M, state_cp), dim=-1)
        # motivation objectives
        value = self.CT_get_value(v_M_fuse)
        threat = self.CT_get_threat(v_M_fuse)

        # choose action selector
        logit2act = self._logit2act_rsn if self.use_policy_resonance and self.stage_planner.is_resonance_active() else self._logit2act
        
        assert not self.alternative_critic
        act, actLogProbs, distEntropy, probs = logit2act(logits, eval_mode=eval_mode, greedy=test_mode, eprsn=eprsn,
                                                                eval_actions=eval_act, avail_act=avail_act)

        def re_scale(t):
            SAFE_LIMIT = 8
            r = 1. /2. * SAFE_LIMIT
            return (torch.tanh_(t/r) + 1.) * r

        others['threat'] = re_scale(threat)
        if not eval_mode: return act, value, actLogProbs
        else:             return value, actLogProbs, distEntropy, probs, others

