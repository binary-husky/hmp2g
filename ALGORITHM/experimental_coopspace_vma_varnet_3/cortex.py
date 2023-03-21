import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from UTIL.tensor_ops import my_view, Args2tensor_Return2numpy, Args2tensor, pt_inf, repeat_at
from ALGORITHM.common.norm import DynamicNorm # Fix as DynamicNorm


def weights_init(m):
    classname = m.__class__.__name__
    if classname ==  'Conv':
        assert False
    elif classname == 'CNet':
        return
    elif classname == 'DynamicNorm':
        return
    elif classname == 'Sequential':
        return
    elif classname == 'ModuleDict':
        return
    elif classname ==  'MultiHeadAttention':
        return
    elif classname ==  'ReLU':
        m.inplace=True
    elif classname ==  'Linear':
        nn.init.orthogonal_(m.weight.data)
        if m.bias is not None: m.bias.data.fill_(0)
    elif classname == 'LinearFinal':
        nn.init.orthogonal_(m.weight.data, gain=0.01)
        m.bias.data.fill_(0)
    else:
        assert False, ('how to handle the initialization of this class? ', classname)





class CNet(nn.Module):

    def __init__(self, num_agents, num_entities, basic_vec_len, hidden_dim=32):
        from .reinforce_foundation import CoopAlgConfig
        super().__init__()
        self.n_cluster = n_cluster = CoopAlgConfig.g_num
        self.n_agent = n_agent = num_agents

        h_dim = hidden_dim
        self.n_target = n_target = num_entities
        activation_func = nn.ReLU

        if not CoopAlgConfig.one_more_container:
            _n_cluster = n_cluster
        else:
            _n_cluster = n_cluster + 1




        self.n_operators = 4


        agent_emb_dim = n_agent + basic_vec_len    +  _n_cluster
        target_emb_dim = n_target + basic_vec_len                     +  _n_cluster
        cluster_emb_dim = _n_cluster              +  n_agent           +  n_target

        self.cluster_enter_encoder = nn.Sequential(
            nn.Linear(cluster_emb_dim, h_dim),
            activation_func(),
            nn.Linear(h_dim, h_dim)
        )

        self.target_enter_encoder = nn.Sequential(
            nn.Linear(target_emb_dim, h_dim),
            activation_func(),
            nn.Linear(h_dim, h_dim)
        )

        self.agent_enter_encoder = nn.Sequential(
            nn.Linear(agent_emb_dim, h_dim),
            activation_func(),
            nn.Linear(h_dim, h_dim)
        )

        self.attns = nn.ModuleDict({
            'right': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'left': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'top': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'bottom': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'right_vtop': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'left_vtop': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'right_vbottom': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),
            'left_vbottom': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=h_dim),

            'ae_agent_pre':   nn.Linear(self.n_agent, h_dim),
            'ae_cluster_pre': nn.Linear(self.n_cluster, h_dim),
            'ae_target_pre':  nn.Linear(self.n_target, h_dim),
            'ae_agent': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=agent_emb_dim),
            'ae_cluster': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=cluster_emb_dim),
            'ae_target': MultiHeadAttention(n_heads=1, input_dim=h_dim, embed_dim=target_emb_dim),
        })

        self.central_zip = nn.Sequential(
            nn.Linear(h_dim * 4, h_dim),
            activation_func(),
            nn.Linear(h_dim, h_dim)
        )


        self.downsample = nn.ModuleDict({
            'final_downsample': nn.Linear(h_dim * 4, h_dim),
            'final_nonlin':   nn.Sequential(
                                nn.Linear(_n_cluster * h_dim, h_dim),
                                activation_func(),
                                nn.Linear(h_dim, h_dim)
                            ),
        })



        self.value_head_top = nn.ModuleDict({
            'centralize':nn.Sequential(
                            nn.Linear(h_dim, h_dim),
                            activation_func(),
                            nn.Linear(h_dim, h_dim // 2)
                        ),

            'nonlin'    :nn.Sequential(
                            nn.Linear(_n_cluster * h_dim // 2, h_dim),
                            activation_func(),
                            nn.Linear(h_dim, self.n_operators)
                        ),
        })

        self.NetDim = NetDim = {
            'operator 1 input size':  h_dim,
            'operator 1 output size': _n_cluster,

            'operator 2 input size': h_dim+_n_cluster,
            'operator 2 output size': _n_cluster,

            'operator 3 input size':  h_dim+2*_n_cluster,
            'operator 3 output size': n_target,

            'operator 4 input size': h_dim+2*_n_cluster+n_target,
            'operator 4 output size': n_target,
        }


        self.fifo_net = nn.ModuleDict({
            'operator 1': nn.Sequential(
                nn.Linear(NetDim['operator 1 input size'], h_dim),
                activation_func(),
                LinearFinal(h_dim, NetDim['operator 1 output size'])
             ),
            'operator 2': nn.Sequential(
                nn.Linear(NetDim['operator 2 input size'], h_dim),
                activation_func(),
                LinearFinal(h_dim, NetDim['operator 2 output size'])
             ),
            'operator 3': nn.Sequential(
                nn.Linear(NetDim['operator 3 input size'], h_dim),
                activation_func(),
                LinearFinal(h_dim, NetDim['operator 3 output size'])
             ),
            'operator 4': nn.Sequential(
                nn.Linear(NetDim['operator 4 input size'], h_dim),
                activation_func(),
                LinearFinal(h_dim, NetDim['operator 4 output size'])
             ),
        })


        self._batch_norm_agent  = DynamicNorm(agent_emb_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True, sample_limit=CoopAlgConfig.normalization_sample_limit)
        self._batch_norm_target = DynamicNorm(target_emb_dim, only_for_last_dim=True, exclude_one_hot=True, exclude_nan=True, sample_limit=CoopAlgConfig.normalization_sample_limit)

        self.is_recurrent = False
        self.apply(weights_init)
        return

    
    @Args2tensor_Return2numpy
    def act(self, *args, **kargs):
        return self._act(*args, **kargs)

    @Args2tensor
    def evaluate_actions(self, *args, **kargs):
        return self._act(*args, **kargs, eval_mode=True)
    
    def _act(self, all_emb, eval_mode=False, eval_actions=None, test_mode=False, avail_act=None):

        feature, value, ae_loss = self.get_feature(all_emb, eval_mode)
        act, actLogProbs, distEntropy = self.get_fifo_act(
            feature=feature,
            eval_mode=eval_mode, 
            avail_act=avail_act,
            eval_actions=eval_actions, 
            test_mode=test_mode)


        if not eval_mode:
            return act, value, actLogProbs
        else:
            return value, actLogProbs, distEntropy, ae_loss

    def chain_acts(self):
        return

    def get_fifo_act(self, feature, eval_mode, eval_actions=None, avail_act=None, test_mode=False):
        _act = []
        _act_onehot = []
        _actLogProbs = []
        _distEntropy = []
        # --- feature ----- |--> feature
        #                   |--> feature + _act_onehot ...
        #                   |--> feature + _act_onehot ... ...
        #                   |--> feature + _act_onehot ... ... ...
        for op_index, _ in enumerate(['operator 1', 'operator 2', 'operator 3', 'operator 4']):
            # 
            if len(_act) > 0:
                prev_op_act_onehot = torch.cat(_act_onehot, -1)
                feature_final_layer = torch.cat((feature, prev_op_act_onehot), -1)
            else:
                feature_final_layer = feature
            # final layer
            act_logits = self.fifo_net[f'operator {op_index+1}'](feature_final_layer)

            # settle avail action
            n_act = self.NetDim[f'operator {op_index+1} output size']
            avail_act_this_op = avail_act[:, op_index, :n_act]
            if avail_act_this_op is not None: 
                act_logits = torch.where(avail_act_this_op>0, act_logits, -pt_inf())

            # go sample
            act_dist = Categorical(logits=act_logits)
            if not test_mode:
                act = act_dist.sample() if not eval_mode else eval_actions[:, op_index]
            else: 
                act = torch.argmax(act_dist.probs, axis=-1)
            distEntropy = act_dist.entropy().mean()
            _act.append(act)
            _act_onehot.append(torch.nn.functional.one_hot(act, act_logits.shape[-1]))
            _actLogProbs.append(self._get_act_log_probs(act_dist, act))
            _distEntropy.append(distEntropy)

        _act = torch.stack(_act, dim=1)
        _actLogProbs = torch.stack(_actLogProbs, dim=1)
        _distEntropy = torch.stack(_distEntropy).mean()
        return _act, _actLogProbs, _distEntropy

    def autoencoder(self, central_enc, agent_final_emb, target_final_emb, cluster_final_emb):
        self.n_thread = central_enc.shape[0]
        
        agent_src = repeat_at(torch.eye(self.n_agent, device=central_enc.device), 0, self.n_thread)
        target_src = repeat_at(torch.eye(self.n_target, device=central_enc.device), 0, self.n_thread)
        cluster_src = repeat_at(torch.eye(self.n_cluster, device=central_enc.device), 0, self.n_thread)

        agent_src = self.attns['ae_agent_pre'](agent_src)
        target_src = self.attns['ae_target_pre'](target_src)
        cluster_src = self.attns['ae_cluster_pre'](cluster_src)

        ae_agent_decoded = self._attention(src=agent_src, passive=central_enc, attn_key='ae_agent', skip_connect=False)
        ae_target_decoded = self._attention(src=target_src, passive=central_enc, attn_key='ae_target', skip_connect=False)
        ae_cluster_decoded = self._attention(src=cluster_src, passive=central_enc, attn_key='ae_cluster', skip_connect=False)

        mask_agent = (agent_final_emb==0) | (agent_final_emb==1) | (agent_final_emb==-1)
        mask_target = (target_final_emb==0) | (target_final_emb==1) | (target_final_emb==-1)
        mask_cluster = (cluster_final_emb==0) | (cluster_final_emb==1) | (cluster_final_emb==-1)
        
        loss  = F.mse_loss(ae_agent_decoded[mask_agent],      agent_final_emb[mask_agent])
        loss += F.mse_loss(ae_target_decoded[mask_target],    target_final_emb[mask_target])
        loss += F.mse_loss(ae_cluster_decoded[mask_cluster],  cluster_final_emb[mask_cluster])

        return loss

    def get_feature(self, all_emb, eval_mode):
        # 1. normaliza all input
        agent_final_emb = self._batch_norm_agent(all_emb['agent_final_emb'])
        target_final_emb = self._batch_norm_target(all_emb['target_final_emb'])
        cluster_final_emb = all_emb['cluster_final_emb']

        # 2. Fc-> ReLu -> Fc
        agent_enc   = self.agent_enter_encoder(agent_final_emb)
        target_enc  = self.target_enter_encoder(target_final_emb)
        cluster_enc = self.cluster_enter_encoder(cluster_final_emb)

        # 3. attention, ac=agent-clustering, ct=cluster-targeting
        ac_attn_enc = self._attention(src=cluster_enc, passive=agent_enc, attn_key='right')
        ct_attn_enc = self._attention(src=cluster_enc, passive=target_enc, attn_key='left')

        # 4. central enc
        central_enc = torch.cat(tensors=(ac_attn_enc, ct_attn_enc), dim=2)
        central_enc = self.central_zip(central_enc)

        # 4-2. auto encoder restore
        if eval_mode:
            ae_loss = self.autoencoder(central_enc, agent_final_emb, target_final_emb, cluster_final_emb)
        else:
            ae_loss = None

        # 5. value
        value = self._get_final_value(central_enc)

        # 6. value
        feature  = self._merge_final_feature(central_enc)

        return feature, value, ae_loss

    def _get_act_log_probs(self, distribution, action):
        return distribution.log_prob(action.squeeze(-1)).unsqueeze(-1)

    def _attention(self, src, passive, attn_key, skip_connect=True):
        src_q = src
        passive_k = passive
        passive_v = passive
        src_attn_res = self.attns[attn_key](q=src_q, k=passive_k, v=passive_v)
        if skip_connect:
            src_attn_res = torch.cat(tensors=(src, src_attn_res), dim=2)
        return src_attn_res

    def _get_final_value(self, enc):
        enc = self.value_head_top['centralize'](enc)
        enc = my_view(x=enc, shape=[0, -1])
        value = self.value_head_top['nonlin'](enc)
        return value

    def _merge_final_feature(self, top_attn_enc):
        enc = my_view(x=top_attn_enc, shape=[0, -1]) # 0 for dim unchanged, -1 for auto calculation
        enc = self.downsample['final_nonlin'](enc)
        return enc


class MultiHeadAttention(nn.Module):
    # taken from https://github.com/wouterkool/attention-tsp/blob/master/graph_encoder.py
    def __init__(
            self,
            n_heads,
            input_dim,
            embed_dim=None,
            val_dim=None,
            key_dim=None
    ):
        super(MultiHeadAttention, self).__init__()

        if val_dim is None:
            assert embed_dim is not None, "Provide either embed_dim or val_dim"
            val_dim = embed_dim // n_heads
        if key_dim is None:
            key_dim = val_dim

        self.n_heads = n_heads
        self.input_dim = input_dim
        self.embed_dim = embed_dim
        self.val_dim = val_dim
        self.key_dim = key_dim

        self.norm_factor = 1 / math.sqrt(key_dim)  # See Attention is all you need

        self.W_query = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_key = nn.Parameter(torch.Tensor(n_heads, input_dim, key_dim))
        self.W_val = nn.Parameter(torch.Tensor(n_heads, input_dim, val_dim))

        if embed_dim is not None:
            self.W_out = nn.Parameter(torch.Tensor(n_heads, key_dim, embed_dim))

        self.init_parameters()

    def init_parameters(self):

        for param in self.parameters():
            stdv = 1. / math.sqrt(param.size(-1))
            param.data.uniform_(-stdv, stdv)

    def forward(self, q, k=None, v=None, mask=None, return_attn=False):
        """
        :param q: queries (batch_size, n_query, input_dim)
        :param k: data (batch_size, n_key, input_dim)
        :param mask: mask (batch_size, n_query, graph_size) or viewable as that (i.e. can be 2 dim if n_query == 1)
        Mask should contain 1 if attention is not possible (i.e. mask is negative adjacency)
        :return:
        """
        if k is None:
            k = q  # compute self-attention
        if v is None:
            v = k
        # k should be (batch_size, graph_size, input_dim)
        batch_size, graph_size, input_dim = k.size()
        n_query = q.size(1)
        assert q.size(0) == batch_size
        assert q.size(2) == input_dim
        assert input_dim == self.input_dim, "Wrong embedding dimension of input"

        kflat = k.contiguous().view(-1, input_dim)
        qflat = q.contiguous().view(-1, input_dim)
        vflat = v.contiguous().view(-1, input_dim)

        # last dimension can be different for keys and values
        shp = (self.n_heads, batch_size, graph_size, -1)
        shp_q = (self.n_heads, batch_size, n_query, -1)

        # Calculate queries, (n_heads, n_query, graph_size, key/val_size)
        Q = torch.matmul(qflat, self.W_query).view(shp_q)
        # Calculate keys and values (n_heads, batch_size, graph_size, key/val_size)
        K = torch.matmul(kflat, self.W_key).view(shp)
        V = torch.matmul(vflat, self.W_val).view(shp)

        # Calculate compatibility (n_heads, batch_size, n_query, graph_size)
        compatibility = self.norm_factor * torch.matmul(Q, K.transpose(2, 3))
        # Optionally apply mask to prevent attention
        if mask is not None:
            mask = mask.view(1, batch_size, n_query, graph_size).expand_as(compatibility)
            compatibility[mask.bool()] = -math.inf

        attn = F.softmax(compatibility, dim=-1)

        # If there are nodes with no neighbours then softmax returns nan so we fix them to 0
        if mask is not None:
            attnc = attn.clone()
            attnc[mask.bool()] = 0
            attn = attnc

        heads = torch.matmul(attn, V)

        out = torch.mm(
            heads.permute(1, 2, 0, 3).contiguous().view(-1, self.n_heads * self.val_dim),
            self.W_out.view(-1, self.embed_dim)
        ).view(batch_size, n_query, self.embed_dim)

        if return_attn:
            return out, attn
        return out


class LinearFinal(nn.Module):

    __constants__ = ['in_features', 'out_features']
    in_features: int
    out_features: int
    weight: torch.Tensor

    def __init__(self, in_features: int, out_features: int, bias: bool = True) -> None:
        super(LinearFinal, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = nn.Parameter(torch.Tensor(out_features, in_features))
        if bias:
            self.bias = nn.Parameter(torch.Tensor(out_features))
        else:
            self.register_parameter('bias', None)

    def forward(self, input: torch.Tensor) -> torch.Tensor:
        return F.linear(input, self.weight, self.bias)

    def extra_repr(self) -> str:
        return 'in_features={}, out_features={}, bias={}'.format(
            self.in_features, self.out_features, self.bias is not None
        )

