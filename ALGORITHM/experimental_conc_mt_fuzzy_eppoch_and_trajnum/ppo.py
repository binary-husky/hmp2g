import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
from UTIL.colorful import *
from UTIL.tensor_ops import _2tensor, _2cpu2numpy, repeat_at
from UTIL.tensor_ops import my_view, scatter_with_nan, sample_balance
from config import GlobalConfig
from UTIL.gpu_share import GpuShareUnit
from ALGORITHM.common.ppo_sampler import TrajPoolSampler
from .fuzzy import projection, gen_feedback_sys_generic




class PPO():
    def __init__(self, policy_and_critic, cfg, mcv=None, team=0, n_agent=None):
        self.policy_and_critic = policy_and_critic
        self.cfg = cfg
        self.clip_param = cfg.clip_param
        self.ppo_epoch = cfg.ppo_epoch
        self.n_agent = n_agent
        self.n_pieces_batch_division = cfg.n_pieces_batch_division
        self.value_loss_coef = cfg.value_loss_coef
        self.entropy_coef = cfg.entropy_coef
        self.max_grad_norm = cfg.max_grad_norm
        self.add_prob_loss = cfg.add_prob_loss
        self.prevent_batchsize_oom = cfg.prevent_batchsize_oom
        self.use_policy_resonance = cfg.use_policy_resonance
        self.preserve_history_pool = cfg.preserve_history_pool
        self.preserve_history_pool_size = cfg.preserve_history_pool_size
        self.fuzzy_controller_param = cfg.fuzzy_controller_param
        self.fuzzy_controller = cfg.fuzzy_controller
        self.fuzzy_controller_scale_param = cfg.fuzzy_controller_scale_param


        self.lr_descent = cfg.lr_descent
        self.lr_descent_coef = cfg.lr_descent_coef
        self.team = team
        self.lr = cfg.lr
        self.extral_train_loop = cfg.extral_train_loop
        self.turn_off_threat_est = cfg.turn_off_threat_est
        self.experimental_rmDeadSample = cfg.experimental_rmDeadSample
        self.gpu_ensure_safe = cfg.gpu_ensure_safe
        self.all_parameter = list(policy_and_critic.named_parameters())
        self.at_parameter = [(p_name, p) for p_name, p in self.all_parameter if 'AT_' in p_name]
        self.ct_parameter = [(p_name, p) for p_name, p in self.all_parameter if 'CT_' in p_name]
        # self.ae_parameter = [(p_name, p) for p_name, p in self.all_parameter if 'AE_' in p_name]
        # 检查剩下是是否全都是不需要训练的参数
        remove_exists = lambda LA,LB: list(set(LA).difference(set(LB)))
        res = self.all_parameter
        res = remove_exists(res, self.at_parameter)
        res = remove_exists(res, self.ct_parameter)
        # res = remove_exists(res, self.ae_parameter)
        for p_name, p in res:   
            assert not p.requires_grad, ('a parameter must belong to either CriTic or AcTor, unclassified parameter:',p_name)

        self.cross_parameter = [(p_name, p) for p_name, p in self.all_parameter if ('CT_' in p_name) and ('AT_' in p_name)]
        assert len(self.cross_parameter)==0,('a parameter must belong to either CriTic or AcTor, not both')
        # 不再需要参数名
        self.at_parameter = [p for p_name, p in self.all_parameter if 'AT_' in p_name]
        self.at_optimizer = optim.Adam(self.at_parameter, lr=self.lr)

        self.ct_parameter = [p for p_name, p in self.all_parameter if 'CT_' in p_name]
        self.ct_optimizer = optim.Adam(self.ct_parameter, lr=self.lr*10.0) #(self.lr)
        # self.ae_parameter = [p for p_name, p in self.all_parameter if 'AE_' in p_name]
        # self.ae_optimizer = optim.Adam(self.ae_parameter, lr=self.lr*100.0) #(self.lr)
        self.g_update_delayer = 0
        self.g_initial_value_loss = 0
        # 轮流训练式
        self.mcv = mcv
        self.ppo_update_cnt = 0
        self.batch_size_reminder = True
        self.trivial_dict = {}

        assert self.n_pieces_batch_division == 1
        self.n_div = 1
        # print亮红(self.n_div)
        if cfg.gpu_party_override == "no-override":
            gpu_party = GlobalConfig.gpu_party
        else:
            gpu_party = cfg.gpu_party_override

        self.gpu_share_unit = GpuShareUnit(GlobalConfig.device, gpu_party=gpu_party, gpu_ensure_safe=self.gpu_ensure_safe)

        self.experimental_useApex = cfg.experimental_useApex

        if self.fuzzy_controller:
            self.fuzzy_adjustment(wr=0.5, pool=None)
            # self.agent_adjust_factor_flat = np.ones(shape=(self.n_agent,))

    def fuzzy_adjustment(self, wr, pool):
        fuzzy_controller_param=self.fuzzy_controller_param
        scale_param=self.fuzzy_controller_scale_param
        wr_input_expand = projection(x=scale_param[0], from_range=[0,  1.0], to_range=[0,  0.5])

        ## --- <1>  --- ##
        # expected input [0, +1], 
        # expected fuzzy output [4, 28] floating
        # expected final output [4, 28] int
        def compute_output_ppo_epoch_floating(feedback_sys, winrate):
            feedback_sys.input['winrate'] = winrate
            feedback_sys.compute()
            ppo_epoch_floating = feedback_sys.output['ppo_epoch_floating']
            return int(ppo_epoch_floating), ppo_epoch_floating
        fc_1 = gen_feedback_sys_generic(
            antecedent_key='winrate',
            antecedent_min=0.5-wr_input_expand,
            antecedent_max=0.5+wr_input_expand,
            consequent_key='ppo_epoch_floating',
            consequent_min=16 - 12,
            consequent_max=16 + 12,
            consequent_num_mf=7,
            fuzzy_controller_param=fuzzy_controller_param[0:5],
            compute_fn=compute_output_ppo_epoch_floating
        )



        ## --- <2>  --- ##
        # expected input [0, +1], 
        # expected fuzzy output [1, 15] floating
        # expected final output [4, 60] int
        def compute_output_trajnum_multiplier_floating(feedback_sys, winrate):
            feedback_sys.input['winrate'] = winrate
            feedback_sys.compute()
            trajnum_multiplier_floating = feedback_sys.output['trajnum_multiplier_floating']
            return int(trajnum_multiplier_floating * 4), trajnum_multiplier_floating
        fc_2 = gen_feedback_sys_generic(
            antecedent_key='winrate',
            antecedent_min=0.5-wr_input_expand,
            antecedent_max=0.5+wr_input_expand,
            consequent_key='trajnum_multiplier_floating',
            consequent_min=8 - 7,
            consequent_max=8 + 7,
            consequent_num_mf=7,
            fuzzy_controller_param=fuzzy_controller_param[5:10],
            compute_fn=compute_output_trajnum_multiplier_floating
        )

        ppo_epoch, _ = fc_1.compute_fn(wr)
        trajnum, _ = fc_2.compute_fn(wr)

        self.cfg.ppo_epoch = ppo_epoch
        self.ppo_epoch = ppo_epoch

        self.train_traj_needed = trajnum

    def calculate_rank(self, arr):
        y = np.zeros_like(arr)
        np.put_along_axis(y, np.argsort(arr), np.arange(len(arr)), axis=-1)
        return y

    def roll_traj_pool(self, traj_pool_feedin):
        # create history_pool_arr if not exist
        if not hasattr(self, 'history_pool_arr'):
            self.history_pool_arr = []
        self.history_pool_arr.extend(traj_pool_feedin)
        
        while len(self.history_pool_arr) > 128:
            self.history_pool_arr.pop(0)

        return self.history_pool_arr

    def train_on_traj(self, traj_pool, task, progress):

        with self.gpu_share_unit:
            self.train_on_traj_(traj_pool, task) 

        if self.fuzzy_controller:
            traj_recent128 = self.roll_traj_pool(traj_pool_feedin = traj_pool)

            wr = np.array([t.win for t in traj_recent128]).mean()
            self.fuzzy_adjustment(wr=wr, pool=traj_pool)
            

    def train_on_traj_(self, traj_pool, task):
        ppo_valid_percent_list = []
        req_dict = ['obs', 'action', 'actionLogProb', 'return', 'reward', 'threat',  'value']
        if self.preserve_history_pool:
            req_dict += ['current_agent_filter',]
        if self.use_policy_resonance:
            req_dict += ['eprsn', 'randl']
        sampler = TrajPoolSampler(
            n_div=self.n_div, traj_pool=traj_pool, flag=task, 
            req_dict = req_dict,
            return_rename = 'return', 
            value_rename = 'value', 
            advantage_rename = 'advantage',
            exclude_eprsn_in_norm = False,
            prevent_batchsize_oom=self.prevent_batchsize_oom)
        assert self.n_div == len(sampler)
        for e in range(self.ppo_epoch):
            # print亮紫('pulse')
            sample_iter = sampler.reset_and_get_iter()
            self.at_optimizer.zero_grad(); self.ct_optimizer.zero_grad()
            for i in range(self.n_div):
                # ! get traj fragment
                sample = next(sample_iter)
                # ! build graph, then update network!
                # self.ae_optimizer.zero_grad()
                loss_final, others = self.establish_pytorch_graph(task, sample, e)
                loss_final = loss_final*0.5 /self.n_div
                if (e+i)==0:
                    print('[PPO.py] Memory Allocated %.2f GB'%(torch.cuda.memory_allocated()/1073741824))
                loss_final.backward()
                # log
                ppo_valid_percent_list.append(others.pop('PPO valid percent').item())
                self.log_trivial(dictionary=others); others = None
            nn.utils.clip_grad_norm_(self.at_parameter, self.max_grad_norm)
            self.at_optimizer.step(); self.ct_optimizer.step()
            if ppo_valid_percent_list[-1] < 0.70: 
                print亮黄('policy change too much, epoch terminate early'); break
        pass # finish all epoch update

        print亮黄(np.array(ppo_valid_percent_list))
        self.log_trivial_finalize()
        # print亮红('Leaky Memory Allocated %.2f GB'%(torch.cuda.memory_allocated()/1073741824))

        self.ppo_update_cnt += 1
        return self.ppo_update_cnt

    def log_trivial(self, dictionary):
        for key in dictionary:
            if key not in self.trivial_dict: self.trivial_dict[key] = []
            item = dictionary[key].item() if hasattr(dictionary[key], 'item') else dictionary[key]
            self.trivial_dict[key].append(item)

    def log_trivial_finalize(self, print=True):
        for key in self.trivial_dict:
            self.trivial_dict[key] = np.array(self.trivial_dict[key])
        
        print_buf = ['[ppo.py] ']
        for key in self.trivial_dict:
            self.trivial_dict[key] = self.trivial_dict[key].mean()
            print_buf.append(' %s:%.3f, '%(key, self.trivial_dict[key]))
            if self.mcv is not None:  self.mcv.rec(self.trivial_dict[key], key+' of=%d'%self.team)
        if print: print紫(''.join(print_buf))
        if self.mcv is not None:
            self.mcv.rec_show()
        self.trivial_dict = {}

    def rmDeadSample(self, obs, *args):
        # assert first dim as Time dim
        # assert second dim as agent dim
        fter = my_view(obs, [0,0,-1])
        fter = torch.isnan(fter).any(-1)
        to_ret = obs[fter].unsqueeze(axis=1), *(a[fter].unsqueeze(axis=1) if a is not None else None for a in args)
        return to_ret


    def establish_pytorch_graph(self, flag, sample, n):
        obs = _2tensor(sample['obs'])
        state = _2tensor(sample['state']) if 'state' in sample else None
        advantage = _2tensor(sample['advantage'])
        action = _2tensor(sample['action'])
        oldPi_actionLogProb = _2tensor(sample['actionLogProb'])
        real_value = _2tensor(sample['return'])
        real_threat = _2tensor(sample['threat'])
        avail_act = _2tensor(sample['avail_act']) if 'avail_act' in sample else None
        eprsn = _2tensor(sample['eprsn']) if 'eprsn' in sample else None

        batchsize = advantage.shape[0]#; print亮紫(batchsize)
        # if self.fuzzy_controller:
            
        #     adjust_weight_agent_wise = self.agent_adjust_factor_flat
        #     adjust_weight_agent_wise = torch.Tensor(adjust_weight_agent_wise).to(obs.device)
        #     # expand thread
        #     adjust_weight_agent_wise = repeat_at(adjust_weight_agent_wise, insert_dim=0, n_times=batchsize)
        #     # unsqueeze
        #     adjust_weight_agent_wise = repeat_at(adjust_weight_agent_wise, insert_dim=-1, n_times=1)

        batch_agent_size = advantage.shape[0]*advantage.shape[1]
        # if self.experimental_rmDeadSample:  # Warning! This operation will merge Time and Agent dim!
        #     obs, advantage, action, oldPi_actionLogProb, real_value, real_threat, avail_act = \
        #         self.rmDeadSample(obs, advantage, action, oldPi_actionLogProb, real_value, real_threat, avail_act)
        #     batch_agent_size = advantage.shape[0]


        assert flag == 'train'
        newPi_value, newPi_actionLogProb, entropy, probs, others = self.policy_and_critic.evaluate_actions(
                obs, eval_actions=action, test_mode=False, avail_act=avail_act, eprsn=eprsn, randl=None)
        
        # threat approximation
        SAFE_LIMIT = 8
        filter = (real_threat<SAFE_LIMIT) & (real_threat>=0)

        if self.preserve_history_pool:
            fltr = _2tensor(sample['current_agent_filter'])

            # || 1 filter critic 
            real_value = real_value[fltr]
            newPi_value = newPi_value[fltr]
            filter = filter.squeeze(-1) & fltr
            
            # || 2 filter actor 
            oldPi_actionLogProb = oldPi_actionLogProb[fltr]
            advantage = advantage[fltr]
            newPi_actionLogProb = newPi_actionLogProb[fltr]
            entropy = entropy[fltr]
            batch_agent_size = advantage.shape[0]


        threat_loss = F.mse_loss(others['threat'][filter], real_threat[filter])

        # Part1: policy gradient
        # dual clip ppo core
        E = newPi_actionLogProb - oldPi_actionLogProb
        E_clip = torch.zeros_like(E)
        E_clip = torch.where(advantage > 0, torch.clamp(E, max=np.log(1.0+self.clip_param)), E_clip)
        E_clip = torch.where(advantage < 0, torch.clamp(E, min=np.log(1.0-self.clip_param), max=np.log(5) ), E_clip)
        ratio = torch.exp(E_clip)
        policy_loss = -(ratio*advantage).mean()
        # if self.fuzzy_controller:
        #     entropy_loss = (entropy*adjust_weight_agent_wise.squeeze(-1)).mean()
        # else:
        entropy_loss = entropy.mean()
        AT_net_loss = policy_loss - entropy_loss*self.entropy_coef

        # Part2: critic regression
        value_loss = 0.5 * F.mse_loss(real_value, newPi_value)
        if 'motivation value' in others:
            value_loss += 0.5 * F.mse_loss(real_value, others['motivation value'])
        CT_net_loss = value_loss * 1.0 + threat_loss * 0.1 # + friend_threat_loss*0.01

        # Add all loses
        loss_final =  AT_net_loss + CT_net_loss  # + AE_new_loss

        ppo_valid_percent = ((E_clip == E).int().sum()/batch_agent_size)

        nz_mask = real_value!=0
        value_loss_abs = (real_value[nz_mask] - newPi_value[nz_mask]).abs().mean()
        others = {
            'Value loss Abs':           value_loss_abs,
            'PPO valid percent':        ppo_valid_percent,
            'threat loss':              threat_loss,
            'CT_net_loss':              CT_net_loss,
            'AT_net_loss':              AT_net_loss,
        }


        return loss_final, others

