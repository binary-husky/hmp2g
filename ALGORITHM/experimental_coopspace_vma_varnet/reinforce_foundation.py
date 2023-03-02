import os, torch, copy, pickle
import numpy as np
try: 
    from numba import njit, jit
except: 
    from UTIL.tensor_ops import dummy_decorator as jit
    from UTIL.tensor_ops import dummy_decorator as njit
from UTIL.colorful import *
from .cortex import CNet
from .ppo import PPO
from .trajectory import BatchTrajManager
from .my_utils import copy_clone
from UTIL.tensor_ops import __hash__
# from ALGORITHM.common.rl_alg_base import RLAlgorithmBase

class CoopAlgConfig(object):
    g_num = 5
    max_internal_step = 1
    decision_interval = 5
    head_start_cnt = 1 # first 3 step have 
    head_start_hold_n = 1 # how many to control at first few step

    eval_mode = False

    checkpoint_reload_cuda = False
    load_checkpoint = False
    load_specific_checkpoint = ''
    one_more_container = False
    reverse_container = False
    use_fixed_random_start = True
    use_zero_start = False 
    use_empty_container = False
    use_complete_random = False

    # PPO part
    clip_param = 0.2
    ppo_epoch = 16
    n_pieces_batch_division = 32    # 8: the batch size in each ppo update is 23280; x/8 *1.5 = x/y, y=8/1.5
    value_loss_coef = 0.5
    entropy_coef = 0.01
    max_grad_norm = 0.5
    clip_param = 0.2
    lr = 1e-4
    balance = 0.5
    
    gamma = 0.99
    tau = 0.95
    # ?
    train_traj_needed = 256
    upper_training_epoch = 5
    h_reward_on_R = True
    continues_type_ceil = True
    invalid_penalty = 0.1
    upper_training_epoch = 5
    render_graph = False
    hidden_dim = 32

    sub_cluster_size = 1
    dropout_prob = 0.0

    normalization_sample_limit = 3e8



class ReinforceAlgorithmFoundation():
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        from config import GlobalConfig
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.team = team
        self.act_space = space['act_space']
        self.obs_space = space['obs_space']
        self.n_cluster = CoopAlgConfig.g_num
        self.ScenarioConfig = ScenarioConfig = GlobalConfig.ScenarioConfig
        self.note = GlobalConfig.note

        if CoopAlgConfig.render_graph:
            from VISUALIZE.mcom import mcom
            self.可视化桥 = mcom(path='TEMP/v2d_logger/', draw_mode='Threejs')
            self.可视化桥.初始化3D()
            self.可视化桥.设置样式('gray')
            self.可视化桥.其他几何体之旋转缩放和平移('box', 'BoxGeometry(1,1,1)',   0,0,0,  1,1,1, 0,0,0) 

        self.n_basic_dim = ScenarioConfig.obs_vec_length
        self.n_entity = ScenarioConfig.num_entity
        self.ObsAsUnity = False
        if hasattr(ScenarioConfig, 'ObsAsUnity'):
            self.ObsAsUnity = ScenarioConfig.ObsAsUnity
        self.agent_uid = ScenarioConfig.uid_dictionary['agent_uid']
        self.target_uid = ScenarioConfig.uid_dictionary['entity_uid']

        self.pos_decs = ScenarioConfig.obs_vec_dictionary['pos']
        self.vel_decs = ScenarioConfig.obs_vec_dictionary['vel']
        self.max_internal_step = CoopAlgConfig.max_internal_step
        self.head_start_cnt = CoopAlgConfig.head_start_cnt
        self.decision_interval = CoopAlgConfig.decision_interval
        self.head_start_hold_n = CoopAlgConfig.head_start_hold_n
        self.sub_cluster_size = CoopAlgConfig.sub_cluster_size

        self.device = GlobalConfig.device
        cuda_n = 'cpu' if 'cpu' in self.device else GlobalConfig.device

        self.policy = CNet(num_agents=self.n_agent//self.sub_cluster_size, num_entities=self.n_entity, basic_vec_len=self.n_basic_dim, hidden_dim=CoopAlgConfig.hidden_dim).to(self.device)
        self.trainer = PPO(self.policy, mcv=mcv)

        self.batch_traj_manager = BatchTrajManager(n_env=n_thread, traj_limit=ScenarioConfig.MaxEpisodeStep*3, trainer_hook=self.trainer.train_on_traj)

        self.load_checkpoint = CoopAlgConfig.load_checkpoint
        self.cnt = 0

        self.logdir = GlobalConfig.logdir
        if not os.path.exists('%s/history_cpt/'%self.logdir): os.makedirs('%s/history_cpt/'%self.logdir)
        if self.load_checkpoint:
            manual_dir = CoopAlgConfig.load_specific_checkpoint
            ckpt_dir = '%s/model.pt'%self.logdir if manual_dir=='' else '%s/%s'%(self.logdir, manual_dir)
            print黄('加载检查点:', ckpt_dir)
            self.policy.load_state_dict(torch.load(ckpt_dir, map_location=cuda_n))
        #################### ?? ####################
        t = [int(np.ceil(self.max_internal_step)) if x<self.head_start_cnt  else 1 if x%self.decision_interval==0 else 0
                for x in range(50)]
        print('control_squence:', t)
        print('hold_squence:', [int(np.ceil(self.head_start_hold_n / 4**x )) if x<self.head_start_cnt  else 1  for x in range(50)])
        self.patience = 500 # skip currupt data detection after patience exhausted

    def interact_with_env(self, State_Recall):
        self.train()
        return self.action_making(State_Recall) # state_recall dictionary will preserve states for next action making

    def train(self):
        if self.batch_traj_manager.can_exec_training():  # time to start a training routine
            update_cnt = self.batch_traj_manager.train_and_clear_traj_pool()
            self.save_model(update_cnt)

    def get_internal_step(self, n_step):

        n_internal_step = [np.ceil(self.max_internal_step) if x<self.head_start_cnt 
                                else 1.0 if x%self.decision_interval==0 else 0.0  for x in n_step]  
        n_internal_step = np.array(n_internal_step, dtype=int)

        hold_n = [np.ceil(self.head_start_hold_n / 4**x ) if x<self.head_start_cnt  else 1.0  for x in n_step]      
        hold_n = np.array(hold_n, dtype=int)

        return n_internal_step, hold_n

    def action_making(self, State_Recall):
        # test mode
        test_mode = State_Recall['Test-Flag']
        active_env = ~State_Recall['ENV-PAUSE']
        # read loop back
        if '_graphstate_' in State_Recall:
            self.coopgraph = State_Recall['_graphstate_']
        else:
            from .coopgraph import CoopGraph
            self.coopgraph = CoopGraph(
                self.n_agent, self.n_thread, 
                n_entity = self.n_entity, 
                load_checkpoint = self.load_checkpoint,
                ObsAsUnity = self.ObsAsUnity,
                agent_uid = self.agent_uid,
                sub_cluster_size = self.sub_cluster_size,
                pos_decs = self.pos_decs,
                vel_decs = self.vel_decs,
                target_uid = self.target_uid,
                test_mode = test_mode,
                logdir = self.logdir,
                n_basic_dim = self.n_basic_dim,
                )
            
        self.coopgraph.reset_terminated_threads(just_got_reset=State_Recall['Env-Suffered-Reset'])

        raw_obs = copy_clone(State_Recall['Latest-Obs'])
        co_step = copy_clone(State_Recall['Current-Obs-Step'])


        # ________RL_Policy_Core_______
        thread_internal_step_o,  hold_n_o = self.get_internal_step(State_Recall['Current-Obs-Step'])
        thread_internal_step = thread_internal_step_o
        iter_n = np.max(thread_internal_step)

        # disturb graph functioning
        LIVE = active_env & (thread_internal_step > 0)
        
        if not test_mode:
            self.coopgraph.random_disturb(prob=CoopAlgConfig.dropout_prob, mask=LIVE)

        if CoopAlgConfig.render_graph:
            self.coopgraph.render_thread0_graph(可视化桥=self.可视化桥, step=State_Recall['Current-Obs-Step'][0], internal_step=thread_internal_step[0])
        
        for _ in range(iter_n):
            LIVE = active_env & (thread_internal_step > 0)
            if not LIVE.any(): continue

            # hold_n = hold_n_o[LIVE]
            Active_emb, Active_act_dec = self.coopgraph.attach_encoding_to_obs_masked(raw_obs, LIVE)
            avail_act = self.coopgraph.get_avail_act(LIVE)
            # _, _, Active_cter_fifoR, Active_cter_fifoL = self.coopgraph.get_graph_encoding_masked(LIVE)

            with torch.no_grad():
                act, value, actLogProbs = self.policy.act(Active_emb, avail_act=avail_act, test_mode=test_mode)
            traj_frag = {   'skip':         ~LIVE,           
                            'done':         False,
                            'act':          act,
                            'obs':        Active_emb, 
                            'avail_act':  avail_act,
                            'value':            value,
                            'actLogProbs':          actLogProbs,
                            'reward':           np.array([0.0 for _ in range(self.n_thread)])}
            # _______Internal_Environment_Step________
            self.coopgraph.adjust_edge(act, mask=LIVE, hold=hold_n_o)

            if CoopAlgConfig.render_graph:
                self.coopgraph.render_thread0_graph(可视化桥=self.可视化桥, step=State_Recall['Current-Obs-Step'][0], internal_step=thread_internal_step[0])
            
            if not test_mode: self.batch_traj_manager.feed_traj(traj_frag, require_hook=False)
            thread_internal_step = thread_internal_step - 1

        traj_frag = {
            'skip': copy_clone(State_Recall['ENV-PAUSE']), 
            'obs': None, 
            'value': None,
            'act': None,
            'avail_act': None,
            'actLogProbs': None, 
        }

        _, act_dec = self.coopgraph.attach_encoding_to_obs_masked(raw_obs, np.array([True]*self.n_thread))
        agent_target_link = self.coopgraph.link_agent_to_target()
        all_action = self.coopgraph.target_micro_control(agent_target_link, act_dec)


        actions_list = []
        for i in range(self.n_agent): actions_list.append(all_action[:,i,:])
        actions_list = np.array(actions_list)

        # return necessary handles to main platform
        wait_reward_hook = self.commit_frag_hook(traj_frag, require_hook = True) if not test_mode else self.__dummy_hook
        # traj_hook = self.batch_traj_manager.feed_traj(traj_frag, require_hook=True) if not test_mode else self.__dummy_hook
        State_Recall['_hook_'] = wait_reward_hook  # leave a hook to grab the reward signal just a moment later
        State_Recall['_graphstate_'] = copy.deepcopy(self.coopgraph)
        self.coopgraph = None
        # if self.cold_start: self.cold_start = False
        return actions_list, State_Recall # state_recall dictionary will preserve states for next action making

    def commit_frag_hook(self, f1, require_hook = True):
        if not hasattr(self, '__incomplete_frag__'): self.__incomplete_frag__ = None
        assert self.__incomplete_frag__ is None
        self.__incomplete_frag__ = f1
        self.__check_data_hash() # this is important!
        if require_hook: return lambda f2: self.rollout_frag_hook(f2) # leave hook
        return

    def 目标解析(self, link_indices, act_dec):
        target_pos, agent_pos, target_vel = (act_dec['target_pos'], act_dec['agent_pos'], act_dec['target_vel'])

        if not CoopAlgConfig.reverse_container:
            final_sel_pos = target_pos
        else:   # 为没有装入任何entity的container解析一个nan动作
            final_sel_pos = np.concatenate( (target_pos,  np.zeros(shape=(self.n_thread, 1, 3))+np.nan ) , axis=1)
            
        sel_target_pos  = np.take_along_axis(final_sel_pos, axis=1, indices=link_indices)  # 6 in final_indices /cluster_target_div
        sel_target_vel  = np.take_along_axis(target_vel, axis=1, indices=link_indices)  # 6 in final_indices /cluster_target_div
        delta_pos = sel_target_pos - agent_pos
        return delta_pos, sel_target_vel


    def save_model(self, update_cnt):
        if update_cnt!=0 and update_cnt%200==0:
            print绿('保存模型中')
            torch.save(self.policy.state_dict(), '%s/history_cpt/model%d.pt'%(self.logdir, update_cnt))
            torch.save(self.policy.state_dict(), '%s/model.pt'%(self.logdir))
            print绿('保存模型完成')






    # debugging functions
    def __check_data_hash(self):
        if self.patience > 0: 
            self.hash_debug = {}
            # for debugging, to detect write protection error
            for key in self.__incomplete_frag__:
                item = self.__incomplete_frag__[key]
                if isinstance(item, dict):
                    self.hash_debug[key]={}
                    for subkey in item:
                        subitem = item[subkey]
                        self.hash_debug[key][subkey] = __hash__(subitem)
                else:
                    self.hash_debug[key] = __hash__(item)

    def __check_data_curruption(self):
        if self.patience > 0: 
            assert self.__incomplete_frag__ is not None
            assert self.hash_debug is not None
            for key in self.__incomplete_frag__:
                item = self.__incomplete_frag__[key]
                if isinstance(item, dict):
                    for subkey in item:
                        subitem = item[subkey]
                        assert self.hash_debug[key][subkey] == __hash__(subitem), ('Currupted data! 发现腐败数据!')
                else:
                    assert self.hash_debug[key] == __hash__(item), ('Currupted data! 发现腐败数据!')
            self.patience -= 1
    def rollout_frag_hook(self, f2):
        '''   <2>  hook is called, reward and next moment observation is ready,
                        now feed them into trajectory manager    '''
        # do data curruption check at beginning, this is important!
        self.__check_data_curruption()
        # put the fragment into memory
        self.__incomplete_frag__.update(f2)
        __completed_frag = self.__incomplete_frag__
        __completed_frag.pop('info')
        __completed_frag.pop('Latest-Obs')
        __completed_frag.pop('Terminal-Obs-Echo')
        self.batch_traj_manager.feed_traj(__completed_frag, require_hook=False)
        self.__incomplete_frag__ = None

    def __dummy_hook(self, f2): 
        return
