import os, torch, json, time
import numpy as np
try: 
    from numba import njit, jit
except: 
    from UTIL.tensor_ops import dummy_decorator as jit
    from UTIL.tensor_ops import dummy_decorator as njit
from UTIL.colorful import *
from .net import Net
from config import GlobalConfig
from UTIL.tensor_ops import __hash__, pad_vec_array, copy_clone, my_view

class AlgorithmConfig:
    gamma = 0.99
    tau = 0.95
    train_traj_needed = 512
    upper_training_epoch = 4
    load_checkpoint = False
    checkpoint_reload_cuda = False
    TakeRewardAsUnity = False
    use_normalization = True
    agent_wise_attention = True
    seperate_critic = False
    add_prob_loss = False

    # PPO part
    clip_param = 0.2
    ppo_epoch = 16
    n_pieces_batch_division = 1    # 8: the batch size in each ppo update is 23280; x/8 *1.5 = x/y, y=8/1.5
    value_loss_coef = 0.1
    entropy_coef = 0.05
    max_grad_norm = 0.5
    clip_param = 0.2
    lr = 1e-4
    balance = 0.5

class ReinforceAlgorithmFoundation(object):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.act_space = space['act_space']
        self.obs_space = space['obs_space']
        ScenarioConfig = GlobalConfig.ScenarioConfig
        alg_config = AlgorithmConfig
        from .shell_env import ShellEnvWrapper
        self.shell_env = ShellEnvWrapper(n_agent, n_thread, space, mcv, self, 
                                        alg_config, ScenarioConfig)
        # if 'm-cuda' in GlobalConfig.device:
        #     gpu_id = json.loads(GlobalConfig.device.split('->')[-1])
        #     device = 'cuda:%d'%gpu_id[0]
        #     assert False, ('not yet support')
        # else:
        #     gpu_id = None
        #     device = GlobalConfig.device
        #     cuda_n = 'cpu' if 'cpu' in device else GlobalConfig.device
        # self.policy = Net(num_agents=self.n_agent, n_basic_dim=ScenarioConfig.obs_vec_length,
        #                   obs_dim = 15*6*2, n_action = 7, use_m_gpu=gpu_id, 
        #                   use_normalization=alg_config.use_normalization,
        #                   seperate_critic=alg_config.seperate_critic)

        # self.device = device
        # self.policy = self.policy.to(self.device)

        # self.AvgRewardAgentWise = alg_config.TakeRewardAsUnity
        # from .ppo import PPO
        # self.trainer = PPO(self.policy, ppo_config=AlgorithmConfig, mcv=mcv)
        # from .trajectory import BatchTrajManager
        # self.batch_traj_manager = BatchTrajManager(n_env=n_thread,
        #                                            traj_limit=int(ScenarioConfig.MaxEpisodeStep), 
        #                                            trainer_hook=self.trainer.train_on_traj,
        #                                            mcv=mcv)
        # self.load_checkpoint = AlgorithmConfig.load_checkpoint
        # logdir = GlobalConfig.logdir
        # if not os.path.exists('%s/history_cpt/'%logdir):
        #     os.makedirs('%s/history_cpt/'%logdir)
        # if self.load_checkpoint:
        #     print黄('加载检查点')
        #     if not alg_config.checkpoint_reload_cuda:
        #         self.policy.load_state_dict(torch.load('%s/model.pt'%logdir))
        #     else:
        #         self.policy.load_state_dict(torch.load('%s/model.pt'%logdir, map_location=cuda_n))
        self.__incomplete_frag__ = None
        self.patience = 500 # skip currupt data detection after patience exhausted

    # _____________________Redirection____________________
    # this is a redirect to shell_env.interact_with_env
    # self.interact_with_env(from here) --> shell_env.interact_with_env --> self.interact_with_env_genuine
    def interact_with_env(self, State_Recall):
        return self.shell_env.interact_with_env(State_Recall)

    def interact_with_env_genuine(self, State_Recall):
        test_mode = State_Recall['Test-Flag']
        if not test_mode: self.train()
        return self.action_making(State_Recall, test_mode) # state_recall dictionary will preserve states for next action making


    def action_making(self, State_Recall, test_mode):
        assert State_Recall['obs'] is not None, ('make sure obs is oks')

        obs, threads_active_flag = State_Recall['obs'], State_Recall['threads_active_flag']
        assert len(obs) == sum(threads_active_flag), ('make sure we have the right batch of obs')

        # make decision
        with torch.no_grad():
            action, value, action_log_prob = self.policy.act(obs, test_mode=test_mode)


        wait_reward_hook = self.commit_frag_hook({
            '_SKIP_':        ~threads_active_flag, # thread mask
            'value':         value,
            'actionLogProb': action_log_prob,
            'obs':           obs,
            'action':        action,
        }, require_hook = True) if not test_mode else self.__dummy_hook
        '''   <1>  we will deal with rollout later after the reward is ready, 
                        now we leave a hook to be callback    '''
        State_Recall['_hook_'] = wait_reward_hook
        return action.copy(), State_Recall


    def commit_frag_hook(self, f1, require_hook = True):
        assert self.__incomplete_frag__ is None
        self.__incomplete_frag__ = f1
        self.__check_data_hash() # this is important!
        if require_hook: return lambda f2: self.rollout_frag_hook(f2) # leave hook
        return


    # ________Rollout Processor_______
    def rollout_frag_hook(self, f2):
        '''   <2>  hook is called, reward and next moment observation is ready,
                        now feed them into trajectory manager    '''
        # do data curruption check at beginning, this is important!
        self.__check_data_curruption()
        # strip info, since it is not array
        items_to_pop = ['info', 'Latest-Obs']
        strip_dict = {}
        for k in items_to_pop:
            if k in f2: 
                strip_dict[k] = f2.pop(k)
        # the agent-wise reward is supposed to be the same, so averge them
        if self.AvgRewardAgentWise: 
            f2['reward'] = np.mean(f2['reward'], axis=-1, keepdims=True)
        # change the name of done to be recognised (by trajectory manager)
        f2['_DONE_'] = f2.pop('done')
        f2['_TOBS_'] = f2.pop('Terminal-Obs-Echo') if 'Terminal-Obs-Echo' in f2 else None
        # integrate frag part1 and part2
        self.__incomplete_frag__.update(f2)
        self.__completed_frag = self.mask_paused_env(self.__incomplete_frag__)
        # put the fragment into memory
        self.batch_traj_manager.feed_traj(self.__completed_frag)
        self.__incomplete_frag__ = None


    def mask_paused_env(self, fragment):
        running = ~fragment['_SKIP_']
        if running.all():
            return fragment
        for key in fragment:
            if not key.startswith('_') and hasattr(fragment[key], '__len__') and len(fragment[key]) == self.n_thread:
                fragment[key] = fragment[key][running]
        return fragment

    def train(self):
        if self.batch_traj_manager.can_exec_training():  # time to start a training routine
            # print('self.decision_interval', self.decision_interval)
            tic = time.time()
            update_cnt = self.batch_traj_manager.train_and_clear_traj_pool()
            toc = time.time()
            print('训练用时:',toc-tic)
            self.__save_model(update_cnt)





    def __dummy_hook(self, f2): 
        return

    def __save_model(self, update_cnt):
        logdir = GlobalConfig.logdir
        flag = '%s/save_now'%logdir
        if os.path.exists(flag) or update_cnt%50==0:
            # dir 1
            pt_path = '%s/model.pt'%logdir
            print绿('saving model to %s'%pt_path)
            torch.save(self.policy.state_dict(), pt_path)

            # dir 2
            pt_path = '%s/history_cpt/model_%d.pt'%(logdir, update_cnt)
            torch.save(self.policy.state_dict(), pt_path)
            try: os.remove(flag)
            except: pass
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
