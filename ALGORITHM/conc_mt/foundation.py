import os, torch, json, time
import numpy as np
from UTIL.colorful import *
from .net import Net
from config import GlobalConfig
from UTIL.tensor_ops import __hash__, repeat_at
from ALGORITHM.commom.rl_alg_base import RLAlgorithmBase
class AlgorithmConfig:
    '''
        AlgorithmConfig: This config class will be 'injected' with new settings from json.
        (E.g., override configs with ```python main.py --cfg example.jsonc```)
        (please see UTIL.config_args to find out how this advanced trick works out.)
    '''
    # configuration, open to jsonc modification
    gamma = 0.99
    tau = 0.95
    train_traj_needed = 512
    upper_training_epoch = 4
    load_checkpoint = False
    checkpoint_reload_cuda = False
    TakeRewardAsUnity = False
    use_normalization = True
    add_prob_loss = False
    alternative_critic = False

    n_focus_on = 2
    turn_off_threat_est = False

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

    # sometimes the episode length gets longer,
    # resulting in more samples and causing GPU OOM,
    # prevent this by fixing the number of samples to initial
    # by randomly sampling and droping
    prevent_batchsize_oom = True
    gamma_in_reward_forwarding = False
    gamma_in_reward_forwarding_value = 0.99

    # extral
    extral_train_loop = False
    actor_attn_mod = False
    load_specific_checkpoint = ''
    dual_conc = True
    use_my_attn = True
    alternative_critic = False

    experimental_rmDeadSample = False
    experimental_useApex = False

    device_override = "no-override"
    gpu_fraction_override = 1.0
    gpu_party_override = "no-override"
    gpu_ensure_safe = False

def override_cuda_settings(AlgorithmConfig):
    # change Local cuda settings according to AlgorithmConfig
    if AlgorithmConfig.device_override != "no-override":
        assert GlobalConfig.device == 'cpu', "please set GlobalConfig.device=cpu if you want to use different GPUs for different teams"
        # reflesh the cuda setting inherited from main.py
        GlobalConfig.device = AlgorithmConfig.device_override
        GlobalConfig.gpu_fraction = AlgorithmConfig.gpu_fraction_override
        from main import pytorch_gpu_init; pytorch_gpu_init(GlobalConfig)
        # reflesh the cached cuda setting in tensor_ops
        from UTIL.tensor_ops import cuda_cfg; cuda_cfg.read_cfg()
    if AlgorithmConfig.device_override != "no-override":
        GlobalConfig.gpu_party = AlgorithmConfig.gpu_party_override

class ReinforceAlgorithmFoundation(RLAlgorithmBase):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        override_cuda_settings(AlgorithmConfig)
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.act_space = space['act_space']
        self.obs_space = space['obs_space']
        self.team = team
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        n_actions = GlobalConfig.ScenarioConfig.n_actions
        from .shell_env import ShellEnvWrapper
        self.shell_env = ShellEnvWrapper(n_agent, n_thread, space, mcv, self, 
                                        AlgorithmConfig, self.ScenarioConfig)
        
        if 'm-cuda' in GlobalConfig.device: assert False, ('not support anymore')
        else: self.device = GlobalConfig.device

        self.policy = Net(rawob_dim=self.ScenarioConfig.obs_vec_length,
                          n_action = n_actions, 
                          use_normalization=AlgorithmConfig.use_normalization,
                          n_focus_on = AlgorithmConfig.n_focus_on, 
                          actor_attn_mod=AlgorithmConfig.actor_attn_mod,
                          dual_conc=AlgorithmConfig.dual_conc)
        self.policy = self.policy.to(self.device)

        self.AvgRewardAgentWise = AlgorithmConfig.TakeRewardAsUnity
        from .ppo import PPO
        self.trainer = PPO(self.policy, cfg=AlgorithmConfig, mcv=mcv, team=self.team)
        from .trajectory import BatchTrajManager
        self.batch_traj_manager = BatchTrajManager(n_env=n_thread,
                                                   traj_limit=int(self.ScenarioConfig.MaxEpisodeStep), 
                                                   trainer_hook=self.trainer.train_on_traj)
                # confirm that reward method is correct
        self.check_reward_type(AlgorithmConfig)

        # load checkpoints
        self.load_checkpoint = AlgorithmConfig.load_checkpoint
        logdir = GlobalConfig.logdir
        # makedirs if not exists
        if not os.path.exists(f'{logdir}/history_cpt/'):
            os.makedirs(f'{logdir}/history_cpt/')
        if self.load_checkpoint:
            self.save_or_load('load', None)

        # data integraty check
        self._unfi_frag_ = None
        # Skip currupt data integraty check after this patience is exhausted
        self.patience = 1000

    # _____________________Redirection____________________
    # this is a redirect to shell_env.interact_with_env
    # self.interact_with_env(from here) --> shell_env.interact_with_env --> self.interact_with_env_genuine
    def interact_with_env(self, StateRecall):
        return self.shell_env.interact_with_env(StateRecall)

    def interact_with_env_genuine(self, StateRecall):
        test_mode = StateRecall['Test-Flag']
        if not test_mode: self.train()
        return self.action_making(StateRecall, test_mode) # StateRecall dictionary will preserve states for next action making


    def action_making(self, StateRecall, test_mode):
        assert StateRecall['obs'] is not None, ('make sure obs is oks')

        obs, threads_active_flag = StateRecall['obs'], StateRecall['threads_active_flag']
        assert len(obs) == sum(threads_active_flag), ('make sure we have the right batch of obs')
        avail_act = StateRecall['avail_act'] if 'avail_act' in StateRecall else None
        # make decision
        with torch.no_grad():
            action, value, action_log_prob = self.policy.act(obs, test_mode=test_mode, avail_act=avail_act)



        traj_frag = {
            '_SKIP_':        ~threads_active_flag, # thread mask
            'value':         value,
            'actionLogProb': action_log_prob,
            'obs':           obs,
            'action':        action,
        }
        if avail_act: traj_frag.update({'avail_act':  avail_act})
        if not test_mode: StateRecall['_hook_'] = self.commit_traj_frag(traj_frag, req_hook = True)
        else: StateRecall['_hook_'] = None


        return action.copy(), StateRecall

    def train(self):
        if self.batch_traj_manager.can_exec_training():  # time to start a training routine
            # print('self.decision_interval', self.decision_interval)
            tic = time.time()
            update_cnt = self.batch_traj_manager.train_and_clear_traj_pool()
            toc = time.time()
            print('训练用时:',toc-tic)
            self.save_or_load('save', update_cnt)



    def save_or_load(self, command, update_cnt):
        logdir = GlobalConfig.logdir
        if command == 'save':
            pt_path = f'{logdir}/{self.team}-model.pt'
            # dir 1
            print绿('saving model to %s'%pt_path)
            torch.save(self.policy.state_dict(), pt_path)
            # dir 2
            pt_path = f'{logdir}/history_cpt/{self.team}-model_{update_cnt}.pt'
            torch.save(self.policy.state_dict(), pt_path)
            print绿('save model finish')
        if command == 'load':
            manual_dir = AlgorithmConfig.load_specific_checkpoint
            ckpt_dir = f'{logdir}/{self.team}-model.pt' if manual_dir=='' else '%s/%s'%(logdir, manual_dir)
            print黄('加载检查点:', ckpt_dir)
            if not AlgorithmConfig.checkpoint_reload_cuda:
                self.policy.load_state_dict(torch.load(ckpt_dir))
            else:
                self.policy.load_state_dict(torch.load(ckpt_dir, map_location=self.device))






