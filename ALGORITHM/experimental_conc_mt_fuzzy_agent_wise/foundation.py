import os, torch, json, time
import numpy as np
from UTIL.colorful import *
from config import GlobalConfig
from UTIL.tensor_ops import __hash__, repeat_at
from ALGORITHM.common.rl_alg_base import RLAlgorithmBase
from ALGORITHM.common.onfly_config import ConfigOnFly
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
    gamma_in_reward_forwarding = True
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

    use_policy_resonance = False
    policy_resonance_method = 'legacy'
    ConfigOnTheFly = True

    reward_for_winrate = False
    reward_for_winrate_n_up = 1

    preserve_history_pool = False
    preserve_history_pool_size = 2

    lr_descent = False
    lr_descent_coef = 2
    fuzzy_controller = False
    fuzzy_controller_param = [2,2,2,2,2]
    fuzzy_controller_scale_param = [0.5]

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

class ReinforceAlgorithmFoundation(RLAlgorithmBase, ConfigOnFly):
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

        # self.policy = Net(rawob_dim=self.ScenarioConfig.obs_vec_length,
        #                   n_action = n_actions, 
        #                   use_normalization=AlgorithmConfig.use_normalization,
        #                   n_focus_on = AlgorithmConfig.n_focus_on, 
        #                   actor_attn_mod=AlgorithmConfig.actor_attn_mod,
        #                   dual_conc=AlgorithmConfig.dual_conc)
        from .stage_planner import StagePlanner
        self.stage_planner = StagePlanner(n_agent=self.n_agent, mcv=mcv)
        from .net import Net
        self.policy = Net(  rawob_dim=self.ScenarioConfig.obs_vec_length, 
                            state_dim=None, 
                            n_action=n_actions, 
                            n_agent=n_agent, 
                            stage_planner=self.stage_planner, 
                            alg=AlgorithmConfig)

        self.policy = self.policy.to(self.device)

        self.AvgRewardAgentWise = AlgorithmConfig.TakeRewardAsUnity
        from .ppo import PPO
        self.trainer = PPO(self.policy, cfg=AlgorithmConfig, mcv=mcv, team=self.team, n_agent=self.n_agent)
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
            os.makedirs(f'{logdir}/history_cpt/', exist_ok=True)
        if self.load_checkpoint:
            self.save_or_load('load', None)

        self._create_config_fly()
        # data integraty check
        self._unfi_frag_ = None
        # Skip currupt data integraty check after this patience is exhausted
        self.patience = 150

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
        eprsn = StateRecall['eprsn'] if 'eprsn' in StateRecall else None
        alive = StateRecall['alive'] if 'alive' in StateRecall else None
        randl = StateRecall['randl'] if 'randl' in StateRecall else None
        # make decision
        with torch.no_grad():
            action, value, action_log_prob = self.policy.act(obs, test_mode=test_mode, avail_act=avail_act, eprsn=eprsn, randl=randl)



        traj_frag = {
            '_SKIP_':        ~threads_active_flag, # thread mask
            'value':         value,
            'alive':         alive,
            'actionLogProb': action_log_prob,
            'obs':           obs,
            'action':        action,
        }
        if AlgorithmConfig.use_policy_resonance:
            traj_frag.update({
                "eprsn":         eprsn,
                "randl":         randl,
            })
        if avail_act: traj_frag.update({'avail_act':  avail_act})
        if not test_mode: StateRecall['_hook_'] = self.commit_traj_frag(traj_frag, req_hook = True)
        else: StateRecall['_hook_'] = None


        return action.copy(), StateRecall

    def train(self):
        if self.batch_traj_manager.can_exec_training():  # time to start a training routine
            # print('self.decision_interval', self.decision_interval)
            update_cnt = self.batch_traj_manager.train_and_clear_traj_pool()
            self.stage_planner.update_plan()
            if AlgorithmConfig.ConfigOnTheFly: self._config_on_fly()

    '''
        Get event from hmp task runner, save model now!
    '''
    def on_notify(self, message, **kargs):
        self.stage_planner.update_test_winrate(kargs['win_rate'])
        self.save_or_load('save', self.batch_traj_manager.update_cnt)


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






