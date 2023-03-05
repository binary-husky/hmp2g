import os, time, torch, shutil
import numpy as np
from UTIL.colorful import *
from config import GlobalConfig
from UTIL.tensor_ops import __hash__
from ALGORITHM.common.rl_alg_base import RLAlgorithmBase
from ALGORITHM.common.onfly_config import ConfigOnFly
from UTIL.tensor_ops import __hash__, repeat_at
from ALGORITHM.common.rl_alg_base import RLAlgorithmBase
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
    TakeRewardAsUnity = False
    use_normalization = True
    add_prob_loss = False
    n_focus_on = 2
    n_entity_placeholder = 10
    load_checkpoint = False

    # PPO part
    clip_param = 0.2
    ppo_epoch = 16
    n_pieces_batch_division = 1
    value_loss_coef = 0.1
    entropy_coef = 0.05
    max_grad_norm = 0.5
    clip_param = 0.2
    lr = 1e-4

    # sometimes the episode length gets longer,
    # resulting in more samples and causing GPU OOM,
    # prevent this by fixing the number of samples to initial
    # by randomly sampling and droping
    prevent_batchsize_oom = True
    gamma_in_reward_forwarding = False
    gamma_in_reward_forwarding_value = 0.99

    net_hdim = 32
    # extral
    extral_train_loop = False
    load_specific_checkpoint = ''
    use_conc_net = True
    dual_conc = True
    use_my_attn = True
    use_policy_resonance = False
    policy_resonance_method = 'level'

    # net
    shell_obs_add_id = True
    shell_obs_add_previous_act = False

    fall_back_to_small_net = False

    distribution_precision = 8
    # pg_target_distribute = [0,1,2,3,4,5]
    target_distribute = [0]
    ConfigOnTheFly = True

    BlockInvalidPg = True
    advantage_norm = True

class ReinforceAlgorithmFoundation(RLAlgorithmBase, ConfigOnFly):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.act_space = space['act_space']
        self.obs_space = space['obs_space']
        self.state_dim = self.obs_space['state_shape']
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        n_actions = GlobalConfig.ScenarioConfig.n_actions
        # self.StagePlanner
        from .stage_planner import StagePlanner
        self.stage_planner = StagePlanner(n_agent=self.n_agent, mcv=mcv)

        if AlgorithmConfig.use_conc_net: from .shell_env import ShellEnvWrapper
        else: from .shell_env_without_conc import ShellEnvWrapper
        self.shell_env = ShellEnvWrapper(
            n_agent, n_thread, space, mcv, self, AlgorithmConfig, self.ScenarioConfig)
            
        if AlgorithmConfig.use_conc_net: from .net import Net
        else: from .net_small import Net
        
        self.device = GlobalConfig.device
        if self.ScenarioConfig.EntityOriented:
            rawob_dim = self.ScenarioConfig.obs_vec_length
        else:
            rawob_dim = space['obs_space']['obs_shape']
        if AlgorithmConfig.shell_obs_add_id:
            rawob_dim = rawob_dim + self.n_agent
        if AlgorithmConfig.shell_obs_add_previous_act:
            rawob_dim = rawob_dim + n_actions
        self.policy = Net(rawob_dim=rawob_dim, state_dim=self.state_dim, n_action=n_actions, n_agent=n_agent, stage_planner=self.stage_planner, alg=AlgorithmConfig)
        self.policy = self.policy.to(self.device)

        # initialize optimizer and trajectory (batch) manager
        from .ppo import PPO
        from ALGORITHM.common.traj_gae import BatchTrajManager
        self.trainer = PPO(self.policy, cfg=AlgorithmConfig, mcv=mcv)
        self.batch_traj_manager = BatchTrajManager(
            n_env=n_thread, traj_limit=int(self.ScenarioConfig.MaxEpisodeStep),
            trainer_hook=self.trainer.train_on_traj, alg_cfg=AlgorithmConfig)
        # confirm that reward method is correct
        self.check_reward_type(AlgorithmConfig)

        # load checkpoints
        self.load_checkpoint = AlgorithmConfig.load_checkpoint
        logdir = GlobalConfig.logdir
        # makedirs if not exists
        if not os.path.exists(f'{logdir}/history_cpt/'):
            os.makedirs(f'{logdir}/history_cpt/', exist_ok=True)
        if self.load_checkpoint:
            manual_dir = AlgorithmConfig.load_specific_checkpoint
            ckpt_dir = f'{logdir}/model.pt' if manual_dir == '' else f'{logdir}/{manual_dir}'
            cuda_n = 'cpu' if 'cpu' in self.device else self.device
            self.policy.load_state_dict(torch.load(ckpt_dir, map_location=cuda_n))
            print黄('loaded checkpoint:', ckpt_dir)

        # data integraty check
        self._unfi_frag_ = None
        # Skip currupt data integraty check after this patience is exhausted
        self.patience = 1000

        # activate config on the fly ability
        if AlgorithmConfig.ConfigOnTheFly:
            self._create_config_fly()

    def action_making(self, StateRecall, test_mode):
        assert StateRecall['obs'] is not None, ('Make sure obs is ok')

        obs, threads_active_flag = StateRecall['obs'], StateRecall['threads_active_flag']
        assert len(obs) == sum(threads_active_flag), ('Make sure the right batch of obs!')
        avail_act = StateRecall['avail_act'] if 'avail_act' in StateRecall else None
        state = StateRecall['state'] if 'state' in StateRecall else None
        eprsn = StateRecall['eprsn'] if 'eprsn' in StateRecall else None
        alive = StateRecall['alive'] if 'alive' in StateRecall else None
        randl = StateRecall['randl'] if 'randl' in StateRecall else None

        with torch.no_grad():
            action, v, action_log_prob = self.policy.act(
                obs, state=state, test_mode=test_mode, avail_act=avail_act, eprsn=eprsn, randl=randl)

        # commit obs to buffer, vars named like _x_ are aligned, others are not!
        traj_framefrag = {
            "_SKIP_":        ~threads_active_flag,
            "actionLogProb": action_log_prob,
            "obs":           obs,
            "alive":         alive,
            "randl":         randl,
            "eprsn":         eprsn,
            "state":         state,
            "action":        action,
        }
        if AlgorithmConfig.policy_resonance_method == 'level':
            traj_framefrag.update({"BAL_value_all_level":      v})
        else:
            traj_framefrag.update({"value":      v})

        if avail_act is not None: traj_framefrag.update({'avail_act':  avail_act})
        # deal with rollout later when the reward is ready, leave a hook as a callback here
        if not test_mode: StateRecall['_hook_'] = self.commit_traj_frag(traj_framefrag, req_hook = True)
        return action.copy(), StateRecall


    def interact_with_env(self, StateRecall):
        '''
            Interfacing with marl, standard method that you must implement
            (redirect to shell_env to help with history rolling)
        '''
        return self.shell_env.interact_with_env(StateRecall)


    def interact_with_env_genuine(self, StateRecall):
        '''
            When shell_env finish the preparation, interact_with_env_genuine is called
            (Determine whether or not to do a training routinue)
        '''
        if not StateRecall['Test-Flag']: self.train()  # when needed, train!
        return self.action_making(StateRecall, StateRecall['Test-Flag'])


    def train(self):
        if self.batch_traj_manager.can_exec_training():
            # time to start a training routine
            self.batch_traj_manager.train_and_clear_traj_pool()
            self.stage_planner.update_plan()
            if AlgorithmConfig.ConfigOnTheFly: self._config_on_fly()


    '''
        Get event from hmp task runner, save model now!
    '''
    def on_notify(self, message, **kargs):
        self.stage_planner.update_test_winrate(kargs['win_rate'])
        self.save_model(
            update_cnt=self.batch_traj_manager.update_cnt,
            info=str(kargs)
        )

    def save_model(self, update_cnt, info=None):
        '''
            save model now!
            save if triggered when: on_notify() is called
            
        '''
        if not os.path.exists(f'{GlobalConfig.logdir}/history_cpt/'): 
            os.makedirs(f'{GlobalConfig.logdir}/history_cpt/')

        # dir 1
        pt_path = f'{GlobalConfig.logdir}/model.pt'
        print绿('saving model to %s' % pt_path)
        torch.save(self.policy.state_dict(), pt_path)

        # dir 2
        info = str(update_cnt) if info is None else ''.join([str(update_cnt), '_', info])
        pt_path2 = f'{GlobalConfig.logdir}/history_cpt/model_{info}.pt'
        shutil.copyfile(pt_path, pt_path2)

        print绿('save_model fin')


