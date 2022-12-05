import os, time, torch, traceback, shutil
import numpy as np
from UTIL.colorful import *
from config import GlobalConfig
from UTIL.tensor_ops import __hash__, repeat_at
from ..commom.rl_alg_base import RLAlgorithmBase
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
    actor_attn_mod = False
    load_specific_checkpoint = ''
    dual_conc = True
    use_my_attn = True
    use_policy_resonance = False

    # net
    shell_obs_add_id = True
    shell_obs_add_previous_act = False

    fall_back_to_small_net = False

class ReinforceAlgorithmFoundation(RLAlgorithmBase):
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
        self.stage_planner = StagePlanner(mcv=mcv)

        from .shell_env import ShellEnvWrapper
        self.shell_env = ShellEnvWrapper(
            n_agent, n_thread, space, mcv, self, AlgorithmConfig, self.ScenarioConfig)
            
        from .net import Net
        self.device = GlobalConfig.device
        if self.ScenarioConfig.EntityOriented:
            rawob_dim = self.ScenarioConfig.obs_vec_length
        else:
            rawob_dim = space['obs_space']['obs_shape']
        if AlgorithmConfig.shell_obs_add_id:
            rawob_dim = rawob_dim + self.n_agent
        if AlgorithmConfig.shell_obs_add_previous_act:
            rawob_dim = rawob_dim + n_actions
        self.policy = Net(rawob_dim=rawob_dim, state_dim=self.state_dim, n_action=n_actions, stage_planner=self.stage_planner)
        self.policy = self.policy.to(self.device)

        # initialize optimizer and trajectory (batch) manager
        from .ppo import PPO
        from .trajectory import BatchTrajManager
        self.trainer = PPO(self.policy, ppo_config=AlgorithmConfig, mcv=mcv)
        self.batch_traj_manager = BatchTrajManager(
            n_env=n_thread, traj_limit=int(self.ScenarioConfig.MaxEpisodeStep),
            trainer_hook=self.trainer.train_on_traj)
        # confirm that reward method is correct
        self.check_reward_type(AlgorithmConfig)


        # load checkpoints
        self.load_checkpoint = AlgorithmConfig.load_checkpoint
        logdir = GlobalConfig.logdir
        # makedirs if not exists
        if not os.path.exists('%s/history_cpt/' % logdir):
            os.makedirs('%s/history_cpt/' % logdir)
        if self.load_checkpoint:
            manual_dir = AlgorithmConfig.load_specific_checkpoint
            ckpt_dir = '%s/model.pt' % logdir if manual_dir == '' else '%s/%s' % (logdir, manual_dir)
            cuda_n = 'cpu' if 'cpu' in self.device else self.device
            self.policy.load_state_dict(torch.load(ckpt_dir, map_location=cuda_n))
            print黄('loaded checkpoint:', ckpt_dir)

        # data integraty check
        self._unfi_frag_ = None
        # Skip currupt data integraty check after this patience is exhausted
        self.patience = 1000


    def action_making(self, StateRecall, test_mode):
        assert StateRecall['obs'] is not None, ('Make sure obs is ok')

        obs, threads_active_flag = StateRecall['obs'], StateRecall['threads_active_flag']
        assert len(obs) == sum(threads_active_flag), ('Make sure the right batch of obs!')
        avail_act = StateRecall['avail_act'] if 'avail_act' in StateRecall else None
        state = StateRecall['state'] if 'state' in StateRecall else None
        eprsn = repeat_at(StateRecall['eprsn'], -1, self.n_agent) if 'eprsn' in StateRecall else None

        with torch.no_grad():
            action, value, action_log_prob = self.policy.act(
                obs, state=state, test_mode=test_mode, avail_act=avail_act, eprsn=eprsn)

        # commit obs to buffer, vars named like _x_ are aligned, others are not!
        traj_framefrag = {
            "_SKIP_":        ~threads_active_flag,
            "value":         value,
            "actionLogProb": action_log_prob,
            "obs":           obs,
            "state":         state,
            "action":        action,
        }
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

    '''
        Get event from hmp task runner, save model now!
    '''
    def train(self):
        if self.batch_traj_manager.can_exec_training():
            # time to start a training routine
            self.batch_traj_manager.train_and_clear_traj_pool()
            self.stage_planner.update_plan()

    def on_notify(self, message, **kargs):
        self.stage_planner.update_test_winrate(kargs['win_rate'])
        self.save_model(
            update_cnt=self.batch_traj_manager.update_cnt,
            info=str(kargs)
        )

    def save_model(self, update_cnt, info=None):
        '''
            save model now!
            save if triggered when:
            1. Update_cnt = 50, 100, ...
            2. Given info, indicating a hmp command
            3. A flag file is detected, indicating a save command from human
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



    def process_framedata(self, traj_framedata):
        ''' 
            hook is called when reward and next moment observation is ready,
            now feed them into trajectory manager.
            Rollout Processor | 准备提交Rollout, 以下划线开头和结尾的键值需要对齐(self.n_thread, ...)
            note that keys starting with _ must have shape (self.n_thread, ...), details see fn:mask_paused_env()
        '''
        # strip info, since it is not array
        items_to_pop = ['info', 'Latest-Obs']
        for k in items_to_pop:
            if k in traj_framedata:
                traj_framedata.pop(k)
        # the agent-wise reward is supposed to be the same, so averge them
        if self.ScenarioConfig.RewardAsUnity:
            traj_framedata['reward'] = repeat_at(traj_framedata['reward'], insert_dim=-1, n_times=self.n_agent)
        # change the name of done to be recognised (by trajectory manager)
        traj_framedata['_DONE_'] = traj_framedata.pop('done')
        traj_framedata['_TOBS_'] = traj_framedata.pop(
            'Terminal-Obs-Echo') if 'Terminal-Obs-Echo' in traj_framedata else None
        # mask out pause thread
        traj_framedata = self.mask_paused_env(traj_framedata)
        # put the frag into memory
        self.batch_traj_manager.feed_traj(traj_framedata)

    def mask_paused_env(self, frag):
        running = ~frag['_SKIP_']
        if running.all():
            return frag
        for key in frag:
            if not key.startswith('_') and hasattr(frag[key], '__len__') and len(frag[key]) == self.n_thread:
                frag[key] = frag[key][running]
        return frag

