import os, time, torch
import numpy as np
from Util.colorful import *
from config import GlobalConfig
from Util.tensor_ops import __hash__, repeat_at

'''
    AlgorithmConfig: This config class will be 'injected' with new settings from json.
    (E.g., override configs with ```python main.py --cfg example.jsonc```)
    (please see Util.config_args to find out how this advanced trick works out.)
'''
class AlgorithmConfig:  
    # configuration, open to jsonc modification
    gamma = 0.99
    tau = 0.95
    train_traj_needed = 512
    upper_training_epoch = 4
    load_checkpoint = False
    TakeRewardAsUnity = False
    use_normalization = True
    add_prob_loss = False
    alternative_critic = False

    n_focus_on = 2
    n_entity_placeholder = 10
    turn_off_threat_est = False

    # PPO part
    clip_param = 0.2
    ppo_epoch = 16
    n_pieces_batch_division = 1
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
    prevent_batchsize_oom = False
    gamma_in_reward_forwarding = False
    gamma_in_reward_forwarding_value = 0.99

    # extral
    extral_train_loop = False
    actor_attn_mod = False
    load_specific_checkpoint = ''
    dual_conc = True
    use_my_attn = True
    alternative_critic = False

    # net
    net_hdim = 32

class ReinforceAlgorithmFoundation(object):
    def __init__(self, n_agent, n_thread, space, mcv=None, team=None):
        self.n_thread = n_thread
        self.n_agent = n_agent
        self.act_space = space['act_space']
        self.obs_space = space['obs_space']
        self.ScenarioConfig = GlobalConfig.ScenarioConfig
        n_actions = GlobalConfig.ScenarioConfig.n_actions

        from .shell_env import ShellEnvWrapper
        self.shell_env = ShellEnvWrapper(
            n_agent, n_thread, space, mcv, self, AlgorithmConfig, self.ScenarioConfig)
            
        from .net import Net
        self.device = GlobalConfig.device
        if self.ScenarioConfig.EntityOriented :
            rawob_dim = self.ScenarioConfig.obs_vec_length
        else:
            rawob_dim = space['obs_space']['obs_shape']
        self.policy = Net(rawob_dim=rawob_dim, n_action=n_actions)
        self.policy = self.policy.to(self.device)

        # initialize policy network and traj memory manager
        from .ppo import PPO
        from .trajectory import BatchTrajManager
        self.trainer = PPO(self.policy, ppo_config=AlgorithmConfig, mcv=mcv)
        self.batch_traj_manager = BatchTrajManager(
            n_env=n_thread, traj_limit=int(self.ScenarioConfig.MaxEpisodeStep),
            trainer_hook=self.trainer.train_on_traj)
                    
        # confirm that reward method is correct
        if self.ScenarioConfig.RewardAsUnity != AlgorithmConfig.TakeRewardAsUnity:
            assert self.ScenarioConfig.RewardAsUnity
            assert not AlgorithmConfig.TakeRewardAsUnity
            print亮紫(
                'Warning, the scenario (Mission) provide `RewardAsUnity`, but AlgorithmConfig does not `TakeRewardAsUnity` !')
            print亮紫(
                'If you continue, team reward will be duplicated to serve as individual rewards, wait 3s to proceed...')
            time.sleep(3)

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

        with torch.no_grad():
            action, value, action_log_prob = self.policy.act(
                obs, test_mode=test_mode, avail_act=avail_act)

        # Warning! vars named like _x_ are aligned, others are not!
        traj_frag = {
            "_SKIP_":        ~threads_active_flag,
            "value":         value,
            "actionLogProb": action_log_prob,
            "obs":           obs,
            "action":        action,
        }
        if avail_act is not None:
            traj_frag.update({'avail_act':  avail_act})
        hook = self.commit_frag(traj_frag, req_hook=True) if not test_mode else self._no_hook

        # deal with rollout later when the reward is ready, leave a hook as a callback here
        StateRecall['_hook_'] = hook
        return action.copy(), StateRecall

    '''
        Interfacing with marl, standard method that you must implement
        (redirect to shell_env to help with history rolling)
    '''
    def interact_with_env(self, StateRecall):
        return self.shell_env.interact_with_env(StateRecall)

    '''
        When shell_env finish the preparation, interact_with_env_genuine is called
        (Determine whether or not to do a training routinue)
    '''
    def interact_with_env_genuine(self, StateRecall):
        if not StateRecall['Test-Flag']: self.train()  # when needed, train!
        return self.action_making(StateRecall, StateRecall['Test-Flag'])

    '''
        Get event from hmp task runner, save model now!
    '''
    def train(self):
        if self.batch_traj_manager.can_exec_training():
            # time to start a training routine
            self.batch_traj_manager.train_and_clear_traj_pool()

    '''
        Get event from hmp task runner, save model now!
    '''
    def on_notify(self, message, **kargs):
        self.save_model(
            update_cnt=self.batch_traj_manager.update_cnt,
            info=str(kargs)
        )

    '''
        save model now!
        save if triggered when:
        1. Update_cnt = 50, 100, ...
        2. Given info, indicating a hmp command
        3. A flag file is detected, indicating a save command from human
    '''
    def save_model(self, update_cnt, info=None):
        logdir = GlobalConfig.logdir
        flag = '%s/save_now' % logdir
        if update_cnt % 50 == 0 or (info is not None) or os.path.exists(flag):
            # dir 1
            pt_path = '%s/model.pt' % logdir
            print绿('saving model to %s' % pt_path)
            torch.save(self.policy.state_dict(), pt_path)

            # dir 2
            info = str(update_cnt) if info is None else ''.join([str(update_cnt), '_', info])
            pt_path = '%s/history_cpt/model_%s.pt' % (logdir, info)
            torch.save(self.policy.state_dict(), pt_path)
            try:
                os.remove(flag)
            except:
                pass
            print绿('save_model fin')

    ''' 
        function to be called when reward is received
    '''
    def commit_frag(self, f1, req_hook=True):
        assert self._unfi_frag_ is None
        self._unfi_frag_ = f1
        self._check_data_hash()  # check data integraty
        if req_hook:
            # leave hook
            return lambda new_frag: self.rollout_frag_hook(new_frag)
        else:
            return None

    ''' 
        hook is called when reward and next moment observation is ready,
        now feed them into trajectory manager.
        Rollout Processor 准备提交Rollout, 以下划线开头和结尾的键值需要对齐(self.n_thread, ...)
        note that keys starting with _ must have shape (self.n_thread, ...), details see fn:mask_paused_env()
    '''
    def rollout_frag_hook(self, new_frag):

        # do data curruption check at beginning, this is important!
        self._check_data_curruption()
        # strip info, since it is not array
        items_to_pop = ['info', 'Latest-Obs']
        for k in items_to_pop:
            if k in new_frag:
                new_frag.pop(k)
        # the agent-wise reward is supposed to be the same, so averge them
        if self.ScenarioConfig.RewardAsUnity:
            new_frag['reward'] = repeat_at(new_frag['reward'], insert_dim=-1, n_times=self.n_agent)
        # change the name of done to be recognised (by trajectory manager)
        new_frag['_DONE_'] = new_frag.pop('done')
        new_frag['_TOBS_'] = new_frag.pop(
            'Terminal-Obs-Echo') if 'Terminal-Obs-Echo' in new_frag else None
        # integrate frag part1 and part2
        self._unfi_frag_.update(new_frag)
        self.__completed_frag = self.mask_paused_env(self._unfi_frag_)
        # put the frag into memory
        self.batch_traj_manager.feed_traj(self.__completed_frag)
        self._unfi_frag_ = None

    def mask_paused_env(self, frag):
        running = ~frag['_SKIP_']
        if running.all():
            return frag
        for key in frag:
            if not key.startswith('_') and hasattr(frag[key], '__len__') and len(frag[key]) == self.n_thread:
                frag[key] = frag[key][running]
        return frag

    def _no_hook(self, new_frag):
        return

    # protect data from overwriting
    def _check_data_hash(self):
        if self.patience > 0:
            self.patience -= 1
            self.hash_db = {}
            # for debugging, to detect write protection error
            for key in self._unfi_frag_:
                item = self._unfi_frag_[key]
                if isinstance(item, dict):
                    self.hash_db[key] = {}
                    for subkey in item:
                        subitem = item[subkey]
                        self.hash_db[key][subkey] = __hash__(subitem)
                else:
                    self.hash_db[key] = __hash__(item)

    # protect data from overwriting
    def _check_data_curruption(self):
        if self.patience > 0:
            self.patience -= 1
            assert self._unfi_frag_ is not None
            assert self.hash_db is not None
            for key in self._unfi_frag_:
                item = self._unfi_frag_[key]
                if isinstance(item, dict):
                    for subkey in item:
                        subitem = item[subkey]
                        assert self.hash_db[key][subkey] == __hash__(subitem), ('Currupted data!')
                else:
                    assert self.hash_db[key] == __hash__(item), ('Currupted data!')
