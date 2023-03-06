base = """
{
    "AutoRL": {
        "MasterAutoRLKey": "test_auto_script",
        "target_server": {   
            "addr": "localhost:2266",
            "usr": "hmp",
            "pwd": "hmp"
        },
        "master_seed": 2132,
        "timeout_hour": 999,
        "n_run_average": 4,
        "n_init": 20,
        "acq": "ei",
        "global_bo": true,
    },

    "config.py->GlobalConfig": {
        "note": "batchexp",
        "env_name": "sr_tasks->hunter_invader3d_v2",
        "env_path": "MISSION.sr_tasks.multiagent.scenarios.hunter_invader3d_v2",
        "draw_mode": "Img",
        "num_threads": 32,
        "report_reward_interval": 32,              // reporting interval
        "test_interval": 2048,                     // test every $test_interval episode
        // "test_interval": 128,                   // test every $test_interval episode
        "test_epoch": 128,                         // test every $test_interval episode
        "fold": 1,                                 // this 'folding' is designed for IPC efficiency, you can thank python GIL for such a strange design... 
        "n_parallel_frame": 5e7,
        "max_n_episode": 5e5,
        // "max_n_episode": 2e6,
        "backup_files": [                               // backup files, pack them up
           "ALGORITHM/experimental_coopspace_vma_varnet_2", 
           "MISSION/sr_tasks/multiagent"
        ],
        "device": "cuda",                             // choose from 'cpu' (no GPU), 'cuda' (auto select GPU), 'cuda:3' (manual select GPU) 
        "gpu_party": "off"                            // default is 'off', 
    },

    "MISSION.sr_tasks.multiagent.scenarios.hunter_invader3d_v2.py->ScenarioConfig": {
        "hunter_num": 15,
        "invader_num": 5,
        "num_landmarks": 6,
        "extreme_sparse": true,
        "render": false,
        "TEAM_NAMES": [ //select team algorithm
            "ALGORITHM.script_ai.manual->DummyAlgorithmFoundationHI3D",
            "ALGORITHM.experimental_coopspace_vma_varnet_2.reinforce_foundation->ReinforceAlgorithmFoundation"
        ]
    },


    "ALGORITHM.experimental_coopspace_vma_varnet_2.reinforce_foundation.py->CoopAlgConfig": {
        "train_traj_needed": 128,
        "n_pieces_batch_division": 1,
        "ppo_epoch": 16,
        "dropout_prob": 0.0,
        "entropy_coef": 0.05,                   "__AutorlDealWithScalar__entropy_coef":             {"Type":"Continuous",    "Range":[ 0.02,  0.08 ]},
        "lr": 0.0001,                           "__AutorlDealWithScalar__lr":                       {"Type":"ContinuousLog", "Range":[ -3,  -5  ]}, // 10**-3, 10**-5
        "max_internal_step": 3,                 "__AutorlDealWithScalar__max_internal_step":        {"Type":"Discrete",      "Range":[ 1,2,3,4,5,6 ]},
        "g_num": 6,                             "__AutorlDealWithScalar__g_num":                    {"Type":"Discrete",      "Range":[ 4,5,6,7,8,9 ]},

        // "fuzzy_controller_param": [ 1,2,3 ],    "__AutorlDealWithList__fuzzy_controller_param":   [
        //     {"Type":"Discrete",   "Range":[ 4,5,6 ]},
        //     {"Type":"Discrete",   "Range":[ 4,5,6 ]},
        //     {"Type":"Discrete",   "Range":[ 4,5,6 ]},
        //     {"Type":"Discrete",   "Range":[ 4,5,6 ]},
        // ],

        "decision_interval": 25,
        "head_start_cnt": 4,
        "head_start_hold_n": 1,
        "hidden_dim": 128,
    },
    "ALGORITHM.script_ai.manual.py->CoopAlgConfig": {
    }
}
"""
device = """
{
    "config.py->GlobalConfig-->device":
    [
        "cuda:6",
        "cuda:6",
        "cuda:5",
        "cuda:5",
    ],
    "config.py->GlobalConfig-->gpu_party":
    [
        "cuda-6#1",
        "cuda-6#2",
        "cuda-5#1",
        "cuda-5#2",
    ],



}
"""


import numpy as np
import commentjson as json
from THIRDPARTY.casmopolitan.bo_interface import BayesianOptimizationInterface


def get_score(conclusion_list):
    # return get_score_acc_win(conclusion_list)
    return get_score_max_win_rate(conclusion_list)

def get_score_acc_win(conclusion_list):
    score_list = []
    for c in conclusion_list:
        conclusion_parsed = {}
        # parse
        for name, line, time in zip(c['name_list'], c['line_list'], c['time_list']):
            conclusion_parsed[name] = line
        s = conclusion_parsed['acc win ratio of=team-0']
        score_list.append(s[-1])
    return score_list


def get_score_max_win_rate(conclusion_list):
    score_list = []
    for c in conclusion_list:
        conclusion_parsed = {}
        # parse
        for name, line, time in zip(c['name_list'], c['line_list'], c['time_list']):
            conclusion_parsed[name] = line
        s = np.array(conclusion_parsed['test top-rank ratio of=team-1']).max()
        score_list.append(s)
    return score_list




class HmpBayesianOptimizationInterface(BayesianOptimizationInterface):
    # 获取基本的实验配置模板

    def compute_(self, X, normalize=False):

        X = np.array(X)
        batch = X.shape[0]
        async_struct = {
            'X': X,
            'future': [None]*batch,
            'y_result_array': np.zeros(shape=(batch, 1))
        }
        
        self.clean_profile_folder()

        for b in range(batch):
            _, async_struct['future'][b] = self.prev_part(async_struct['X'][b], batch=b)

        for b in range(batch):
            async_struct['y_result_array'][b] = self.post_part(async_struct['X'][b], get_score, async_struct['future'][b], batch=b)

        return async_struct['y_result_array']



    def prev_part(self, X, batch):
        conf_override = {"config.py->GlobalConfig-->seed": self.seed_list,"config.py->GlobalConfig-->note": self.note_list,}
        conf_override.update(self.device_conf)

        self.encounter_discrete = 0
        self.encounter_contious = 0
        parsed = []
        for i, injection in enumerate(self.inject_info_list):
            if (injection['Type']=='Discrete'):
                self.encounter_discrete += 1
            elif (injection['Type']=='Continuous') or (injection['Type']=='ContinuousLog'):
                self.encounter_contious += 1

            if injection['Type']=='Discrete':
                p_tmp = self.convert_categorical(from_x = X[i],  to_list=injection['Range'], p_index=i)
            elif (injection['Type']=='Continuous'):
                p_tmp = self.convert_continuous( from_x = X[i], to_range=injection['Range'], p_index=i)
            elif (injection['Type']=='ContinuousLog'):
                p_tmp = self.convert_continuous_log( from_x = X[i], to_range=injection['Range'], p_index=i)
            parsed.append(p_tmp)
            if len(injection['key_path']) == 3:
                l1,l2,l3 = injection['key_path']
                if f'{l1}-->{l2}' not in conf_override: conf_override[f'{l1}-->{l2}'] = [[None]*injection['list_len']] * self.n_run
                for n in range(self.n_run): conf_override[f'{l1}-->{l2}'][n][l3] = p_tmp

            elif len(injection['key_path']) == 2:
                l1,l2 = injection['key_path']
                if f'{l1}-->{l2}' not in conf_override: conf_override[f'{l1}-->{l2}'] = {}
                conf_override[f'{l1}-->{l2}'] = [p_tmp] * self.n_run

            print(injection)


        self.logger.info(f'input X={X}, parsed {parsed}')



        self.internal_step_cnt += 1
        future_list = self.push_experiments_and_execute(conf_override, batch)
        return X, future_list
    
    def post_part(self, X, get_score, future_list, batch):
        from UTIL.batch_exp import fetch_experiment_conclusion
        conclusion_list = fetch_experiment_conclusion(step=self.internal_step_cnt, future_list=future_list, n_run_mode=self.n_run_mode_withbatch[batch], timeout_hour=self.timeout_hour)
        y_array = get_score(conclusion_list)
        y = np.array(y_array).mean()
        self.logger.info(f'input X={X} | output {y_array}, average {y}')
        res = (y - self.y_offset)
        if self.optimize_direction == 'maximize': res = -res
        else: assert self.optimize_direction == 'minimize'
        return res

    def __init__(self, MasterAutoRLKey, base_conf, device_conf, normalize=False, seed=None, n_run=None):
        super().__init__(MasterAutoRLKey)
        self.base_conf = base_conf
        self.device_conf = device_conf
        if self.base_conf["config.py->GlobalConfig"]["max_n_episode"] < 1000:
            input('conf_override["config.py->GlobalConfig-->max_n_episode"] < 1000, confirm?')
        self.internal_step_cnt = 0

        self.n_run = n_run
        self.seed = seed if seed is not None else 0
        self.seed_list = [self.seed + i for i in range(self.n_run)]
        self.note_list = [f'parallel-{i}' for i in range(self.n_run)]

        self.normalize = False
        self.y_offset = 0
        self.optimize_direction = 'maximize' # 'minimize'

    def convert_categorical(self, from_x, to_list, p_index):
        assert p_index in self.P_CategoricalDims
        where = np.where(self.P_CategoricalDims==p_index)[0]
        assert len(to_list) == self.P_NumCategoryList[where]
        from_x_ = int(from_x)
        assert from_x_-from_x == 0
        return to_list[from_x_]

    def convert_continuous(self, from_x, to_range, p_index):
        assert p_index in self.P_ContinuousDims
        where = np.where(self.P_ContinuousDims==p_index)[0]
        new_range = to_range[1] - to_range[0]
        xx = ((from_x - self.P_ContinuousLowerBound[where]) * new_range / (self.P_ContinuousUpperBound[where] - self.P_ContinuousLowerBound[where])) + to_range[0]
        return float(xx)
    
    def convert_continuous_log(self, from_x, to_range, p_index):
        assert p_index in self.P_ContinuousDims
        where = np.where(self.P_ContinuousDims==p_index)[0]
        new_range = to_range[1] - to_range[0]
        xx = ((from_x - self.P_ContinuousLowerBound[where]) * new_range / (self.P_ContinuousUpperBound[where] - self.P_ContinuousLowerBound[where])) + to_range[0]
        return 10**float(xx)

    def clean_profile_folder(self):
        import shutil, os, glob, datetime, time
        if os.path.exists('PROFILE'):
            res = glob.glob('PROFILE/*')
            for exp in res:
                input_str = exp # "PROFILE/2023-03-03-09-56-59-Bo_AutoRL"
                # Extract the time substring
                time_str = input_str.split("/")[-1][:19]
                # Parse the time substring using datetime.strptime()
                time_then = datetime.datetime.strptime(time_str, "%Y-%m-%d-%H-%M-%S").timestamp()
                time_now = time.time()
                dt_hour = (time_now - time_then)/3600
                if dt_hour > 2:
                    shutil.copytree(exp, f'TEMP/{time_str}')
                    shutil.rmtree(exp)

    def push_experiments_and_execute(self, conf_override, batch):
        # copy the experiments
        import shutil, os
        shutil.copyfile(__file__, os.path.join(os.path.dirname(__file__), 'batch_experiment_backup.py'))
        # run experiments remotely
        from UTIL.batch_exp import run_batch_exp
        print('Execute in server:', self.n_run_mode_withbatch[batch][0])
        future = run_batch_exp(self.sum_note, self.n_run, self.n_run_mode_withbatch[batch], self.base_conf, conf_override, __file__, skip_confirm=True, master_folder='AutoRL', logger=self.logger)
        return future






class AutoRlTask():
    def __init__(self, baseconf, deviceconf):
        self.base_conf   = json.loads(baseconf)
        self.device_conf = json.loads(deviceconf)


    def begin(self):
        # from THIRDPARTY.casmopolitan.mixed_test_func import *
        import logging, torch, gpytorch
        import argparse
        import time
        from VISUALIZE.mcom import mcom
        from THIRDPARTY.casmopolitan.bo_interface_new import BayesianOptimisation

        MasterAutoRLKey = self.base_conf["AutoRL"]["MasterAutoRLKey"]
        # Set up the objective function
        parser = argparse.ArgumentParser('Run Experiments')
        parser.add_argument('--max_iters', type=int, default=300, help='Maximum number of BO iterations.')
        parser.add_argument('--lamda', type=float, default=1e-6, help='the noise to inject for some problems')
        # parser.add_argument('--ard', action='store_true', help='whether to enable automatic relevance determination')
        parser.add_argument('--ard', action='store_false', help='whether to enable automatic relevance determination')
        parser.add_argument('--random_seed_objective', type=int, default=20, help='The default value of 20 is provided also in COMBO')
        parser.add_argument('-d', '--debug', action='store_true', help='Whether to turn on debugging mode (a lot of output will be generated).')
        parser.add_argument('--no_save', action='store_true', help='If activated, do not save the current run into a log folder.')
        parser.add_argument('-k', '--kernel_type', type=str, default=None, help='specifies the kernel type')
        parser.add_argument('--infer_noise_var', action='store_true')

        args = parser.parse_args()
        args.batch_size = 1
        args.n_trials = 1
        args.seed = self.base_conf["AutoRL"]["master_seed"]
        args.save_path = f'AUTORL/{MasterAutoRLKey}/'
        args.acq = self.base_conf["AutoRL"]["acq"]
        args.n_init = self.base_conf["AutoRL"]["n_init"]
        args.global_bo = self.base_conf["AutoRL"]["global_bo"]
        options = vars(args)
        print(options)
        # Set numpy seed
        np.random.seed(args.seed)
        np.set_printoptions(3, suppress=True)
        torch.cuda.manual_seed(args.seed)


        if args.debug: logging.basicConfig(level=logging.INFO)
        assert args.acq in ['ucb', 'ei', 'thompson'], 'Unknown acquisition function choice ' + str(args.acq)
        mcv = mcom(
            path = f'AUTORL/{MasterAutoRLKey}/', 
            rapid_flush = True, draw_mode = 'Img', 
            image_path = f'AUTORL/{MasterAutoRLKey}/decend.jpg', 
            tag = 'BayesianOptimisation')
        mcv.rec_init(color='g')
        self.n_run = self.base_conf["AutoRL"]["n_run_average"]
        f = HmpBayesianOptimizationInterface(MasterAutoRLKey=MasterAutoRLKey, base_conf=self.base_conf, device_conf=self.device_conf, n_run=self.n_run)
        f.n_run_mode_withbatch =  [[self.base_conf["AutoRL"]["target_server"]]*self.n_run, ]
        f.sum_note = "Bo_AutoRL"
        f.timeout_hour = self.base_conf["AutoRL"]["timeout_hour"]

        self.read_conf(f, self.base_conf)
        BayesianOptimisation(0, mcv, args, MasterAutoRLKey=MasterAutoRLKey, interface=f)

    def read_conf(self, f, base_conf):
        base_conf.pop("AutoRL")
        f.problem_type = 'categorical'
        self.type_cnt = {'Continuous':0, 'ContinuousLog':0, 'Discrete':0}
        self.inject_info_list = []
        f.P_ContinuousDims = []
        f.P_ContinuousLowerBound = []
        f.P_ContinuousUpperBound = []
        f.P_CategoricalDims = []
        f.P_NumCategoryList = []

        def deal_scalar(opt_item, keys, list_len=None):
            Type = opt_item['Type']
            self.type_cnt[Type] += 1
            Range = opt_item['Range']
            this_is_n_th_parameter = len(self.inject_info_list)
            self.inject_info_list.append({
                "key_path": keys,
                "Type": Type,
                "Range": Range,
                "list_len": list_len
            })

            if Type == "Continuous":
                f.P_ContinuousDims.append(this_is_n_th_parameter)
                f.P_ContinuousLowerBound.append(-1)
                f.P_ContinuousUpperBound.append(+1)
                f.problem_type = 'mixed'
            elif Type == "ContinuousLog":
                f.P_ContinuousDims.append(this_is_n_th_parameter)
                f.P_ContinuousLowerBound.append(-1)
                f.P_ContinuousUpperBound.append(+1)
                f.problem_type = 'mixed'
            elif Type == "Discrete":
                f.P_CategoricalDims.append(this_is_n_th_parameter)
                f.P_NumCategoryList.append(len(Range))
            return True
        

        for main_key in base_conf:
            sub_conf = base_conf[main_key]
            for sub_key in list(sub_conf.keys()):
                if sub_key.startswith('__AutorlDealWithScalar__'):
                    k = sub_key.replace('__AutorlDealWithScalar__','')
                    if sub_conf[sub_key]["Type"] in ("Discrete"):
                        opt_item = sub_conf.pop(sub_key)
                        deal_scalar(opt_item, keys=(main_key, k))
                if sub_key.startswith('__AutorlDealWithList__'):
                    k = sub_key.replace('__AutorlDealWithList__','')
                    if sub_conf[sub_key][0]["Type"] in ("Discrete"):
                        opt_list = sub_conf.pop(sub_key)
                        for i, opt_item in enumerate(opt_list):
                            deal_scalar(opt_item, keys=(main_key, k, i), list_len=len(opt_list))

        for main_key in base_conf:
            sub_conf = base_conf[main_key]
            for sub_key in list(sub_conf.keys()):
                if sub_key.startswith('__AutorlDealWithScalar__'):
                    k = sub_key.replace('__AutorlDealWithScalar__','')
                    if sub_conf[sub_key]["Type"] in ("Continuous", "ContinuousLog"):
                        opt_item = sub_conf.pop(sub_key)
                        deal_scalar(opt_item, keys=(main_key, k))
                if sub_key.startswith('__AutorlDealWithList__'):
                    k = sub_key.replace('__AutorlDealWithList__','')
                    if sub_conf[sub_key][0]["Type"] in ("Continuous", "ContinuousLog"):
                        opt_list = sub_conf.pop(sub_key)
                        for i, opt_item in enumerate(opt_list):
                            deal_scalar(opt_item, keys=(main_key, k, i), list_len=len(opt_list))

        f.inject_info_list = self.inject_info_list
        f.P_ContinuousDims = np.array(f.P_ContinuousDims)
        f.P_ContinuousLowerBound = np.array(f.P_ContinuousLowerBound)
        f.P_ContinuousUpperBound = np.array(f.P_ContinuousUpperBound)
        f.P_CategoricalDims = np.array(f.P_CategoricalDims)
        f.P_NumCategoryList = np.array(f.P_NumCategoryList)

        print('read done')

if __name__ == '__main__':
    autorl_task = AutoRlTask(baseconf=base, deviceconf=device)
    autorl_task.begin()
