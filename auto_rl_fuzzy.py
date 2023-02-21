# An abstract class implementation for all test functions

from abc import abstractmethod
import numpy as np
import commentjson as json
import logging




#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
###################################### 第一部分 ：贝叶斯优化接口父类 ##############################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
class BayesianOptimizationInterface:
    """
    The abstract class for all benchmark functions acting as objective functions for BO.
    Note that we assume all problems will be minimization problem, so convert maximisation problems as appropriate.
    """

    # this should be changed if we are tackling a mixed, or continuous problem, for e.g.
    problem_type = 'categorical'

    def __init__(self, normalize=True, **kwargs):
        self.normalize = normalize
        self.n_vertices = None
        self.P_NumCategoryList = None
        self.dim = None
        self.continuous_dims = None
        self.categorical_dims = None
        self.int_constrained_dims = None

    def _check_int_constrained_dims(self):
        if self.int_constrained_dims is None:
            return
        assert self.continuous_dims is not None, 'int_constrained_dims must be a subset of the continuous_dims, ' \
                                                 'but continuous_dims is not supplied!'
        int_dims_np = np.asarray(self.int_constrained_dims)
        cont_dims_np = np.asarray(self.continuous_dims)
        assert np.all(np.in1d(int_dims_np, cont_dims_np)), "all continuous dimensions with integer " \
                                                           "constraint must be themselves contained in the " \
                                                           "continuous_dimensions!"

    @abstractmethod
    def compute(self, x, normalize=None):
        raise NotImplementedError()

    def sample_normalize(self, size=None):
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for i in range(size):
            x = np.array([np.random.choice(self.P_NumCategoryList[_]) for _ in range(self.dim)])
            y.append(self.compute(x, normalize=False, ))
        y = np.array(y)
        return np.mean(y), np.std(y)

    def __call__(self, *args, **kwargs):
        return self.compute(*args, **kwargs)



#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
###################################### 第二部分 ：贝叶斯优化HMAP接口继承父类 ###########################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

class HmpBayesianOptimizationInterface(BayesianOptimizationInterface):
    def __init__(self, seed=None):
        super().__init__()
        self.problem_type = 'categorical' # 'mixed'
        self.seed = seed if seed is not None else 0
        self.n_run = 4
        self.seed_list = [self.seed + i for i in range(self.n_run)]
        self.note_list = [f'parallel-{i}' for i in range(self.n_run)]
        self.n_run_mode = [
            {   
                # "addr": "172.18.116.149:2266",
                "addr": "210.75.240.143:2236",
                "usr": "hmp",
                "pwd": "hmp"
            },
        ]*self.n_run
        self.sum_note = "Bo_AutoRL"
        self.base_conf = json.loads(self.obtain_base_experiment())
        self.internal_step_cnt = 0

        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.DEBUG)
        # create a file handler
        handler = logging.FileHandler('auto_rl_processing.log')
        # set the logging level for the handler
        handler.setLevel(logging.DEBUG)
        # create a formatter
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        # add the formatter to the handler
        handler.setFormatter(formatter)
        # add the handler to the self.logger
        self.logger.addHandler(handler)
        self.logger.debug('logger start')

        self.P_CategoricalDims = np.array([0, 1, 2, 3, 4, 5])
        self.P_NumCategoryList = np.array([3, 3, 3, 3, 3, 3])

        # self.P_ContinuousDims = np.array([2, 3])
        # self.P_ContinuousLowerBound = np.array([-1] * len(self.P_ContinuousDims))
        # self.P_ContinuousUpperBound = np.array([1] * len(self.P_ContinuousDims))

        self.normalize = False
        self.y_offset = 0.5
        self.optimize_direction = 'maximize' # 'minimize'

    def convert_categorical(self, from_x, to_list, p_index):
        assert p_index in self.P_CategoricalDims
        assert len(to_list) == self.P_NumCategoryList[p_index]
        from_x_ = int(from_x)
        assert from_x_-from_x == 0
        return to_list[from_x_]

    def convert_continuous(self, from_x, to_range, p_index):
        assert p_index in self.P_ContinuousDims
        where = np.where(self.P_ContinuousDims==p_index)[0]

        new_range = to_range[1] - to_range[0]
        xx = ((from_x - self.P_ContinuousLowerBound[where]) * new_range / (self.P_ContinuousUpperBound[where] - self.P_ContinuousLowerBound[where])) + to_range[0]

        return float(xx)

    def compute(self, X, normalize=False):
        batch = X.shape[0]
        y_result_array = np.zeros(shape=(batch, 1))

        for b in range(batch):
            X = X[b]
            # 获取实验配置模板
            # Average among multiple different seeds
            conf_override = {
                "config.py->GlobalConfig-->seed": self.seed_list,
                "config.py->GlobalConfig-->note": self.note_list,
            }
            conf_override.update(self.get_device_conf())

            # AutoRL::learn hyper parameter ---> fuzzy
            p1 = self.convert_categorical(from_x = X[0], to_list=[0, 1, 2], p_index=0)
            p2 = self.convert_categorical(from_x = X[1], to_list=[0, 1, 2], p_index=1)
            p3 = self.convert_categorical(from_x = X[2], to_list=[0, 1, 2], p_index=2)
            p4 = self.convert_categorical(from_x = X[3], to_list=[0, 1, 2], p_index=3)
            p5 = self.convert_categorical(from_x = X[4], to_list=[0, 1, 2], p_index=4)
            p6 = self.convert_categorical(from_x = X[5], to_list=[0, 1, 2], p_index=5)


            conf_override.update({
                "ALGORITHM.experimental_conc_mt.foundation.py->AlgorithmConfig-->fuzzy_controller_param": [
                    [p1,p2,p3,p4,p5,p6],
                    [p1,p2,p3,p4,p5,p6],
                    [p1,p2,p3,p4,p5,p6],
                    [p1,p2,p3,p4,p5,p6],
                ]
            })


            self.internal_step_cnt += 1

            try:
                future_list = self.push_experiments_and_execute(conf_override)
                from UTIL.batch_exp import fetch_experiment_conclusion
                conclusion_list = fetch_experiment_conclusion(
                    step = self.internal_step_cnt,
                    future_list = future_list,
                    n_run_mode = self.n_run_mode)
            except:
                print('Experiment result timeout, trying again')
                # 如果失败再尝试一次，还不行就抛出错误
                future_list = self.push_experiments_and_execute(conf_override)
                from UTIL.batch_exp import fetch_experiment_conclusion
                conclusion_list = fetch_experiment_conclusion(
                    step = self.internal_step_cnt,
                    future_list = future_list,
                    n_run_mode = self.n_run_mode)
                

            def get_score(conclusion_list):
                score_list = []
                for c in conclusion_list:
                    conclusion_parsed = {}
                    # parse
                    for name, line, time in zip(c['name_list'],c['line_list'],c['time_list']):
                        conclusion_parsed[name] = line
                    s = conclusion_parsed['acc win ratio of=team-0']
                    score_list.append(s[-1])
                return score_list

            y_array = get_score(conclusion_list)
            y = np.array(y_array).mean()
            self.logger.debug(f'input {X}, [p1,p2,p3,p4,p5,p6] = {[p1,p2,p3,p4,p5,p6]}| output {y_array}, average {y}')
            y_result_array[b] = (y - self.y_offset)
            if self.optimize_direction == 'maximize':
                y_result_array[b] = -y_result_array[b]
            else:
                assert self.optimize_direction == 'minimize'

        return y_result_array

    def clean_profile_folder(self):
        import shutil, os
        if os.path.exists('PROFILE'):
            time_mark_only = time.strftime("%Y-%m-%d-%H-%M-%S", time.localtime())
            shutil.copytree('PROFILE', f'TEMP/PROFILE-{time_mark_only}')
            shutil.rmtree('PROFILE')


    def push_experiments_and_execute(self, conf_override):
        # copy the experiments
        import shutil, os
        shutil.copyfile(__file__, os.path.join(os.path.dirname(__file__), 'batch_experiment_backup.py'))
        # run experiments remotely
        from UTIL.batch_exp import run_batch_exp, fetch_experiment_conclusion
        print('Execute in server:', self.n_run_mode[0])
        self.clean_profile_folder()
        future = run_batch_exp(self.sum_note, self.n_run, self.n_run_mode, self.base_conf, conf_override, __file__, skip_confirm=True, master_folder='AutoRL')
        return future



    def get_device_conf(self):
        return {
            ########################################
            "ALGORITHM.experimental_conc_mt.foundation.py->AlgorithmConfig-->device_override":
                [
                    "cuda:0",
                    "cuda:0",
                    "cuda:1",
                    "cuda:1",
                ],
            "ALGORITHM.experimental_conc_mt.foundation.py->AlgorithmConfig-->gpu_party_override":
                [
                    "cuda0_party0", # 各子实验的party可以相同， 但每个实验的子队伍party建议设置为不同值
                    "cuda0_party0",
                    "cuda1_party0", # 各子实验的party可以相同， 但每个实验的子队伍party建议设置为不同值
                    "cuda1_party0",
                ],

            ########################################
            "TEMP.TEAM2.ALGORITHM.experimental_conc_mt.foundation.py->AlgorithmConfig-->device_override":
                [
                    "cuda:2",
                    "cuda:2",
                    "cuda:3",
                    "cuda:3",
                ],
            "TEMP.TEAM2.ALGORITHM.experimental_conc_mt.foundation.py->AlgorithmConfig-->gpu_party_override":
                [
                    "cuda2_party0",
                    "cuda2_party0",
                    "cuda3_party0",
                    "cuda3_party0",
                ],

        }



    # 获取基本的实验配置模板
    def obtain_base_experiment(self):
        return """
{
    "config.py->GlobalConfig": {
        "note": "Run1-Lr-Study",   // 实验存储路径
        "env_name": "dca_multiteam",  // 环境（任务名称）
        "env_path": "MISSION.dca_multiteam", 
        "draw_mode": "Img",
        "num_threads": 32,    // 环境并行数量
        "report_reward_interval": 32,
        "test_interval": 65536,
        "test_epoch": 256,
        "mt_parallel": true,
        "device": "cpu", // 使用哪张显卡
        "fold": "1",        // 使用的进程数量 = 环境并行数量/fold
        "n_parallel_frame": 50000000.0,
        "max_n_episode": 128.0,
        "seed": 22334, // 随机数种子
        "mt_act_order": "new_method",
        "backup_files": [
            "ALGORITHM/experimental_conc_mt",
            "MISSION/dca_multiteam"
        ]
    },
    "MISSION.dca_multiteam.collective_assult_parallel_run.py->ScenarioConfig": {
        "N_TEAM": 2,
        "N_AGENT_EACH_TEAM": [35, 35],
        "introduce_terrain": true,
        "terrain_parameters": [0.15, 0.2],
        "size": "5",
        "random_jam_prob": 0.05,
        "MaxEpisodeStep": 150,     // 时间限制， 胜利条件：尽量摧毁、存活
        "render": false,           // 高效渲染,只有0号线程环境会渲染
        "RewardAsUnity": true,
        "half_death_reward": true,
        "TEAM_NAMES": [
            "ALGORITHM.experimental_conc_mt.foundation->ReinforceAlgorithmFoundation",
            "TEMP.TEAM2.ALGORITHM.experimental_conc_mt.foundation->ReinforceAlgorithmFoundation",
        ]
    },
    "ALGORITHM.experimental_conc_mt.foundation.py->AlgorithmConfig": {
        "train_traj_needed": 32,
        "n_focus_on": 4,
        "lr": 0.0003,
        "ppo_epoch": 16,
        "lr_descent": false,
        "fuzzy_controller": true,
        "use_policy_resonance": false,
        "gamma": 0.99,
    },
    "TEMP.TEAM2.ALGORITHM.experimental_conc_mt.foundation.py->AlgorithmConfig": {
        "train_traj_needed": 32,
        "n_focus_on": 4,
        "lr": 0.0003,
        "ppo_epoch": 16,
        "lr_descent": false,
        "use_policy_resonance": false,
        "gamma": 0.99,
    },
}
"""













#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
###################################### 第三部分 ：贝叶斯优化主函数 ####################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################




def BayesianOptimisation(nth_trial, mcv, args):
    # n_trials: number of trials for the experiment
    kwargs = {}
    # random_seed_objective? what is 
    if args.random_seed_objective is not None:  
        assert 1 <= int(args.random_seed_objective) <= 25
        args.random_seed_objective -= 1

    # defining problem here
    f = HmpBayesianOptimizationInterface(seed=args.seed)

    n_categories = f.n_vertices
    problem_type = f.problem_type

    print('----- Starting trial %d / %d -----' % ((nth_trial + 1), args.n_trials))
    res = pd.DataFrame(np.nan, index=np.arange(int(args.max_iters*args.batch_size)),
                       columns=['Index', 'LastValue', 'BestValue', 'Time'])
    if args.infer_noise_var: noise_variance = None
    else: noise_variance = f.lamda if hasattr(f, 'lamda') else None

    if args.kernel_type is None:  kernel_type = 'mixed' if problem_type == 'mixed' else 'transformed_overlap'
    else: kernel_type = args.kernel_type

    if problem_type == 'mixed':
        optim = MixedOptimizer(
            f.P_NumCategoryList, 
            f.P_ContinuousLowerBound, 
            f.P_ContinuousUpperBound, 
            f.P_ContinuousDims, 
            f.P_CategoricalDims,
            n_init=args.n_init, 
            use_ard=args.ard, 
            acq=args.acq,
            kernel_type=kernel_type,
            noise_variance=noise_variance,
            **kwargs)
    else:
        optim = Optimizer(
            f.P_NumCategoryList, 
            n_init=args.n_init, 
            use_ard=args.ard, 
            acq=args.acq,
            kernel_type=kernel_type,
            noise_variance=noise_variance, **kwargs)
        
    for i in range(args.max_iters):
        start = time.time()

        x_next = optim.suggest(args.batch_size) 
        # x_next是？  x_next.shape = (1, 8)

        y_next = f.compute(x_next, normalize=f.normalize)
        # y_next是 ... y=f(x)？  y_next.shape = (1, )

        # Send an observation of a suggestion back to the optimizer
        optim.observe(x_next, y_next)

        # 时间    optim.suggest + f.compute + optim.observe
        end = time.time()


        # Save the full history: optim.casmopolitan.fX.shape = (iters, 1)
        if f.normalize:
            Y = np.array(optim.casmopolitan.fX) * f.std + f.mean
        else:
            Y = np.array(optim.casmopolitan.fX)


        if Y[:i].shape[0]:
            # res = pd.DataFrame
            mcv.rec(    i                   ,  'time'        )
            mcv.rec(    float(Y[-1])        ,  'this Y'      )
            mcv.rec(    float(np.min(Y[:i])),  'best Y'      )
            mcv.rec(    end-start           ,  'time cost'   )
            mcv.rec_show()
            # sequential
            if args.batch_size == 1:
                # iloc: index location，即对数据进行位置索引，从而在数据表中提取出相应的数据
                # [      iter,   本次Y,  最优Y,  本次时间花销         ]
                res.iloc[i, :] = [i, float(Y[-1]), float(np.min(Y[:i])), end-start]
            # batch
            else:
                for idx, j in enumerate(range(i*args.batch_size, (i+1)*args.batch_size)):
                    res.iloc[j, :] = [j, float(Y[-idx]), float(np.min(Y[:i*args.batch_size])), end-start]
            # x_next = x_next.astype(int)
            argmin = np.argmin(Y[:i*args.batch_size])

            print('Iter %d, Last X [%s] || fX:  %.4f || X_best: %s || fX_best: %.4f'
                  % (i, 
                     ''.join(['%.4f '%xx for xx in x_next.flatten()]),
                     float(Y[-1]),
                     ''.join([str(int(i)) for i in optim.casmopolitan.X[:i * args.batch_size][argmin].flatten()]),
                     Y[:i*args.batch_size][argmin]))

    if args.seed is not None:
        args.seed += 1




#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
###################################### 第四部分 ：BO参数选择，启动运行 ##############################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################

if __name__ == '__main__':

    # from THIRDPARTY.casmopolitan.mixed_test_func import *
    from THIRDPARTY.casmopolitan.bo.optimizer_mixed import MixedOptimizer
    from THIRDPARTY.casmopolitan.bo.optimizer import Optimizer
    import logging
    import argparse
    import pandas as pd
    import time, datetime
    from THIRDPARTY.casmopolitan.test_funcs.random_seed_config import *
    from VISUALIZE.mcom import mcom

    # Set up the objective function
    parser = argparse.ArgumentParser('Run Experiments')
    parser.add_argument('-p', '--problem', type=str, default='xgboost-mnist')
    parser.add_argument('--max_iters', type=int, default=150, help='Maximum number of BO iterations.')
    parser.add_argument('--lamda', type=float, default=1e-6, help='the noise to inject for some problems')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size for BO.')
    parser.add_argument('--n_trials', type=int, default=20, help='number of trials for the experiment')
    parser.add_argument('--n_init', type=int, default=20, help='number of initialising random points')
    parser.add_argument('--save_path', type=str, default='output/', help='save directory of the log files')
    parser.add_argument('--ard', action='store_true', help='whether to enable automatic relevance determination')
    parser.add_argument('-a', '--acq', type=str, default='ei', help='choice of the acquisition function.')
    parser.add_argument('--random_seed_objective', type=int, default=20, help='The default value of 20 is provided also in COMBO')
    parser.add_argument('-d', '--debug', action='store_true', help='Whether to turn on debugging mode (a lot of output will be generated).')
    parser.add_argument('--no_save', action='store_true', help='If activated, do not save the current run into a log folder.')
    parser.add_argument('--seed', type=int, default=None, help='**initial** seed setting')
    parser.add_argument('-k', '--kernel_type', type=str, default=None, help='specifies the kernel type')
    parser.add_argument('--infer_noise_var', action='store_true')
    args = parser.parse_args()
    options = vars(args)
    print(options)
    if args.debug: logging.basicConfig(level=logging.INFO)
    # Sanity checks
    assert args.acq in ['ucb', 'ei', 'thompson'], 'Unknown acquisition function choice ' + str(args.acq)
    mcv = mcom(path = './Temp', rapid_flush = True, draw_mode = 'Img', image_path = 'temp.jpg', tag = 'BayesianOptimisation')
    mcv.rec_init(color='g')
    BayesianOptimisation(0, mcv, args)

