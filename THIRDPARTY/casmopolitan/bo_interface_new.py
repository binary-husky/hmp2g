import numpy as np
import commentjson as json
import logging
from UTIL.exp_helper import read_json_handle_empty, write_json_handle_empty
from UTIL.tensor_ops import objdumpf, objloadf
from UTIL.colorful import *


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
    
    def compute(self, X, normalize=False):
        print亮绿(f'*** computing f({X}) ***')
        with FileLock(self.recall_cache_path+'.lock'): 
            X_Y_already_calculated = read_json_handle_empty(self.recall_cache_path)

        if str(X) in X_Y_already_calculated:
            print亮靛(f'*** find cache, skip computing f({X}) ***')
            return np.array(X_Y_already_calculated[str(X)])
        else:
            result = self.compute_(X, normalize=normalize)
            X_Y_already_calculated.update({str(X): result.tolist()})
            with FileLock(self.recall_cache_path+'.lock'): 
                write_json_handle_empty(self.recall_cache_path, X_Y_already_calculated)
            print亮绿(f'*** computing f({X}) done ***')
            return np.array(X_Y_already_calculated[str(X)])

    def __init__(self, MasterAutoRLKey='aurl', normalize=True, **kwargs):
        self.logger = logging.getLogger('my_logger')
        self.logger.setLevel(logging.DEBUG)
        # create a file handler
        handler = logging.FileHandler(f'AUTORL/{MasterAutoRLKey}/auto_rl_processing.log')
        self.recall_cache_path = f'AUTORL/{MasterAutoRLKey}/auto_rl_processing.cache'
        # set the logging level for the handler
        handler.setLevel(logging.DEBUG)
        # create a formatter
        formatter = logging.Formatter('%(asctime)s %(levelname)s: %(message)s', datefmt='%Y-%m-%d %H:%M:%S')
        # add the formatter to the handler
        handler.setFormatter(formatter)
        # add the handler to the self.logger
        self.logger.addHandler(handler)
        self.logger.debug('logger start')

        self.MasterAutoRLKey = MasterAutoRLKey
        self.summary_note = "BO_AUTORL"

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
###################################### 第三部分 ：贝叶斯优化主函数 ####################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################
#####################################################################################################################


import numpy as np
import commentjson as json
import logging, os
from UTIL.file_lock import FileLock
from THIRDPARTY.casmopolitan.bo.optimizer_mixed import MixedOptimizer
from THIRDPARTY.casmopolitan.bo.optimizer import Optimizer
import pandas as pd
import time, datetime
from THIRDPARTY.casmopolitan.test_funcs.random_seed_config import *
from VISUALIZE.mcom import mcom
    

def BayesianOptimisation(nth_trial, mcv, args, MasterAutoRLKey, interface):
    # n_trials: number of trials for the experiment
    kwargs = {}
    # random_seed_objective? what is 
    if args.random_seed_objective is not None:  
        assert 1 <= int(args.random_seed_objective) <= 25
        args.random_seed_objective -= 1

    # defining problem here
    f = interface

    n_categories = f.n_vertices
    problem_type = f.problem_type

    print('----- [%s] Starting trial %d / %d -----' % (MasterAutoRLKey, (nth_trial + 1), args.n_trials))
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
        
    if os.path.exists(f"{args.save_path}/opti.bo"):
        input('warning, loading and overriding old checkpoint! confirm?')
        optim = objloadf(f"{args.save_path}/opti.bo")
        begin_iter = objloadf(f"{args.save_path}/opti_step.bo")
    else:
        begin_iter = 0
        
    for i in range(begin_iter, args.max_iters):
        st_start = time.time()

        x_next = optim.suggest(args.batch_size) 
        # x_next是？  x_next.shape = (1, 8)
        st_end = time.time()

        y_next = f.compute(x_next, normalize=f.normalize)
        # y_next是 ... y=f(x)？  y_next.shape = (1, )

        # Send an observation of a suggestion back to the optimizer
        optim.observe(x_next, y_next)

        # 时间    optim.suggest + f.compute + optim.observe
        b_end = time.time()

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
            mcv.rec(    st_end-st_start           ,  'bo time cost'   )
            mcv.rec(    b_end-st_end           ,  'eval time cost'   )
            mcv.rec_show()
            # sequential
            if args.batch_size == 1:
                # iloc: index location，即对数据进行位置索引，从而在数据表中提取出相应的数据
                # [      iter,   本次Y,  最优Y,  本次时间花销         ]
                res.iloc[i, :] = [i, float(Y[-1]), float(np.min(Y[:i])), b_end-st_start]
            # batch
            else:
                for idx, j in enumerate(range(i*args.batch_size, (i+1)*args.batch_size)):
                    res.iloc[j, :] = [j, float(Y[-idx]), float(np.min(Y[:i*args.batch_size])), b_end-st_start]
            # x_next = x_next.astype(int)
            argmin = np.argmin(Y[:i*args.batch_size])

            print('Iter %d, Last X [%s] || fX:  %.4f || X_best: %s || fX_best: %.4f'
                  % (i, 
                     ''.join(['%.4f '%xx for xx in x_next.flatten()]),
                     float(Y[-1]),
                     ''.join([str(int(i)) for i in optim.casmopolitan.X[:i * args.batch_size][argmin].flatten()]),
                     Y[:i*args.batch_size][argmin]))
            objdumpf(optim, f"{args.save_path}/opti.bo")
            objdumpf(i, f"{args.save_path}/opti_step.bo")

        #             import pickle
        # res = pickle.dumps(optim)


    if args.seed is not None:
        args.seed += 1
