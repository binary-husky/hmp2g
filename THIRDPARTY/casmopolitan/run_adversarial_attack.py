# Xingchen Wan <xwan@robots.ox.ac.uk> | 2020
# The adversarial attack problem is slightly different from the other problems, since in other problems
# we directly optimise the objective function, but here we aim to achieve higher ASR (attack success rate)
# so a dedicated file to run this problem is made.

from mixed_test_func import *
from bo.optimizer_mixed import MixedOptimizer
import logging
import argparse
import os
import pickle
import pandas as pd
import time, datetime
from test_funcs.random_seed_config import *

# Set up the objective function
parser = argparse.ArgumentParser('Run Adversarial Attack')
parser.add_argument('--model_path', default=f'./mixed_test_func/AdvAttack/tf_models/',
                    help='the save path of the victim model. This is a required argument.')
parser.add_argument('-o', '--optimizer', type=str, default='localbo', help='Optimiser to use')
parser.add_argument('-n', '--n_trust_regions', type=int, default=1)
parser.add_argument('--starting_offset', type=int, default=0, help='Starting index of the image sequence.')
parser.add_argument('--max_iters', type=int, default=250, help='Maximum number of BO iterations.')
parser.add_argument('--batch_size', type=int, default=1, help='batch size for BO.')
parser.add_argument('--n_images', type=int, default=50,
                    help='number of images in CIFAR-10 validation set *that the CNN originally predicts correctly* to attack.')
parser.add_argument('--n_init', type=int, default=20)
parser.add_argument('--save_path', type=str, default='output/')
parser.add_argument('--ard', action='store_true')
parser.add_argument('-a', '--acq', type=str, default='thompson', help='choice of the acquisition function.')
parser.add_argument('-d', '--debug', action='store_true', help='Whether to turn on debugging mode (a lot of output will'
                                                               'be generated).')
parser.add_argument('--seed', type=int, default=None, help='**initial** seed setting')
parser.add_argument('--global_bo', action='store_true',
                    help='whether to use global BO modelling only (disabling the local BO modelling)')

args = parser.parse_args()
options = vars(args)
print(options)

# Time string will be used as the directory name
time_string = datetime.datetime.now()
time_string = time_string.strftime('%Y%m%d_%H%M%S')

if args.debug:
    logging.basicConfig(level=logging.INFO)

# Sanity checks
assert args.n_trust_regions >= 1
assert args.acq in ['ucb', 'ei', 'thompson'], 'Unknown acquisition function choice ' + str(args.acq)
assert args.optimizer in ['random', 'localbo', ], 'Unknown optimzer' + str(args.optimizer)

# Create the relevant folders, and save the arguments to reproduce the experiment, etc.
save_path = os.path.join(args.save_path, 'AdvAttack', time_string)
if not os.path.exists(save_path):
    os.makedirs(save_path)
option_file = open(save_path + "/command.txt", "w+")
option_file.write(str(options))
option_file.close()

# Each trial is a sample
for t in range(args.n_images):
    print('----- Starting image number %d / %d -----' % ((t + 1), args.n_images))
    # Create a trial-specific path
    save_path_trial = os.path.join(save_path, 'image-%d' % (t + args.starting_offset))
    if not os.path.exists(save_path_trial):
        os.mkdir(save_path_trial)

    # within each trial, we attack each of the target class (there are total 10 classes - 1 correct class)
    for i in range(9):
        # The second loop iterates through the image samples that are used as attack instances.
        kwargs = {
            'length_max_discrete': 43,
            'length_init_discrete': 15,
            'length_init': 0.6,
            'failtol': 20,
        }
        f = AdversarialAttack(args.model_path,
                              save_dir=save_path_trial,
                              target_label=i,
                              img_offset=t + args.starting_offset
                              )
        n_categories = f.n_vertices
        problem_type = f.problem_type

        res = pd.DataFrame(np.nan, index=np.arange(int(args.max_iters * args.batch_size)),
                           columns=['Index', 'LastValue', 'BestValue', 'Time'])

        if args.n_trust_regions != 1:
            raise NotImplementedError("Casmopolitan-M optimiser for mixed search space is not yet implemented.")

        optim = MixedOptimizer(f.config, f.lb, f.ub, f.continuous_dims, f.categorical_dims,
                               n_init=args.n_init, use_ard=args.ard, acq=args.acq,
                               global_bo=args.global_bo,
                               **kwargs)

        for k in range(args.max_iters):
            start = time.time()
            x_next = optim.suggest(args.batch_size)
            y_next = f.compute(x_next, normalize=f.normalize)
            optim.observe(x_next, y_next)
            end = time.time()
            Y = np.array(optim.casmopolitan.fX)
            if Y[:k].shape[0]:
                # Check for adversarial attack success
                if f.cnn.success:
                    print('!!!! --- Attack Succeeded --- !!!')
                    break

                # sequential
                if args.batch_size == 1:
                    res.iloc[k, :] = [k, float(Y[-1]), float(np.min(Y[:k])), end - start]
                # batch
                else:
                    for idx, j in enumerate(range(k * args.batch_size, (k + 1) * args.batch_size)):
                        res.iloc[j, :] = [j, float(Y[-idx]), float(np.min(Y[:k * args.batch_size])), end - start]
                # x_next = x_next.astype(int)
                argmin = np.argmin(Y[:k * args.batch_size])

                print('Iter %d, Last X %s; fX:  %.4f. X_best: %s, fX_best: %.4f'
                      % (k, x_next.flatten(),
                         float(Y[-1]),
                         ''.join([str(int(i)) for i in optim.casmopolitan.X[:k * args.batch_size][argmin].flatten()]),
                         Y[:k * args.batch_size][argmin]))
                # print(bo.bo.length_discrete)
            if save_path is not None:
                pickle.dump(res, open(os.path.join(save_path_trial, 'target_class-%d.pickle' % i), 'wb'))

    if args.seed is not None:
        args.seed += 1
