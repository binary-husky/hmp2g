from copy import deepcopy
from collections import OrderedDict

import numpy as np
import scipy.stats as ss
from .localbo_cat import CASMOPOLITANCat
from .localbo_mixed import CASMOPOLITANMixed
from .localbo_utils import from_unit_cube, latin_hypercube, to_unit_cube, ordinal2onehot, onehot2ordinal
import torch
import logging
from .optimizer import *
from torch.quasirandom import SobolEngine


class MixedOptimizer(Optimizer):

    def __init__(self, config,
                 lb, ub,
                 cont_dims,
                 cat_dims,
                 int_constrained_dims=None,
                 n_init: int = None,
                 wrap_discrete: bool = True,
                 guided_restart: bool = True,
                 **kwargs):
        """Build wrapper class to use an optimizer in benchmark.

        Parameters
        ----------
        config: list. e.g. [2, 3, 4, 5] -- denotes there are 4 categorical variables, with numbers of categories
            being 2, 3, 4, and 5 respectively.
        guided_restart: whether to fit an auxiliary GP over the best points encountered in all previous restarts, and
            sample the points with maximum variance for the next restart.
        """
        super(MixedOptimizer, self).__init__(config, n_init, wrap_discrete, guided_restart, **kwargs)

        self.kwargs = kwargs
        # Maps the input order.
        self.d_cat, self.d_cont = cat_dims, cont_dims
        self.true_dim = len(cont_dims) + len(cat_dims)

        # Number of one hot dimensions
        self.n_onehot = int(np.sum(config))
        # One-hot bounds
        self.lb = np.hstack((np.zeros(self.n_onehot), lb))
        self.ub = np.hstack((np.ones(self.n_onehot), ub))
        self.dim = len(self.lb)

        self.casmopolitan = CASMOPOLITANMixed(
            config=self.config,
            cat_dim=cat_dims,
            cont_dim=cont_dims,
            int_constrained_dims=int_constrained_dims,
            lb=lb[self.n_onehot:], ub=ub[self.n_onehot:],
            n_init=n_init if n_init is not None else 2 * self.true_dim + 1,
            max_evals=self.max_evals,
            batch_size=1,  # We need to update this later
            verbose=False,
            **kwargs
        )

    def restart(self):
        from .localbo_utils import train_gp

        if self.guided_restart and len(self.casmopolitan._fX):
            # Get the best index
            best_idx = self.casmopolitan._fX.argmin()
            # Obtain the best X and fX within each restart (bo._fX and bo._X get erased at each restart,
            # but bo.X and bo.fX always store the full history
            if self.best_fX_each_restart is None:
                self.best_fX_each_restart = deepcopy(self.casmopolitan._fX[best_idx])
                self.best_X_each_restart = deepcopy(self.casmopolitan._X[best_idx])
            else:
                self.best_fX_each_restart = np.vstack((self.best_fX_each_restart, deepcopy(self.casmopolitan._fX[best_idx])))
                self.best_X_each_restart = np.vstack((self.best_X_each_restart, deepcopy(self.casmopolitan._X[best_idx])))

            X_tr_torch = torch.tensor(self.best_X_each_restart, dtype=torch.float32).reshape(-1, self.true_dim)
            fX_tr_torch = torch.tensor(self.best_fX_each_restart, dtype=torch.float32).view(-1)
            # Train the auxiliary
            self.auxiliary_gp = train_gp(X_tr_torch, fX_tr_torch, False, 300,
                                         cat_dims=self.d_cat,
                                         cont_dims=self.d_cont,
                                         kern='mixed',
                                         noise_variance=self.kwargs[
                                             'noise_variance'] if 'noise_variance' in self.kwargs else None
                                         )
            # Generate random points in a Thompson-style sampling
            X_init = latin_hypercube(self.casmopolitan.n_cand, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            # Isolate the continuous part and categorical part
            X_init_cat, X_init_cont = X_init[:, :self.n_onehot], X_init[:, self.n_onehot:]
            if self.wrap_discrete:
                X_init_cat = self.warp_discrete(X_init_cat, )
            X_init_cat = onehot2ordinal(X_init_cat, self.cat_dims)
            # Put the two parts back by a hstack
            X_init = np.hstack((X_init_cat, X_init_cont))

            with torch.no_grad():
                self.auxiliary_gp.eval()
                X_init_torch = torch.tensor(X_init, dtype=torch.float32)
                # LCB sampling
                y_cand_mean, y_cand_var = self.auxiliary_gp(
                    X_init_torch).mean.cpu().detach().numpy(), self.auxiliary_gp(
                    X_init_torch).variance.cpu().detach().numpy()
                y_cand = y_cand_mean - 1.96 * np.sqrt(y_cand_var)

                # Maximum variance sampling
                # y_init = self.auxiliary_gp(X_init_torch).variance.cpu().detach().numpy()
            self.X_init = np.ones((self.casmopolitan.n_init, self.true_dim))
            indbest = np.argmin(y_cand)
            # cThe initial trust region centre for the restart
            centre = deepcopy(X_init[indbest, :])
            # Separate the continuous and categorical parts of the centre.
            centre_cat, centre_cont = centre[self.d_cat], centre[self.d_cont]

            # Generate random samples around the continuous centre similar to the original TuRBO
            centre_cont = centre_cont[None, :]
            lb = np.clip(centre_cont - self.casmopolitan.length / 2.0, self.lb[self.n_onehot:], self.ub[self.n_onehot:])
            ub = np.clip(centre_cont + self.casmopolitan.length / 2.0, self.lb[self.n_onehot:], self.ub[self.n_onehot:])
            seed = np.random.randint(int(1e6))
            sobol = SobolEngine(len(self.d_cont), scramble=True, seed=seed)
            pert = sobol.draw(self.casmopolitan.n_init).to(dtype=torch.float32).cpu().detach().numpy()
            pert = lb + (ub - lb) * pert
            prob_perturb = min(20.0 / len(self.d_cont), 1.0)
            mask = np.random.rand(self.casmopolitan.n_init, len(self.d_cont)) <= prob_perturb
            ind = np.where(np.sum(mask, axis=1) == 0)[0]
            mask[ind, np.random.randint(0, len(self.d_cont) - 1, size=len(ind))] = 1
            X_init_cont = centre_cont.copy() * np.ones((self.casmopolitan.n_init, len(self.d_cont)))
            X_init_cont[mask] = pert[mask]

            # Generate the random samples around the categorical centre similar to the discrete CASMOPLTN
            X_init_cat = []
            for i in range(self.casmopolitan.n_init):
                cat_sample = deepcopy(
                    random_sample_within_discrete_tr_ordinal(centre_cat, self.casmopolitan.length_init_discrete, self.config),
                )
                X_init_cat.append(cat_sample)
            X_init_cat = np.array(X_init_cat)
            self.X_init = np.hstack((X_init_cat, X_init_cont))

            self.casmopolitan._restart()
            self.casmopolitan._X = np.zeros((0, self.casmopolitan.dim))
            self.casmopolitan._fX = np.zeros((0, 1))
            del X_tr_torch, fX_tr_torch, X_init_torch

        else:
            # If guided restart is not enabled, simply sample a number of points equal to the number of evaluated
            self.casmopolitan._restart()
            self.casmopolitan._X = np.zeros((0, self.casmopolitan.dim))
            self.casmopolitan._fX = np.zeros((0, 1))
            X_init = latin_hypercube(self.casmopolitan.n_init, self.dim)
            X_init = from_unit_cube(X_init, self.lb, self.ub)
            X_init_cat, X_init_cont = X_init[:, :self.n_onehot], X_init[:, self.n_onehot:]
            if self.wrap_discrete:
                X_init_cat = self.warp_discrete(X_init_cat, )
            X_init_cat = onehot2ordinal(X_init_cat, self.cat_dims)
            # Put the two parts back by a hstack
            self.X_init = np.hstack((X_init_cat, X_init_cont))

    def suggest(self, n_suggestions=1):
        if self.batch_size is None:  # Remember the batch size on the first call to suggest
            self.batch_size = n_suggestions
            self.casmopolitan.batch_size = n_suggestions
            self.casmopolitan.n_init = max([self.casmopolitan.n_init, self.batch_size])
            self.restart()

        X_next = np.zeros((n_suggestions, self.true_dim))

        # Pick from the initial points
        n_init = min(len(self.X_init), n_suggestions)
        if n_init > 0:
            X_next[:n_init, :] = deepcopy(self.X_init[:n_init, :])
            self.X_init = self.X_init[n_init:, :]  # Remove these pending points

        # Get remaining points from TuRBO
        n_adapt = n_suggestions - n_init
        if n_adapt > 0:
            if len(self.casmopolitan._X) > 0:  # Use random points if we can't fit a GP
                X = deepcopy(self.casmopolitan._X)
                # Pre-process the continuous dimensions
                X[:, self.casmopolitan.cont_dims] = to_unit_cube(X[:, self.casmopolitan.cont_dims], self.lb[self.n_onehot:],
                                                          self.ub[self.n_onehot:])
                fX = copula_standardize(deepcopy(self.casmopolitan._fX).ravel())  # Use Copula
                next = self.casmopolitan._create_and_select_candidates(X, fX, length=self.casmopolitan.length_discrete,
                                                                       n_training_steps=300, hypers={})[-n_adapt:, :]
                next[:, self.casmopolitan.cont_dims] = from_unit_cube(next[:, self.casmopolitan.cont_dims], self.lb[self.n_onehot:],
                                                               self.ub[self.n_onehot:])
                X_next[-n_adapt:, :] = next

                # Unwarp the suggestions
        suggestions = X_next
        return suggestions
