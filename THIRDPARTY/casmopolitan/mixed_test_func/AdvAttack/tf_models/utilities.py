# @author: Robin Ru (robin@robots.ox.ac.uk)

import numpy as np
from pyDOE import lhs
from scipy.optimize import fmin_l_bfgs_b
import torch
import torch.nn.functional as F
# import numpy as np



def upsample_projection(dim_reduction, X_low, low_dim, high_dim, nchannel=1, align_corners=True):
    """
    Various upsampling methods: CLUSTER,  BILI (bilinear), NN (bilinear), BICU (bicubic)

    :param dim_reduction: dimension reduction method used in upsampling
    :param X_low: input data in low dimension
    :param low_dim: the low dimension
    :param high_dim: the high dimension
    :param nchannel: number of image channels
    :param align_corners: align corner option for interpolate
    :return X_high: input data in high dimension
    """

    if dim_reduction == 'CLUSTER':
        n = X_low.shape[0]
        high_int = int(np.sqrt(high_dim))
        low_int = int(np.sqrt(low_dim))
        ratio = np.floor(high_dim / low_dim)
        low_edge = int(np.floor(np.sqrt(ratio)) + 1)
        high_edge = low_edge * low_int
        high_obs = np.zeros((n, high_edge, high_edge, nchannel))
        for ch in range(nchannel):
            for row in range(low_int):
                for col in range(low_int):
                    k = row * low_int + col
                    for jj in range(low_edge):
                        for ii in range(low_edge):
                            high_obs[:, row * low_edge + jj, col * low_edge + ii, ch] = X_low[:, ch * low_dim + k]

        high_obs = high_obs[:, :high_int, :high_int, :]
        X_high = high_obs.reshape(X_low.shape[0], high_dim * nchannel)

    else:

        if dim_reduction == 'BILI':
            upsample_mode = 'bilinear'
        elif dim_reduction == 'NN':
            upsample_mode = 'nearest'
            align_corners = None
        elif dim_reduction == 'BICU':
            upsample_mode = 'bicubic'

        X_low_tensor = torch.FloatTensor(X_low).view(X_low.shape[0], nchannel, int(np.sqrt(low_dim)),
                                                     int(np.sqrt(low_dim)))
        X_high_tensor_resize = F.interpolate(X_low_tensor,
                                             size=(int(np.sqrt(high_dim)), int(np.sqrt(high_dim))), mode=upsample_mode,
                                             align_corners=align_corners)
        X_high = X_high_tensor_resize.data.numpy().squeeze().reshape(X_low.shape[0], high_dim * nchannel)
    return X_high


def downsample_projection(dim_reduction, X_high, low_dim, high_dim, nchannel=1, align_corners=True):
    """
    Various downsampling methods: CLUSTER,  BILI (bilinear), NN (bilinear), BICU (bicubic)

    :param dim_reduction: dimension reduction method used in upsampling
    :param X_high: input data in high dimension
    :param low_dim: the low dimension
    :param high_dim: the high dimension
    :param nchannel: number of image channels
    :param align_corners: align corner option for interpolate
    :return X_low: input data in low dimension
    """

    if dim_reduction == 'CLUSTER':

        n = X_high.shape[0]
        high_int = int(np.sqrt(high_dim))
        X_high = np.reshape(X_high, (n, high_int, high_int, nchannel))
        low_int = int(np.sqrt(low_dim))
        ratio = np.floor(high_dim / low_dim)
        low_edge = int(np.floor(np.sqrt(ratio)) + 1)
        high_edge = low_edge * low_int
        npadd = high_edge - high_int
        high_obs = np.zeros((n, high_edge, high_edge, nchannel))
        high_obs[:, :high_int, :high_int, :] = X_high
        for kk in range(npadd):
            high_obs[:, high_int + kk, :high_int, :] = high_obs[:, high_int - 1, :high_int, :]
        for kk in range(npadd):
            high_obs[:, :, high_int + kk, :] = high_obs[:, :, high_int - 1, :]
        low_obs = np.zeros((n, low_int, low_int, nchannel))
        for ch in range(nchannel):
            for row in range(low_int):
                for col in range(low_int):
                    for point in range(n):
                        low_obs[point, row, col, ch] = np.mean(
                            high_obs[point, (row * low_edge):((row + 1) * low_edge),
                            (col * low_edge):((col + 1) * low_edge), ch])
        X_low_tensor_resize = low_obs
        X_low = X_low_tensor_resize.reshape(X_high.shape[0], low_dim * nchannel)

    else:
        if dim_reduction == 'BILI':
            upsample_mode = 'bilinear'
        elif dim_reduction == 'NN':
            upsample_mode = 'nearest'
            align_corners = None
        elif dim_reduction == 'BICU':
            upsample_mode = 'bicubic'

        X_high_tensor = torch.FloatTensor(X_high).view(X_high.shape[0], nchannel, int(np.sqrt(high_dim)),
                                                       int(np.sqrt(high_dim)))
        X_low_tensor_resize = F.interpolate(X_high_tensor,
                                            size=(int(np.sqrt(low_dim)), int(np.sqrt(low_dim))), mode=upsample_mode,
                                            align_corners=align_corners)
        X_low = X_low_tensor_resize.data.numpy().squeeze().reshape(X_high.shape[0], low_dim * nchannel)
    return X_low


def generate_attack_data_set(data, num_sample, img_offset, model, attack_type="targeted", random_target_class=None,
                             shift_index=False):
    """
    Generate the data for conducting attack. Only select the data being classified correctly.
    """
    orig_img = []
    orig_labels = []
    target_labels = []
    orig_img_id = []

    pred_labels = np.argmax(model.model.predict(data.test_data), axis=1)
    true_labels = np.argmax(data.test_labels, axis=1)
    correct_data_indices = np.where([1 if x == y else 0 for (x, y) in zip(pred_labels, true_labels)])

    print(
        "Total testing data:{}, correct classified data:{}".format(len(data.test_labels), len(correct_data_indices[0])))

    data.test_data = data.test_data[correct_data_indices]
    data.test_labels = data.test_labels[correct_data_indices]
    true_labels = true_labels[correct_data_indices]

    np.random.seed(img_offset)  # for parallel running
    class_num = data.test_labels.shape[1]
    for sample_index in range(num_sample):

        if attack_type == "targeted":
            if random_target_class is not None:
                np.random.seed(0)  # for parallel running
                # randomly select one class to attack, except the true labels
                # seq_imagenet = np.random.choice(random_target_class, 100)
                seq_imagenet = [1, 4, 6, 8, 9, 13, 15, 16, 19, 20, 22, 24, 25, 27, 28, 30, 34, 35, 36, 37, 38, 44, 49,
                                51, 56, 59, 60, 61, 62, 63, 67, 68, 70, 71, 74, 75, 76, 77, 78, 79, 82, 84, 85, 87, 88,
                                91, 94, 96, 97, 99]

                seq = [seq_imagenet[img_offset + sample_index]]
                while seq == true_labels[img_offset + sample_index]:
                    seq = np.random.choice(random_target_class, 1)

            else:
                seq = list(range(class_num))
                seq.remove(true_labels[img_offset + sample_index])

            for s in seq:
                if shift_index and s == 0:
                    s += 1
                orig_img.append(data.test_data[img_offset + sample_index])
                target_labels.append(np.eye(class_num)[s])
                orig_labels.append(data.test_labels[img_offset + sample_index])
                orig_img_id.append(img_offset + sample_index)

        elif attack_type == "untargeted":
            orig_img.append(data.test_data[img_offset + sample_index])
            target_labels.append(data.test_labels[img_offset + sample_index])
            orig_labels.append(data.test_labels[img_offset + sample_index])
            orig_img_id.append(img_offset + sample_index)

    orig_img = np.array(orig_img)
    target_labels = np.array(target_labels)
    orig_labels = np.array(orig_labels)
    orig_img_id = np.array(orig_img_id)

    return orig_img, target_labels, orig_labels, orig_img_id


def model_prediction(model, inputs):
    prob = model.model.predict(inputs)
    predicted_class = np.argmax(prob)
    prob_str = np.array2string(prob).replace('\n', '')
    return prob, predicted_class, prob_str


def get_init_data(obj_func, n_init, bounds, method='lhs'):
    """
    Generate initial data for starting BO

    :param obj_func:
    :param n_init: number of initial data
    :param bounds: input space bounds
    :param method: random sample method
    :return x_init: initial input data
    :return y_init: initial output data
    """
    noise_var = 1.0e-10
    d = bounds.shape[0]

    if method == 'lhs':
        x_init = lhs(d, n_init) * (bounds[0, 1] - bounds[0, 0]) + bounds[0, 0]
    else:
        x_init = np.random.uniform(bounds[0, 0], bounds[0, 1], (n_init, d))
    f_init = obj_func(x_init)
    y_init = f_init + np.sqrt(noise_var) * np.random.randn(n_init, 1)
    return x_init, y_init


def subset_select(X_all, Y_all, select_metric='RAND'):
    """
    Select the subset of the observed data for sparse GP
    :param X_all: observed input data
    :param Y_all: observed output data
    :param select_metric: subset selection criterion
    :return X_ob: subset observed input data
    :return Y_ob: subset observed output data
    """

    N_ob = X_all.shape[0]

    if N_ob <= 500:
        X_ob = X_all
        Y_ob = Y_all
    else:
        # selecting subset if the number of observed data exceeds 500
        if N_ob > 500 and N_ob <= 1000:
            subset_size = 500
        else:
            subset_size = 1000

        print(f'use subset={subset_size} of observed data via {select_metric}')
        if 'SUBRAND' in select_metric:
            x_indices_random = np.random.permutation(range(N_ob))
            x_subset_indices = x_indices_random[:subset_size]
        elif 'SUBGREEDY' in select_metric:
            pseudo_prob_nexp = np.exp(-(Y_all - Y_all.min()))
            pseudo_prob = pseudo_prob_nexp / np.sum(pseudo_prob_nexp)
            x_subset_indices = np.random.choice(N_ob, subset_size, p=pseudo_prob.flatten(), replace=False)
        X_ob = X_all[x_subset_indices, :]
        Y_ob = Y_all[x_subset_indices, :]

    return X_ob, Y_ob


def subset_select_for_learning(X_all, Y_all, select_metric='ADDRAND'):
    """
    Select the subset of the observed data for sparse GP used only in the phase of learning dr or decomposition
    :param X_all: observed input data
    :param Y_all: observed output data
    :param select_metric: subset selection criterion
    :return X_ob: subset observed input data
    :return Y_ob: subset observed output data
    """
    N_ob = X_all.shape[0]
    subset_size = 200
    pseudo_prob_nexp = np.exp(-(Y_all - Y_all.min()))
    pseudo_prob = pseudo_prob_nexp / np.sum(pseudo_prob_nexp)
    x_subset_indices = np.random.choice(N_ob, subset_size, p=pseudo_prob.flatten(), replace=False)
    X_ob = X_all[x_subset_indices, :]
    Y_ob = Y_all[x_subset_indices, :]
    return X_ob, Y_ob


def optimise_acqu_func(acqu_func, bounds, X_ob, func_gradient=True, gridSize=10000, n_start=5):
    """
    Optimise acquisition function built on GP model

    :param acqu_func: acquisition function
    :param bounds: input space bounds
    :param X_ob: observed input data
    :param func_gradient: whether to use the acquisition function gradient in optimisation
    :param gridSize: random grid size
    :param n_start: the top n_start points in the random grid search from which we do gradient-based local optimisation
    :return np.array([opt_location]): global optimum input
    :return f_opt: global optimum
    """

    # Turn the acquisition function to be - acqu_func for minimisation
    target_func = lambda x: - acqu_func._compute_acq(x)

    # Define a new function combingin the acquisition function and its derivative
    def target_func_with_gradient(x):
        acqu_f, dacqu_f = acqu_func._compute_acq_withGradients(x)
        return -acqu_f, -dacqu_f

    # Define bounds for the local optimisers
    bounds_opt = list(bounds)

    # Create grid for random search
    d = bounds.shape[0]
    Xgrid = np.tile(bounds[:, 0], (gridSize, 1)) + np.tile((bounds[:, 1] - bounds[:, 0]),
                                                           (gridSize, 1)) * np.random.rand(gridSize, d)
    Xgrid = np.vstack((Xgrid, X_ob))
    results = target_func(Xgrid)

    # Find the top n_start candidates from random grid search to perform local optimisation
    top_candidates_idx = results.flatten().argsort()[
                         :n_start]  # give the smallest n_start values in the ascending order
    random_starts = Xgrid[top_candidates_idx]
    f_min = results[top_candidates_idx[0]]
    opt_location = random_starts[0]

    # Print('done random grid search')
    # Perform multi-start gradient-based optimisation
    for random_start in random_starts:
        if func_gradient:
            x, f_at_x, info = fmin_l_bfgs_b(target_func_with_gradient, random_start, bounds=bounds_opt,
                                            approx_grad=False, maxiter=5000)
        else:
            x, f_at_x, info = fmin_l_bfgs_b(target_func, random_start, bounds=bounds_opt,
                                            approx_grad=True, maxiter=5000)
        if f_at_x < f_min:
            f_min = f_at_x
            opt_location = x

    f_opt = - f_min

    return np.array([opt_location]), f_opt


def optimise_acqu_func_mledr(acqu_func, bounds, X_ob, func_gradient=True, gridSize=10000, n_start=5):
    """
    Optimise acquisition function built on GP- model with learning dr

    :param acqu_func: acquisition function
    :param bounds: input space bounds
    :param X_ob: observed input data
    :param func_gradient: whether to use the acquisition function gradient in optimisation
    :param gridSize: random grid size
    :param n_start: the top n_start points in the random grid search from which we do gradient-based local optimisation
    :return np.array([opt_location]): global optimum input
    :return f_opt: global optimum
    """

    # Turn the acquisition function to be - acqu_func for minimisation
    target_func = lambda x: - acqu_func._compute_acq(x)

    # Define a new function combingin the acquisition function and its derivative
    def target_func_with_gradient(x):
        acqu_f, dacqu_f = acqu_func._compute_acq_withGradients(x)
        return -acqu_f, -dacqu_f

    # Define bounds for the local optimisers based on the optimal dr
    nchannel = acqu_func.model.nchannel
    d = acqu_func.model.opt_dr
    d_vector = int(acqu_func.model.opt_dr ** 2 * nchannel)
    bounds = np.vstack([[-1, 1]] * d_vector)

    # Project X_ob to optimal dr learnt
    h_d = int(X_ob.shape[1] / acqu_func.model.nchannel)
    X_ob_d_r = downsample_projection(acqu_func.model.dim_reduction, X_ob, int(d ** 2), h_d, nchannel=nchannel,
                                     align_corners=True)

    # Create grid for random search but split the grid into n_batches to avoid memory overflow
    good_results_list = []
    random_starts_candidates_list = []
    n_batch = 5
    gridSize_sub = int(gridSize / n_batch)
    for x_grid_idx in range(n_batch):
        Xgrid_sub = np.tile(bounds[:, 0], (gridSize_sub, 1)) + np.tile((bounds[:, 1] - bounds[:, 0]),
                                                                       (gridSize_sub, 1)) * np.random.rand(gridSize_sub,
                                                                                                           d_vector)
        if x_grid_idx == 0:
            Xgrid_sub = np.vstack((Xgrid_sub, X_ob_d_r))
        results = target_func(Xgrid_sub)
        top_candidates_sub = results.flatten().argsort()[:5]  # give the smallest n_start values in the ascending order
        random_starts_candidates = Xgrid_sub[top_candidates_sub]
        good_results = results[top_candidates_sub]
        random_starts_candidates_list.append(random_starts_candidates)
        good_results_list.append(good_results)

    # Find the top n_start candidates from random grid search to perform local optimisation
    results = np.vstack(good_results_list)
    X_random_starts = np.vstack(random_starts_candidates_list)
    top_candidates_idx = results.flatten().argsort()[
                         :n_start]  # give the smallest n_start values in the ascending order
    random_starts = X_random_starts[top_candidates_idx]
    f_min = results[top_candidates_idx[0]]
    opt_location = random_starts[0]

    # Perform multi-start gradient-based optimisation
    for random_start in random_starts:
        if func_gradient:
            x, f_at_x, info = fmin_l_bfgs_b(target_func_with_gradient, random_start, bounds=bounds,
                                            approx_grad=False, maxiter=5000)
        else:
            x, f_at_x, info = fmin_l_bfgs_b(target_func, random_start, bounds=bounds,
                                            approx_grad=True, maxiter=5000)
        if f_at_x < f_min:
            f_min = f_at_x
            opt_location = x

    f_opt = -f_min

    return np.array([opt_location]), f_opt


def optimise_acqu_func_additive(acqu_func, bounds, X_ob, func_gradient=True, gridSize=5000, n_start=1, nsubspace=12):
    """
    Optimise acquisition function built on ADDGP model

    :param acqu_func: acquisition function
    :param bounds: input space bounds
    :param X_ob: observed input data
    :param func_gradient: whether to use the acquisition function gradient in optimisation
    :param gridSize: random grid size
    :param n_start: the top n_start points in the random grid search from which we do gradient-based local optimisation
    :param nsubspace: number of subspaces in the decomposition
    :return np.array([opt_location]): global optimum input
    :return f_opt: global optimum
    """

    # Create grid for random search
    d = bounds.shape[0]
    Xgrid = np.tile(bounds[:, 0], (gridSize, 1)) + np.tile((bounds[:, 1] - bounds[:, 0]),
                                                           (gridSize, 1)) * np.random.rand(gridSize, d)
    Xgrid = np.vstack((Xgrid, X_ob))
    f_opt_join = []

    # Get the learnt decomposition
    active_dims_list = acqu_func.model.active_dims_list
    opt_location_join_array = np.zeros(d)

    # Optimise the acquisition function in each subspace separately in sequence
    for i in range(nsubspace):
        print(f'start optimising subspace{i}')

        # Define the acquisition function for the subspace and turn it to be - acqu_func for minimisation
        def target_func(x_raw):
            x = np.atleast_2d(x_raw)
            N = x.shape[0]
            if x.shape[1] == d:
                x_aug = x.copy()
            else:
                x_aug = np.zeros([N, d])
                x_aug[:, active_dims_list[i]] = x
            return - acqu_func._compute_acq(x_aug, subspace_id=i)

        # Define a new function combingin the acquisition function and its derivative
        def target_func_with_gradient(x_raw):
            x = np.atleast_2d(x_raw)
            N = x.shape[0]
            if x.shape[1] == d:
                x_aug = x.copy()
            else:
                x_aug = np.zeros([N, d])
                x_aug[:, active_dims_list[i]] = x

            acqu_f, dacqu_f = acqu_func._compute_acq_withGradients(x_aug, subspace_id=i)
            return -acqu_f, -dacqu_f

        # Find the top n_start candidates from random grid search to perform local optimisation
        results = target_func(Xgrid)
        top_candidates_idx = results.flatten().argsort()[
                             :n_start]  # give the smallest n_start values in the ascending order
        random_starts = Xgrid[top_candidates_idx][:, active_dims_list[i]]
        f_min = results[top_candidates_idx[0]]
        opt_location = random_starts[0]

        # Define bounds for the local optimisers for the subspace
        bounds_opt_sub = list(bounds[active_dims_list[i], :])
        for random_start in random_starts:
            if func_gradient:
                x, f_at_x, info = fmin_l_bfgs_b(target_func_with_gradient, random_start, bounds=bounds_opt_sub,
                                                approx_grad=False, maxiter=5000)
            else:
                x, f_at_x, info = fmin_l_bfgs_b(target_func, random_start, bounds=bounds_opt_sub,
                                                approx_grad=True, maxiter=5000)
            if f_at_x < f_min:
                f_min = f_at_x
                opt_location = x

        f_opt = -f_min
        opt_location_join_array[active_dims_list[i]] = opt_location
        f_opt_join.append(f_opt)

    f_opt_join_sum = np.sum(f_opt_join)

    return np.atleast_2d(opt_location_join_array), f_opt_join_sum


def optimise_acqu_func_for_NN(acqu_func, bounds, X_ob, func_gradient=False, gridSize=20000, num_chunks=10):
    """
    Optimise acquisition function built on BNN surrogate

    :param acqu_func: acquisition function
    :param bounds: input space bounds
    :param X_ob: observed input data
    :param func_gradient: whether to use the acquisition function gradient in optimisation
    :param gridSize: random grid size
    :param num_chunks: divide the random grid into a number of chunks to avoid memory overflow
    :return np.array([opt_location]): global optimum input
    :return f_opt: global optimum
    """

    # Turn the acquisition function to be - acqu_func for minimisation
    target_func = lambda x: - acqu_func._compute_acq(x)

    # Create grid for random search but split the grid into num_chunks to avoid memory overflow
    d = bounds.shape[0]
    Xgrid = np.tile(bounds[:, 0], (gridSize, 1)) + np.tile((bounds[:, 1] - bounds[:, 0]),
                                                           (gridSize, 1)) * np.random.rand(gridSize, d)
    X_chunks = np.split(Xgrid, num_chunks)
    x_ob_chunk = X_ob[-int(gridSize / num_chunks):, :]
    X_chunks.append(x_ob_chunk)
    Xgrid = np.vstack((Xgrid, x_ob_chunk))
    results_list = [target_func(x_chunk) for x_chunk in X_chunks]
    results = np.vstack(results_list)

    # Find the top candidate from random grid search
    top_candidates_idx = results.flatten().argsort()[:1]  # give the smallest n_start values in the ascending order
    random_starts = Xgrid[top_candidates_idx]
    print('done selecting rs')
    f_min = results[top_candidates_idx[0]]
    opt_location = random_starts[0]
    f_opt = -f_min

    return np.array([opt_location]), f_opt
