# @author: Robin Ru (robin@robots.ox.ac.uk)
# This code is modified based on the code by Moustafa Alzantot
# used for his paper "GenAttack: Practical Black-box Attacks with Gradient-Free".

import os

import numpy as np

from mixed_test_func.AdvAttack.tf_models.setup_cifar import CIFAR, CIFARModel
from mixed_test_func.AdvAttack.tf_models.setup_mnist import MNIST, MNISTModel
from mixed_test_func.AdvAttack.tf_models.utilities import upsample_projection, generate_attack_data_set


class CNN(object):
    def __init__(self, dataset_name, num_img=1, img_offset=0, epsilon=0.05, rescale=True,
                 low_dim=None, high_dim=784, obj_metric=2, results_folder=None, directory=None):
        """
        CNN Classifier on MNIST, CIFAR10 and ImageNet

        :param dataset_name: image dataset name
        :param num_img: number of images to be attacked (default=1)
        :param img_offset: the image id e.g. img_offset=4 means 4th image in the correctly classified test set
        :param epsilon: maximum perturbation
        :param rescale: rescale the adversarial image to the range of the original image
        :param low_dim: reduced dimension (drxdr)
        :param high_dim: image dimension (e.g. 32x32 for CIFAR10) or high-dimensional search space for imagenet (96x96)
        :param obj_metric: Metric used to compute objective function (default = 2)
        :param results_folder: results saving folder directory
        :param directory: path for target model to be attacked and the image dataset
        """

        self.epsilon = epsilon
        self.dataset_name = dataset_name
        self.num_img = num_img
        self.low_dim = low_dim
        self.high_dim = high_dim
        self.objective_metric = obj_metric
        self.results_folder = results_folder
        self.rescale = rescale
        folder_path = directory
        self.dim_reduction_method_list = ['BILI', 'NN', 'BICU']

        # Get the image data and the target model
        if 'mnist' in dataset_name:
            self.d1 = 28
            self.nchannel = 1
            self.dataset_name = 'mnist'
            self.total_classes = 10
            data, model = MNIST(folder_path), MNISTModel(f'{folder_path}models/mnist', use_softmax=True)

        elif 'cifar10' in dataset_name:
            self.d1 = 32
            self.nchannel = 3
            self.total_classes = 10
            self.dataset_name = 'cifar10'
            data, model = CIFAR(folder_path), CIFARModel(f'{folder_path}models/cifar', use_softmax=True)

        random_target = False
        shift_index = False
        attack_type = 'targeted'
        print(f'Loading data and classification model: {self.dataset_name}')

        # Get the correctly classified image data to be attacked
        if random_target:
            # random target on all possible classes
            class_num = data.test_labels.shape[1]
            all_orig_img, all_target_labels, all_orig_labels, all_orig_img_id = generate_attack_data_set(
                data, num_img, img_offset, model, attack_type=attack_type, random_target_class=list(range(class_num)),
                shift_index=shift_index)
        else:
            all_orig_img, all_target_labels, all_orig_labels, all_orig_img_id = generate_attack_data_set(
                data, num_img, img_offset, model, attack_type=attack_type, shift_index=shift_index)

        self.blackbox_model = model.model
        self.all_orig_img = all_orig_img
        self.all_target_labels = all_target_labels
        self.all_orig_img_id = all_orig_img_id
        self.all_orig_labels_int = np.argmax(all_orig_labels, 1)

        # Check the original image labels
        if (dataset_name == 'mnist' or dataset_name == 'cifar10'):
            if (self.all_orig_labels_int[0] != self.all_orig_labels_int[-1]):
                assert False

    def get_data_sample(self, i=0):
        """
        :param i: the attack target label id
        """

        if i > (self.total_classes - 2):
            assert False

        # Get the original image label and save original image
        self.X_origin = self.all_orig_img[i:i + 1]
        self.orig_img_id = self.all_orig_img_id[i:i + 1][0]
        self.input_label = self.all_orig_labels_int[i:i + 1][0]
        X_orig_img_file = os.path.join(self.results_folder,
                                       f'X_{self.dataset_name}_origin_{self.input_label}_id{self.orig_img_id}')
        np.save(X_orig_img_file, self.X_origin)

        # Get the attack target label
        target_label_vector = self.all_target_labels[i:i + 1]
        self.target_label = np.argmax(target_label_vector, 1)

    def evaluate(self, delta_vector):
        """
        :param delta_vector: adversarial perturbation in the range of [-epsilon, epsilon]
        :return score:  = log_p_max - log_p_target [N] if obj_metric=1;
                        = log_sum_{j \not target} p_j - log_p_target [N] if obj_metric=2 (default)
                        both to be minimised
        """

        # Add adversarial delta to the original image
        delta = delta_vector.reshape(-1, self.d1, self.d1, self.nchannel)
        X_adv = self.X_origin.copy() + delta

        # Rescale to the value range of the original image
        if self.rescale:
            X_adv = X_adv / (1 + 2 * self.epsilon)
        else:
            X_adv = X_adv.clip(-0.5, 0.5)

        # Query the black-box gets the log(softmax output) over all classes: (1,10)
        # input to the blackbox_model has the shape (N, size, size, nchannel) e.g. (1,32,32,3)
        prob_all_labels = self.blackbox_model.predict(X_adv)
        prob_all_labels = np.atleast_2d(prob_all_labels)
        log_prob_all_labels = np.log(prob_all_labels + 1e-30)
        log_p_target = log_prob_all_labels[:, self.target_label]
        log_p_predicted_label = np.max(log_prob_all_labels, 1)
        predicted_labels = np.argmax(log_prob_all_labels, 1)

        # Select the objective function for BO
        if self.objective_metric == 1:
            score = (log_p_predicted_label[:, None] - log_p_target)
        elif self.objective_metric == 2:
            p_target_labels = prob_all_labels[:, self.target_label]
            sum_p_other_labels = np.sum(prob_all_labels, 1)[:, None] - p_target_labels
            log_sum_p_other_labels = np.log(sum_p_other_labels + 1e-30)
            score = (log_sum_p_other_labels - log_p_target)

        # Check whether the attack is successful
        all_queries_success = predicted_labels == self.target_label
        if all_queries_success.sum() > 0:
            self.success = True
            score = -1
            X_success_adv = X_adv[predicted_labels == self.target_label]
            # Save the successful adversarial example
            print(f'attack succeed! || origin={self.input_label}| target={self.target_label[0]}')
            X_success_saving_path = os.path.join(self.results_folder,
                                                 f'X_{self.dataset_name}_adv_i{self.input_label}_t{self.target_label[0]}_'
                                                 f'eps{self.epsilon}_id{self.orig_img_id}')
            if 'imagenet' in self.dataset_name:
                np.save(X_success_saving_path, np.array([0]))
            else:
                np.save(X_success_saving_path, X_success_adv)
        else:
            self.success = False
            print(f'attack succeed={False}|| origin={self.input_label}| target={self.target_label[0]}|'
                  f'min score at targer={score}| predicted={predicted_labels}')

        return score

    def np_evaluate(self, delta_vector_np):
        """
        :param delta_vector_np: adversarial perturbation in the range of [-1, 1]
        :return score: objective function value
        """

        # Scale the adversarial delta to [-epsilon, + epsilon]
        delta_vector_np = delta_vector_np * self.epsilon
        score = self.evaluate(delta_vector_np)

        return score

    def np_coca_evaluate(self, h_list, x_array):
        """
        :param delta_vector_np: adversarial perturbation in the range of [-1, 1] with dimension (low_dim x nchannel)
        :return score: objective function value
        """

        assert len(h_list) - 1 == len(x_array)
        # h[-1] - which upsampling technique to use: ['BILI','NN','BICU']
        dim_reduction = self.dim_reduction_method_list[h_list[-1]]

        n_row = int(np.sqrt(self.low_dim))
        n_colum = int(np.sqrt(self.low_dim))
        column_range = list(range(n_colum))
        # h1 to 14 - which column to pick for
        perturbation_matrix = np.zeros([n_row, n_colum, self.nchannel])
        h_array = np.array(h_list[:-1]).reshape(self.nchannel, n_row)
        x_array = x_array.reshape(self.nchannel, n_row)

        for i in range(self.nchannel):
            for j in range(n_row):
                row_id = j
                column_id = column_range[h_array[i, j]]
                perturbation_matrix[row_id, column_id, i] = x_array[i, j]

        delta_vector_ld_np = np.atleast_2d(perturbation_matrix.flatten())
        # Scale the adversarial delta to [-epsilon, + epsilon]
        delta_vector_np = delta_vector_ld_np * self.epsilon
        delta_vector_hg_np = upsample_projection(dim_reduction, delta_vector_np, self.low_dim, self.high_dim,
                                                 nchannel=self.nchannel)
        score = self.evaluate(delta_vector_hg_np)
        return score

    def make_gpyopt_space(self, info: dict = None):
        if info is None:
            info = self.get_info()

        category = []
        bounds = []

        for h_name, h_bound in zip(info['h_names'], info['h_bounds']):
            category.append(len(h_bound))
            bounds.append(
                {'name': h_name, 'type': 'categorical',
                 'domain': tuple(h_bound)}
            )

        for x_name, x_bound in zip(info['x_names'], info['x_bounds']):
            bounds.append(
                {'name': x_name, 'type': 'continuous', 'domain': tuple(x_bound)}
            )

        return bounds, category

    def get_info(self):

        reduced_with = int(np.sqrt(self.low_dim))
        n_pixels_to_perturb = int(reduced_with * self.nchannel)
        info = {}
        info['h_bounds'] = [list(range(reduced_with))] * n_pixels_to_perturb + [
            list(range(len(self.dim_reduction_method_list)))]
        info['x_bounds'] = np.vstack([[-1, 1]] * n_pixels_to_perturb)
        info['x_names'] = [f'perturb{i}' for i in range(n_pixels_to_perturb)]
        info['h_names'] = [f'perturb_loc{i}' for i in range(n_pixels_to_perturb)] + ['dim_reduction_method']

        return info


if __name__ == '__main__':

    # Define the model and the original images to be attacked
    reduced_dim_width = 14
    low_dim = int(reduced_dim_width * reduced_dim_width)
    img_offset = 1
    target_label = 0
    results_data_folder = f'./trial/'
    if not os.path.exists(results_data_folder):
        os.mkdir(results_data_folder)
    directory = f'./tf_models/'
    # preprocess
    cnn = CNN(dataset_name='cifar10', img_offset=img_offset, epsilon=0.3,
              low_dim=low_dim, high_dim=int(32 * 32),
              obj_metric=2, results_folder=results_data_folder,
              directory=directory)
    cnn.get_data_sample(target_label)
    bounds, category = cnn.make_gpyopt_space()
    seed = 0
    np.random.seed(seed)
    # range for first
    h_list = list(np.random.choice(range(reduced_dim_width), int(reduced_dim_width * 3))) + [0]
    x_array = np.random.rand(int(reduced_dim_width * 3)) * 2 - 1
    score = cnn.np_coca_evaluate(h_list, x_array)
