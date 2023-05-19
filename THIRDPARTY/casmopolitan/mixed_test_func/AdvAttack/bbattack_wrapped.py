# @author: Xingchen Wan
# Wrapper around the bbattack_objective.py to provide a compatible interface into our codebase

from test_funcs.base import TestFunction
from mixed_test_func.AdvAttack.bbattack_objective import CNN
import numpy as np

class AdversarialAttack(TestFunction):

    problem_type = 'mixed'

    def __init__(self, data_dir, dataset='cifar10',
                 target_label=0,
                 img_offset=1,
                 epsilon=0.3,
                 low_dim=int(14*14),
                 high_dim=int(32*32),
                 obj_metric=2,
                 save_dir=None,
                 ):
        if dataset != 'cifar10':
            raise NotImplementedError('dataset ' + str(dataset) + ' is not implemented!')
        super(AdversarialAttack, self).__init__(normalize=False)
        # Initialise a CNN model
        self.cnn = CNN(
            dataset_name=dataset,
            img_offset=img_offset,
            epsilon=epsilon,
            low_dim=low_dim, high_dim=high_dim,
            obj_metric=obj_metric, results_folder=save_dir,
            directory=data_dir
        )
        self.cnn.get_data_sample(target_label)
        self.input_label = self.cnn.input_label
        self.target_label = self.cnn.target_label[0]
        # May be subjected to changes?
        # 43 categorical variables
        self.categorical_dims = np.arange(43)
        self.continuous_dims = np.arange(43, 43+42)
        self.n_vertices = np.array([14] * 42 + [3])
        self.config = self.n_vertices
        # 42 continuous variables
        self.ub = np.ones(42)
        self.lb = -1. * np.ones(42)

    def compute(self, x, normalize=None):
        if x.ndim == 1:
            x = x.reshape(1, -1)
        x[:, self.categorical_dims] = np.round(x[:, self.categorical_dims])
        res = []
        for i, x_ in enumerate(x):
            cat = list(x_[self.categorical_dims].astype(int))
            cont = x_[self.continuous_dims]
            res.append(self.cnn.np_coca_evaluate(cat, cont))
        return np.array(res).reshape(-1, 1)


if __name__ == '__main__':
    f = AdversarialAttack(f'./tf_models/', save_dir=f'./trial/')
    x_array = np.random.rand(int(14*3)) * 2 -1
    h_list = np.array(list(np.random.choice(range(14), int(14*3))) + [0])
    z = np.hstack((h_list, x_array))
    print(f.compute(z))
