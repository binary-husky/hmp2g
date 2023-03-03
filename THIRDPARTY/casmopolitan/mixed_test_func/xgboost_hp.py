from sklearn import model_selection, metrics, datasets
import xgboost
import numpy as np
from test_funcs.base import TestFunction


class XGBoostOptTask(TestFunction):
    problem_type = 'mixed'

    def __init__(self, lamda=1e-6, task=None, split=0.3, normalize=True, seed=None):
        super().__init__(task=task, split=split)

        self.seed = seed
        self.normalize = normalize
        self.lamda = lamda
        self.eval_cnt = 0
        ###########################################################################
        self.original_x_bounds = np.array([
            [0, 1], 
            [1, 10], 
            [0, 10],
            [0.001, 1], 
            [0, 5], 
        ])
        self.continuous_dims = np.array([3, 4, 5, 6, 7])
        self.categorical_dims = np.array([0, 1, 2])
        self.n_vertices = np.array([2, 2, 2])
        # categorical_dims: the dimension indices that are categorical/discrete
        # continuous_dims: the dimension indices that are continuous
        # integer_dims: the **continuous indices** that additionally require integer constraint.
        ###########################################################################

        self.dim = len(self.categorical_dims) + len(self.continuous_dims)
        self.config = self.n_vertices
        self.lb = np.array([-1] * len(self.continuous_dims))
        self.ub = np.array([1] * len(self.continuous_dims))

        # Whether continuous inputs are scaled
        self.split = split
        self.data, self.reg_or_clf = get_data_and_task_type(task)
        self.task = task

        if self.reg_or_clf == 'clf':
            stratify = self.data['target']
        else:
            stratify = None
        self.train_x, self.test_x, self.train_y, self.test_y = \
            model_selection.train_test_split(self.data['data'],
                                             self.data['target'],
                                             test_size=self.split,
                                             stratify=stratify,
                                             random_state=self.seed)
        # self.normalize = False
        if self.normalize:
            self.mean, self.std = self.sample_normalize()
        else:
            self.mean, self.std = None, None

    def evaluate_sinlge_hyperparameters(self, h, x) -> float:
        """
        Evaluate the quality of the hyperparameters

        Parameters
        ----------
        x
            Array of continuous hyperparameters
        h
            Array (or list) of integers corresponding to categorical
            hyperparameters

        Returns
        -------
        score
            1 - Accuracy score on the test set
        """
        # print(f"Evaluating inputs {h, x}")

        # Create model using the chosen hps
        model = self.create_model(h, x)

        # Train model
        model.fit(self.train_x, self.train_y)

        # Test model performance
        y_pred = model.predict(self.test_x)

        # 1-acc for minimization
        if self.reg_or_clf == 'clf':
            score = 1 - metrics.accuracy_score(self.test_y, y_pred)
        elif self.reg_or_clf == 'reg':
            score = metrics.mean_squared_error(self.test_y, y_pred)
        else:
            raise NotImplementedError

        return score

    def compute(self, X, normalize=False):
        if X.ndim == 1: # 如果是一维变量，则伸展一维
            X = X.reshape(1, -1)
        N = X.shape[0]  # N=thread
        res = np.zeros((N,))
        X_cat = X[:, self.categorical_dims] # 读取分类维度
        X_cont = X[:, self.continuous_dims] # 读取连续维度


        self.eval_cnt += 1
        print(f'This is the {self.eval_cnt} evaluation')

        for i, X in enumerate(X):
            h = [int(x_j) for x_j in X_cat[i, :]]

            # x Array of continuous hyperparameters
            # h Array (or list) of integers corresponding to categorical hyperparameters
            res[i] = self.evaluate_sinlge_hyperparameters(h, X_cont[i, :])

            # if self.fX_lb is not None and res[i] < self.fX_lb:
            #     res[i] = self.fX_lb
            # elif self.fX_ub is not None and res[i] > self.fX_ub:
            #     res[i] = self.fX_ub

        res += self.lamda * np.random.rand(*res.shape)

        if normalize:
            res = (res - self.mean) / self.std
        return res

    def sample_normalize(self, size=None):
        from bo.localbo_utils import latin_hypercube, from_unit_cube
        if size is None:
            size = 2 * self.dim + 1
        y = []
        for i in range(size):
            x_cat = np.array([np.random.choice(self.config[_]) for _ in range(self.categorical_dims.shape[0])])
            x_cont = latin_hypercube(1, self.continuous_dims.shape[0])
            x_cont = from_unit_cube(x_cont, self.lb, self.ub).flatten()
            x = np.hstack((x_cat, x_cont))
            y.append(self.compute(x, normalize=False))
        y = np.array(y)
        return np.mean(y), np.std(y)

    def create_model(self, h, x):
        xgboost_kwargs = self.convert_input_into_kwargs(h, x)

        if self.reg_or_clf == 'clf':
            model = xgboost.XGBClassifier(**xgboost_kwargs)
        else:
            model = xgboost.XGBRegressor(**xgboost_kwargs)
        return model

    def convert_input_into_kwargs(self, h, x) -> dict:
        """
        Overwrites default values.

        Info of the different parameters:
        https://xgboost.readthedocs.io/en/latest/parameter.html

        Parameters
        ----------
        x
            continuous params
        h
            categorical params

        Returns
        -------
        dict with xgboost model params

        """
        x = x.flatten()

        new_range = self.original_x_bounds[:, 1] - self.original_x_bounds[:, 0]
        x = ((x - self.lb) * new_range / (self.ub - self.lb)) \
            + self.original_x_bounds[:, 0]

        kwargs = {}

        # Categorical vars
        boosters = ['gbtree', 'dart']  # linear booster ignored
        booster_idx = h[0]
        kwargs['booster'] = boosters[booster_idx]

        grow_policies = ['depthwise', 'lossguide']
        grow_policy_idx = h[1]
        kwargs['grow_policy'] = grow_policies[grow_policy_idx]

        if self.reg_or_clf == 'clf':
            objectives = ['multi:softmax', 'multi:softprob']
        elif self.reg_or_clf == 'reg':
            objectives = ['reg:linear', 'reg:logistic', 'reg:gamma',
                          'reg:tweedie']
        else:
            raise NotImplementedError
        objective_idx = h[2]
        kwargs['objective'] = objectives[objective_idx]

        # Continuous vars
        kwargs['learning_rate'] = x[0]  # [0, 1]
        kwargs['max_depth'] = int(x[1])  # [1, 10]
        kwargs['min_split_loss'] = x[2]  # [0, 10]
        kwargs['subsample'] = x[3]  # [0.001, 1]
        kwargs['reg_lambda'] = x[4]  # [0, 5]

        return kwargs

def get_data_and_task_type(task):
    if task == 'boston':
        data = datasets.load_boston()
        reg_or_clf = 'reg'
    elif task == 'mnist':
        data = datasets.load_digits()
        reg_or_clf = 'clf'

    else:
        raise NotImplementedError("Bad choice for task")

    return data, reg_or_clf


if __name__ == '__main__':
    t = XGBoostOptTask(task='mnist', split=0.5)
    X = np.atleast_2d([0, 0, 1] + [0.5] * 5)
    y = t.compute(X)
