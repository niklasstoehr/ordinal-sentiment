import numpy as np
from scipy import stats


class Majority_Vote():

    def __init__(self, task_type = "", data_type = "", seq = 1, n_features=1):
        print(f"majority vote on {task_type} with {data_type} seq {seq} n_features {n_features}")


    def train(self, x_train, y_train):
        self.y_pred = stats.mode(y_train)[0]
        print(self.y_pred, np.unique(y_train, return_counts=True))

    def predict(self, x_test):
        y_pred = self.y_pred.repeat(x_test.shape[0])
        return y_pred
