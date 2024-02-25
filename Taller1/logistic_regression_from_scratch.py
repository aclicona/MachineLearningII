from MachineLearningII.Taller1.logistic_regression import LogisticRegressionExample as BaseLogisticReggresion

import pandas as pd
import copy
import numpy as np
from sklearn.metrics import accuracy_score


class LogisticRegressionFromScratch:
    """
    This class creates a logistic regression model from scratch
    """
    def __init__(self):
        self.bias = None
        self.weights = None
        self.losses = []
        self.train_accuracies = []

    def fit(self, x, y, epochs=150):
        x = self._transform_x(x)
        y = self._transform_y(y)

        self.weights = np.zeros(x.shape[1])
        self.bias = 0

        for i in range(epochs):
            x_dot_weights = np.matmul(self.weights, x.transpose()) + self.bias
            pred = self._sigmoid(x_dot_weights)
            loss = self.compute_loss(y, pred)
            error_w, error_b = self.compute_gradients(x, y, pred)
            self.update_model_parameters(error_w, error_b)

            pred_to_class = [1 if p > 0.5 else 0 for p in pred]
            self.train_accuracies.append(accuracy_score(y, pred_to_class))
            self.losses.append(loss)

    @staticmethod
    def compute_loss(y_true, y_pred):
        # binary cross entropy
        y_zero_loss = y_true * np.log(y_pred + 1e-9)
        y_one_loss = (1 - y_true) * np.log(1 - y_pred + 1e-9)
        return -np.mean(y_zero_loss + y_one_loss)

    @staticmethod
    def compute_gradients(x, y_true, y_pred):
        # derivative of binary cross entropy
        difference = y_pred - y_true
        gradient_b = np.mean(difference)
        gradients_w = np.matmul(x.transpose(), difference)
        gradients_w = np.array([np.mean(grad) for grad in gradients_w])

        return gradients_w, gradient_b

    def update_model_parameters(self, error_w, error_b):
        self.weights = self.weights - 0.1 * error_w
        self.bias = self.bias - 0.1 * error_b

    def predict(self, x):
        x_dot_weights = np.matmul(x, self.weights.transpose()) + self.bias
        probabilities = self._sigmoid(x_dot_weights)
        return [1 if p > 0.5 else 0 for p in probabilities]

    def _sigmoid(self, x):
        return np.array([self._sigmoid_function(value) for value in x])

    @staticmethod
    def _sigmoid_function(x):
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            z = np.exp(x)
            return z / (1 + z)

    @staticmethod
    def _transform_x(x):
        try:
            x = pd.DataFrame(x)
        except:
            pass
        x = copy.deepcopy(x)

        return x.values

    @staticmethod
    def _transform_y(y):
        try:
            y = pd.DataFrame(y)
        except:
            pass
        y = copy.deepcopy(y)
        return y.values.reshape(y.shape[0], 1)


class LogisticRegressionExample(BaseLogisticReggresion):
    from_scratch = True

    def train_model(self, x=None, y=None):
        if x is None:
            x = self.x_train.copy()
        if y is None:
            y = self.y_train.copy()

        model = LogisticRegressionFromScratch()
        model.fit(x, y, epochs=150)
        self.model = model
