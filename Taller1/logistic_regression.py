import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import confusion_matrix
from MachineLearningII.unsupervised.dim_red.pca import PCA
from MachineLearningII.unsupervised.dim_red.svd import SVD
from MachineLearningII.unsupervised.dim_red.tsne import TSNE
from sklearn.datasets import load_digits
from sklearn.model_selection import train_test_split
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA as SCI_PCA
from sklearn.manifold import TSNE as SCI_TSNE
import pickle
import matplotlib.gridspec as gridspec


def get_digits_from_scikit_learn():
    digits = load_digits()
    df = pd.DataFrame(np.column_stack([digits['data'], digits['target']]), columns=digits['feature_names'] + ['target'])
    df = df[(df['target'] == 0.0) | (df['target'] == 8.0)]
    df_num = df.drop(columns=['target']).to_numpy()
    df_target = df['target'].to_numpy()
    return train_test_split(df_num, df_target, test_size=0.20, random_state=42)


def get_digits_from_csv():
    train: pd.DataFrame = pd.read_csv('mnist/mnist_train_ceros_eights.csv', header=0)
    y_train = train['label'].copy().to_numpy()
    x_train = train.drop(columns=['label']).to_numpy()

    test: pd.DataFrame = pd.read_csv('mnist/mnist_test_ceros_eights.csv')
    y_test = test['label'].copy().to_numpy()
    x_test = test.drop(columns=['label']).to_numpy()
    return x_train, x_test, y_train, y_test


class LogisticRegressionExample:
    model = None
    from_scratch = False

    def __init__(self, dataset='scikit_learn', solver='pca', num_components=2):
        if dataset == 'scikit_learn':
            self.x_train, self.x_test, self.y_train, self.y_test = get_digits_from_scikit_learn()
        elif dataset == 'from_csv':
            self.x_train, self.x_test, self.y_train, self.y_test = get_digits_from_csv()
        self.solver = solver
        self.num_components = num_components

    def train_model(self, x=None, y=None):
        if x is None:
            x = self.x_train.copy()
        if y is None:
            y = self.y_train.copy()

        model = LogisticRegression(solver='liblinear', random_state=0)
        model.fit(x, y)
        self.model = model

    def predict(self, model=None, x=None, y=None):
        if x is None:
            x = self.x_test.copy()
        if y is None:
            y = self.y_test.copy()
        if model is None:
            model = self.model
        y_predicted = model.predict(x)
        cm = confusion_matrix(y, y_predicted)

        return y_predicted, cm

    @staticmethod
    def plot_confusion_matrix(cm, show=True):
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.imshow(cm)
        ax.grid(False)
        ax.xaxis.set(ticks=(0, 1), ticklabels=('Predicted 0s', 'Predicted 1s'))
        ax.yaxis.set(ticks=(0, 1), ticklabels=('Actual 0s', 'Actual 1s'))
        ax.set_ylim(1.5, -0.5)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, cm[i, j], ha='center', va='center', color='red')
        plt.show()

    def extract_features(self, x):
        if self.from_scratch:
            if self.solver == 'pca':
                reductor_of_dim = PCA()
            elif self.solver == 'svd':
                reductor_of_dim = SVD()
            elif self.solver == 'tsne':
                reductor_of_dim = TSNE()
                transform, _ = reductor_of_dim.fit_transform(x, perplexity=10, T=1000, η=200, early_exaggeration=4,
                                                             n_dimensions=self.num_components)
                return transform
            else:
                reductor_of_dim = None
            if reductor_of_dim is not None:
                return reductor_of_dim.fit_transform(x, self.num_components)
        else:
            if self.solver == 'pca':
                reductor_of_dim = SCI_PCA(n_components=self.num_components)
            elif self.solver == 'svd':
                reductor_of_dim = TruncatedSVD(n_components=self.num_components)
            elif self.solver == 'tsne':
                reductor_of_dim = SCI_TSNE(n_components=self.num_components, learning_rate='auto', init='random',
                                           perplexity=3)
            else:
                reductor_of_dim = None
            if reductor_of_dim is not None:
                return reductor_of_dim.fit_transform(x)

    @staticmethod
    def graph_features_applying_dim_red(x=None, y=None, mask=8, show=True):
        plots = []
        legend = []
        for num, color in zip([0, mask], ['b', 'r']):
            where = np.where(y == num)[0]
            plot = plt.scatter(x[where][:, 0], x[:, 1][where], c=color)
            plots.append(plot)
            legend.append('Num {num}'.format(num=8 if num == mask else num))
        plt.legend(plots, legend)
        plt.show()

    def graph_features_and_predict_using_dimensional_reduction(self, x=None, y=None, x_test=None, y_test=None, mask=8):
        if x is None:
            x = self.x_train.copy()
        if y is None:
            y = self.y_train.copy()
        if x_test is None:
            x_test = self.x_test.copy()
        if y_test is None:
            y_test = self.y_test.copy()

        x_transformed = self.extract_features(x)
        # graph the features
        self.graph_features_applying_dim_red(x_transformed, y, mask=mask)
        self.train_model(x=x_transformed, y=y)
        # transform the x_test using dimensional reduction
        x_test_transformed = self.extract_features(x_test)
        # run again the prediction and plot the confusion matrix
        _, conf_matrix = self.predict(x=x_test_transformed, y=y_test)
        self.plot_confusion_matrix(conf_matrix)

    def save_model(self, model_name, path='http_classifier/digit_classifier/classifier/ml_models'):
        pickle.dump(self.model, open(f'{path}/{model_name}.pkl', 'wb'))
