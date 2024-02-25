# coding:utf-8
import numpy as np


class PCA:
    x = None
    components = None
    mean = None

    def fit(self, x):
        # center the data
        self.mean = np.mean(x, axis=0)
        self.x = x - self.mean

        # compute the covariance matrix
        cov = np.cov(x, rowvar=False)

        # compute the eigenvalues and eigenvectors of the covariance matrix
        eigenvalues, eigenvectors = np.linalg.eigh(cov)

        # sort the eigenvalues and eigenvectors in decreasing order
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]

        # store the first n_components eigenvectors as the principal components
        self.components = eigenvectors

    def transform(self, n_components):
        # center the data
        x = self.x

        # project the data onto the principal components
        x_transformed = np.dot(x, self.components[:, : n_components])

        return x_transformed

    def fit_transform(self, x, n_components):
        self.fit(x)
        return self.transform(n_components)
