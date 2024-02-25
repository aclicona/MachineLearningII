import numpy as np


class SVD:
    vh = None
    u = None
    s = None
    x = None

    def fit(self, x):
        self.mean = np.mean(x, axis=0)
        self.x = x.copy()
        self._decompose()

    def _decompose_scratch(self):

        x = self.x.copy()

        def get_unitary_matrix(left=True):
            if left:
                B = np.dot(x, x.T)
            else:
                B = np.dot(x.T, x)
            eigenvalues, eigenvectors = np.linalg.eig(B)
            ncols = np.argsort(eigenvalues)[::-1]
            if left:
                return eigenvectors[:, ncols].real
            return eigenvectors[:, ncols].real.T

        def sigma():
            if np.size(np.dot(x, x.T)) <= np.size(np.dot(x.T, x)):
                B = np.dot(x.T, x)
            else:
                B = np.dot(x, x.T)
            eigenvalues, _ = np.linalg.eig(B)
            eigenvalues = np.sqrt(np.abs(eigenvalues))
            return np.sort(eigenvalues)[::-1]

        u = get_unitary_matrix()
        vh = get_unitary_matrix(left=False)
        s = sigma()

        self.u = u

        s_squared = s ** 2
        variance_ratio = s_squared / s_squared.sum()
        # print("Explained variance ratio: %s" % (variance_ratio[: self.n_components]))
        self.vh = vh
        self.s = s

    def _decompose(self):
        # Mean centering
        x = self.x.copy()

        u, s, vh = np.linalg.svd(x, full_matrices=False)
        self.u = u

        s_squared = s ** 2
        variance_ratio = s_squared / s_squared.sum()
        # print("Explained variance ratio: %s" % (variance_ratio[: self.n_components]))
        self.vh = vh
        self.s = s

    def reconstruct_with_eigen_vectors(self, n_components):
        try:
            return self.u[:, :n_components].dot(np.diag(self.s[:n_components]).dot(self.vh[:n_components, :]))
        except:
            return self.u[:, :n_components] @ np.diag(self.s[:n_components]) @ self.vh[:n_components, :]

    def transform(self, n_components):
        return self.x.dot(self.vh[:n_components, :].T)

    def _predict(self, n_components):
        return self.transform(n_components)

    def fit_transform(self, x, n_components):
        self.fit(x)
        return self.transform(n_components)
