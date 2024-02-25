import numpy as np


# Define the number of rows and columns
class MatrixExercises:
    def __init__(self):
        self.matrix: np.ndarray = None

    def create_matrix(self, num_columns: int, num_rows: int, min_value: int = 0, max_value: int = 100):
        """
        Return a matrix of m x n
        :param num_columns: Number of columns (m)
        :param num_rows: Number of rows (n)
        :param min_value: The min value of the matrix. Default 0
        :param max_value: The max value of the matrix. Default 100
        :return: A numpy array
        """

        # Generate a random matrix with integers between min_value and max_value
        matrix: np.ndarray = np.random.randint(min_value, max_value + 1, size=(num_columns, num_rows))
        self.matrix = matrix

    def rank_of_matrix(self, matrix: np.ndarray = None):
        """
        Calculate the rank of the matrix (the maximum number of rows (or columns) linearly independent)
        :return: The rank of the matrix
        """

        matrix: np.ndarray = matrix if matrix is not None else self.matrix
        if matrix is not None:
            return np.linalg.matrix_rank(matrix)
        else:
            raise ValueError('ERROR: You must pass a matrix to class or the func')

    def trace_of_matrix(self, matrix: np.ndarray = None):
        """
        Calculate the trace of a matrix
        :param matrix:
        :return:
        """
        matrix: np.ndarray = matrix if matrix is not None else self.matrix
        if matrix is not None:
            return matrix.trace()
        else:
            raise ValueError('ERROR: You must pass a matrix to class or the func')

    def determinant_of_matrix(self, matrix: np.ndarray = None):
        """
        Calculate the determinant of a square matrix
        :param matrix:
        :return:
        """
        matrix: np.ndarray = matrix if matrix is not None else self.matrix
        if matrix is not None:
            if not np.all(np.shape(matrix) == (matrix.shape[0], matrix.shape[0])):
                raise ValueError("Matrix must be square.")
            return np.linalg.det(matrix)
        else:
            print('ERROR: You must pass a matrix to class or the func')

    def inverted_matrix(self, matrix: np.ndarray = None):
        """
          Inverts a square matrix using numpy.linalg.inv.

          Args:
            matrix: A 2D numpy array representing the square matrix.

          Returns:
            The inverse of the matrix, or None if it is not invertible.
          """
        matrix: np.ndarray = matrix if matrix is not None else self.matrix
        if matrix is not None:
            if not np.all(np.shape(matrix) == (matrix.shape[0], matrix.shape[0])):
                raise ValueError("Matrix must be square.")
            try:
                return np.linalg.inv(matrix)
            except np.linalg.LinAlgError:
                print("Matrix is not invertible.")
                return None
        else:
            raise ValueError('ERROR: You must pass a matrix to class or the func')

    def eigenvalues_eigen_vectors(self, matrix: np.ndarray = None):

        """
        Calculate
        :param matrix:
        :return:
        """
        matrix: np.ndarray = matrix if matrix is not None else self.matrix
        if matrix is not None:
            if not np.all(np.shape(matrix) == (matrix.shape[0], matrix.shape[0])):
                raise ValueError("Matrix must be square.")
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            return eigenvalues, eigenvectors
        else:
            raise ValueError('ERROR: You must pass a matrix to class or the func')

    def get_squared_matrix(self, matrix: np.array = None, n_dimensional_matrix=False, replace_matrix=True):
        """

        :param matrix:
        :param n_dimensional_matrix:
        :param replace_matrix:
        :return:
        """
        matrix: np.ndarray = matrix if matrix is not None else self.matrix
        if matrix is not None:
            if n_dimensional_matrix:
                resultado = np.dot(matrix.transpose(), matrix)
            else:
                resultado = np.dot(matrix, matrix.transpose())
            if replace_matrix:
                self.matrix = resultado
            return resultado
        else:
            print('ERROR: You must pass a matrix to class or the func')
