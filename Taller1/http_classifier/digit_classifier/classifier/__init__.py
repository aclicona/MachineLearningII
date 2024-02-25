import pickle
import os
module_dir = os.path.dirname(__file__)  # get current directory
LOGISTIC_REGRESSION_MODEL = pickle.load(open(f'{module_dir}/ml_models/logistic_regression_for_digits.pkl', 'rb'))

