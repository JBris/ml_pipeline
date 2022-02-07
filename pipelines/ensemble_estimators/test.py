from pycaret.regression import *
from sklearn.linear_model import GammaRegressor
from pycaret.datasets import get_data
boston = get_data('boston')

exp_name = setup(data = boston,  target = 'medv', silent = True)
best_models = compare_models(include=["lr", "dt", "lasso"], n_select = 2)
print(_all_models_internal)
