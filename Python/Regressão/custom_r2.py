import numpy as np

from sklearn.metrics import make_scorer

def coefficient_of_determination(y_true, y_pred):
    residuals_sum_of_squares = np.sum((y_true-y_pred)**2)
    total_sum_of_squares = np.sum((y_true-np.mean(y_true))**2)
    return 1-residuals_sum_of_squares/total_sum_of_squares

custom_r2 = make_scorer(coefficient_of_determination, greater_is_better = True)


def coefficient_of_determination_by_correlation(y_true, y_pred):
    observado = (y_true-np.mean(y_true))
    predito = (y_pred-np.mean(y_pred))
    soma_observado_quadrado = np.sum(observado**2)
    soma_predito_quadrado = np.sum(predito**2)
    numerador = (np.sum(observado*predito))**2
    denominador = soma_observado_quadrado*soma_predito_quadrado
    return numerador/denominador
                 
custom_r2_by_correlation = make_scorer(coefficient_of_determination_by_correlation, greater_is_better = True)

def squared_error(ys_orig,ys_line):
    return sum((ys_line - ys_orig) * (ys_line - ys_orig))

def coefficient_of_determination_r2(ys_orig,ys_line):
    y_mean_line = [np.mean(ys_orig) for y in ys_orig]
    squared_error_regr = squared_error(ys_orig, ys_line)
    squared_error_y_mean = squared_error(ys_orig, y_mean_line)
    return 1 - (squared_error_regr/squared_error_y_mean)

y_true = [1, 2, 3]
y_pred = [3, 2, 1]

y_true = pd.DataFrame(y_true)

y_pred = pd.DataFrame(y_pred)

coefficient_of_determination(y_true, y_pred)

coefficient_of_determination_by_correlation(y_true, y_pred)