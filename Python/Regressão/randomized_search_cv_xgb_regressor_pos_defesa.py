#TESTE 05: Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor

#Criação das Variáveis dos Parâmetros a Serem Testados

#max_depth [default=6]: The maximum depth of a tree, same as GBM.
#Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
#Should be tuned using CV. Typical values: 3-10.
max_depth_xgb = [int(x) for x in np.linspace(start = 50, stop = 1000, num = 20)]

#learning_rate:
learning_rate_xgb = [0.001, 0.01, 0.03, 0.05, 0.1, 0.15]

#min_child_weight [default=1]: Defines the minimum sum of weights of all observations required in a child.
#This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
#Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
#Too high values can lead to under-fitting hence, it should be tuned using CV.
min_child_weight_xgb = [int(x) for x in np.linspace(10, 100, num = 19)]

#n_estimators:
n_estimators_xgb = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 19)]

# Criação e Fit do Random Grid

random_grid_xgb = { 'max_depth': max_depth_xgb,
                    'learning_rate': learning_rate_xgb,
                    'min_child_weight': min_child_weight_xgb,
                    'n_estimators': n_estimators_xgb }

regressor_xgb_rscv = XGBRegressor()

regressor_xgb_rscv_random = RandomizedSearchCV( estimator = regressor_xgb_rscv, 
                                                param_distributions = random_grid_xgb, 
                                                n_iter = 50, 
                                                cv = 3, 
                                                verbose = 2, 
                                                random_state = 42,
                                                scoring = 'neg_mean_squared_error' )

import time

start = time.time()
regressor_xgb_rscv_random.fit(X_train_reg, Y_train_reg)
end = time.time()

print("Tempo de Execução: {} min".format((end - start)/60))
#Tempo de Execução: 512.3018831451734 min

regressor_xgb_rscv_random.best_params_

#TMP 10 Dias de Corte
{'learning_rate': 0.01,
 'max_depth': 500,
 'min_child_weight': 50,
 'n_estimators': 400}

regressor_xgb_rscv_random.best_score_

#TMP 10 Dias de Corte
-5.051954873646905

regressor_xgb_rscv_random.best_estimator_

regressor_xgb_rscv_random.best_index_

#TMP 10 Dias de Corte
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=500, min_child_weight=50, missing=None, n_estimators=400,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)

regressor_xgb = regressor_xgb_rscv_random.best_estimator_

#TMP 10 Dias de Corte
#Parâmetros Utilizados pelo Modelo
from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_xgb.get_params())

Parameters Currently In Use:

{'base_score': 0.5,
 'booster': 'gbtree',
 'colsample_bylevel': 1,
 'colsample_bynode': 1,
 'colsample_bytree': 1,
 'gamma': 0,
 'importance_type': 'gain',
 'learning_rate': 0.01,
 'max_delta_step': 0,
 'max_depth': 500,
 'min_child_weight': 50,
 'missing': None,
 'n_estimators': 400,
 'n_jobs': 1,
 'nthread': None,
 'objective': 'reg:linear',
 'random_state': 0,
 'reg_alpha': 0,
 'reg_lambda': 1,
 'scale_pos_weight': 1,
 'seed': None,
 'silent': None,
 'subsample': 1,
 'verbosity': 1}

# Fit

import time
start = time.time()

regressor_xgb.fit(X_train_reg, Y_train_reg)
#regressor_xgb.fit(X_reg, Y_reg)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 3.28 min

#Predição

Y_pred_train_reg_xgb = regressor_xgb.predict(X_train_reg)

Y_pred_test_reg_xgb = regressor_xgb.predict(X_test_reg)

#Análise de Métricas

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_xgb = explained_variance_score(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_mae_xgb = mean_absolute_error(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_mdae_xgb = median_absolute_error(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_mse_xgb = mean_squared_error(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_rmse_xgb = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_xgb))
mtrc_r2_xgb = r2_score(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_max_error_xgb = max_error(Y_test_reg, Y_pred_test_reg_xgb)

print('Explained Variance Score: {:.16f}'.format(mtrc_evs_xgb))
print('Mean Absolute Error: {:.16f}'.format(mtrc_mae_xgb)) 
print('Median Absolute Error: {:.16f}'.format(mtrc_mdae_xgb)) 
print('Mean Squared Error: {:.16f}'.format(mtrc_mse_xgb))
print('Root Mean Squared Error: {:.16f}'.format(mtrc_rmse_xgb))
print('R² Score: {:.16f}'.format(mtrc_r2_xgb))
print('Max Error: {:.16f}'.format(mtrc_max_error_xgb))

Explained Variance Score: 0.2027154312519833
Mean Absolute Error: 1.7256828886987661
Median Absolute Error: 1.3613290786743164
Mean Squared Error: 5.0578681654644351
Root Mean Squared Error: 2.2489704678951288
R² Score: 0.2017536700689200
Max Error: 8.6951642036437988


#TESTE 06: Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor

#Criação das Variáveis dos Parâmetros a Serem Testados

#max_depth [default=6]: The maximum depth of a tree, same as GBM.
#Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
#Should be tuned using CV. Typical values: 3-10.
max_depth_xgb = [int(x) for x in np.linspace(start = 50, stop = 500, num = 19)]

#learning_rate:
learning_rate_xgb = [0.001, 0.01, 0.03, 0.05, 0.1, 0.15]

#min_child_weight [default=1]: Defines the minimum sum of weights of all observations required in a child.
#This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
#Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
#Too high values can lead to under-fitting hence, it should be tuned using CV.
min_child_weight_xgb = [int(x) for x in np.linspace(10, 100, num = 19)]

#n_estimators:
n_estimators_xgb = [int(x) for x in np.linspace(start = 100, stop = 1000, num = 37)]

# Criação e Fit do Random Grid

random_grid_xgb = { 'max_depth': max_depth_xgb,
                    'learning_rate': learning_rate_xgb,
                    'min_child_weight': min_child_weight_xgb,
                    'n_estimators': n_estimators_xgb }

regressor_xgb_rscv = XGBRegressor()

regressor_xgb_rscv_random = RandomizedSearchCV( estimator = regressor_xgb_rscv, 
                                                param_distributions = random_grid_xgb, 
                                                n_iter = 100, 
                                                cv = 3, 
                                                verbose = 2, 
                                                random_state = 42,
                                                n_jobs = 2,
                                                scoring = 'neg_mean_squared_error' )

import time

start = time.time()
regressor_xgb_rscv_random.fit(X_train_reg, Y_train_reg)
end = time.time()

print("Tempo de Execução: {} min".format((end - start)/60))
#Tempo de Execução: 905.8435114304225 min

regressor_xgb_rscv_random.best_params_

#TMP 10 Dias de Corte
{'n_estimators': 550,
 'min_child_weight': 30,
 'max_depth': 375,
 'learning_rate': 0.01}

regressor_xgb_rscv_random.best_score_

#TMP 10 Dias de Corte
-5.045786263507449

regressor_xgb_rscv_random.best_estimator_

regressor_xgb_rscv_random.best_index_

#TMP 10 Dias de Corte
XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
             colsample_bynode=1, colsample_bytree=1, gamma=0,
             importance_type='gain', learning_rate=0.01, max_delta_step=0,
             max_depth=375, min_child_weight=30, missing=None, n_estimators=550,
             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
             silent=None, subsample=1, verbosity=1)

regressor_xgb = regressor_xgb_rscv_random.best_estimator_

#TMP 10 Dias de Corte
#Parâmetros Utilizados pelo Modelo
from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_xgb.get_params())

# Fit
import time
start = time.time()
regressor_xgb.fit(X_train_reg, Y_train_reg)
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 13.91 min

#Predição

Y_pred_train_reg_xgb = regressor_xgb.predict(X_train_reg)
Y_pred_test_reg_xgb = regressor_xgb.predict(X_test_reg)

#Análise de Métricas

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_xgb = explained_variance_score(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_mae_xgb = mean_absolute_error(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_mdae_xgb = median_absolute_error(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_mse_xgb = mean_squared_error(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_rmse_xgb = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_xgb))
mtrc_r2_xgb = r2_score(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_max_error_xgb = max_error(Y_test_reg, Y_pred_test_reg_xgb)

print('Explained Variance Score: {:.16f}'.format(mtrc_evs_xgb))
print('Mean Absolute Error: {:.16f}'.format(mtrc_mae_xgb)) 
print('Median Absolute Error: {:.16f}'.format(mtrc_mdae_xgb)) 
print('Mean Squared Error: {:.16f}'.format(mtrc_mse_xgb))
print('Root Mean Squared Error: {:.16f}'.format(mtrc_rmse_xgb))
print('R² Score: {:.16f}'.format(mtrc_r2_xgb))
print('Max Error: {:.16f}'.format(mtrc_max_error_xgb))

Explained Variance Score: 0.2026630237613579
Mean Absolute Error: 1.7270944941561133
Median Absolute Error: 1.3638865947723389
Mean Squared Error: 5.0534494591824659
Root Mean Squared Error: 2.2479878690025146
R² Score: 0.2024510421547940
Max Error: 8.7001738548278809























