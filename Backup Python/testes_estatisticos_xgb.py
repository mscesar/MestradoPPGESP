#Comparar 6,9,7

regressor_xgb_6 = XGBRegressor( n_estimators = 500,
                                min_child_weight = 70,
                                max_depth = 45,
                                learning_rate = 0.01 )

#TMP 10 Dias de Corte
#Explained Variance Score: 0.2077
#Mean Absolute Error: 1.7256
#Median Absolute Error: 1.3653
#Mean Squared Error: 5.0220
#Root Mean Squared Error: 2.2410
#R² Score: 0.2074
#Max Error: 8.6376

regressor_xgb_7 = XGBRegressor( n_estimators = 400,
                                min_child_weight = 60,
                                max_depth = 50,
                                learning_rate = 0.01 )

#TMP 10 Dias de Corte
#Explained Variance Score: 0.2095
#Mean Absolute Error: 1.7203
#Median Absolute Error: 1.3627
#Mean Squared Error: 5.0155
#Root Mean Squared Error: 2.2395
#R² Score: 0.2084
#Max Error: 8.7341

#5X2CV COMBINED F TEST
from mlxtend.evaluate import combined_ftest_5x2cv

import time
start = time.time()

f, p = combined_ftest_5x2cv( estimator1 = regressor_xgb_6,
                             estimator2 = regressor_xgb_7,
                             X = X_reg, 
                             y = Y_reg,
                             scoring = 'r2',
                             random_seed = 42 )

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 77.60 min

print('F statistic: %.3f' % f)
print('p value: %.3f' % p)

F statistic: 4.181
p value: 0.064

#5X2CV PAIRED T TEST
from mlxtend.evaluate import paired_ttest_5x2cv

import time
start = time.time()

t, p = paired_ttest_5x2cv( estimator1 = regressor_xgb_6,
                           estimator2 = regressor_xgb_7,
                           X = X_reg, 
                           y = Y_reg,
                           scoring = 'r2',
                           random_seed = 42)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 77.87 min

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

t statistic: -0.210
p value: 0.842

#K-FOLD CROSS-VALIDATED PAIRED T TEST
from mlxtend.evaluate import paired_ttest_kfold_cv

import time
start = time.time()

t, p = paired_ttest_kfold_cv( estimator1 = regressor_xgb_6,
                              estimator2 = regressor_xgb_7,
                              X = X_reg, 
                              y = Y_reg,
                              scoring = 'r2',
                              random_seed = 42 )

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 164.88 min

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

t statistic: 1.886
p value: 0.092

#RESAMPLED PAIRED T TEST
from mlxtend.evaluate import paired_ttest_resampled

import time
start = time.time()

t, p = paired_ttest_resampled( estimator1 = regressor_xgb_6,
                               estimator2 = regressor_xgb_7,
                               X = X_reg, 
                               y = Y_reg,
                               scoring = 'r2',
                               random_seed = 42 )

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
Tempo de Execução: 362.52 min

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

t statistic: 1.388
p value: 0.176