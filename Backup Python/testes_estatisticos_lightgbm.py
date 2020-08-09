#FIT E PREDIÇÃO DOS CLASSIFICADORES

from lightgbm import LGBMClassifier

#Classificador 04

classifier_lgbm_4 = LGBMClassifier( max_depth = 100, 
                                    learning_rate = 0.3,
                                    num_leaves = 500,
                                    n_estimators = 500 )

import time
start = time.time()
classifier_lgbm_4.fit(X_train, Y_train)
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 1.49 min

Y_pred_lgbm_4 = classifier_lgbm_4.predict(X_test)

#Classificador 07

classifier_lgbm_7 = LGBMClassifier( max_depth = 1000, 
                                    learning_rate = 0.15,
                                    num_leaves = 2000,
                                    min_data_in_leaf = 200,
                                    n_estimators = 2000 )

import time
start = time.time()
classifier_lgbm_7.fit(X_train, Y_train)
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 3.51 min
Tempo de Execução: 10.47 min

Y_pred_lgbm_7 = classifier_lgbm_7.predict(X_test)

#Classificador 08

classifier_lgbm_8 = LGBMClassifier( max_depth = 1000, 
                                    learning_rate = 0.1,
                                    num_leaves = 2000,
                                    min_data_in_leaf = 200,
                                    n_estimators = 2000 )

import time
start = time.time()
classifier_lgbm_8.fit(X_train, Y_train)
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 3.02 min
Tempo de Execução: 4.89 min
Tempo de Execução: 14.47 min

Y_pred_lgbm_8 = classifier_lgbm_8.predict(X_test)

#Classificador 28

classifier_lgbm_28 = LGBMClassifier( max_depth = 1880, 
                                     learning_rate = 0.1,
                                     num_leaves = 100,
                                     n_estimators = 4500,
                                     min_data_in_leaf = 140 )

import time
start = time.time()
classifier_lgbm_28.fit(X_train, Y_train)
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 4.06 min
Tempo de Execução: 19.90 min

Y_pred_lgbm_28 = classifier_lgbm_28.predict(X_test)

#Classificador 12

classifier_lgbm_12 = LGBMClassifier( max_depth = 1000, 
                                     learning_rate = 0.01,
                                     num_leaves = 2000,
                                     min_data_in_leaf = 200,
                                     n_estimators = 20000,
                                     colsample_bytree = 0.8 )

import time
start = time.time()
classifier_lgbm_12.fit(X_train, Y_train)
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 110.09 min

Y_pred_lgbm_12 = classifier_lgbm_28.predict(X_test)

#Classificador 18

classifier_lgbm_18 = LGBMClassifier( max_depth = 1000, 
                                     learning_rate = 0.01,
                                     num_leaves = 2000,
                                     min_data_in_leaf = 200,
                                     n_estimators = 50000 )

import time
start = time.time()
classifier_lgbm_18.fit(X_train, Y_train)
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 167.31 min

classifier_lgbm_18 = classifier_lgbm_18.predict(X_test)

#F-TEST (4,7,8)

from mlxtend.evaluate import ftest

f, p_value = ftest( Y_test, 
                    Y_pred_lgbm_4, 
                    Y_pred_lgbm_7, 
                    Y_pred_lgbm_8 )

print('F: %.3f' % f)
print('p-value: %.3f' % p_value)

F: 0.044
p-value: 0.957

#F-TEST (8,28)

from mlxtend.evaluate import ftest

f, p_value = ftest( Y_test, 
                    Y_pred_lgbm_28, 
                    Y_pred_lgbm_8 )

print('F: %.3f' % f)
print('p-value: %.3f' % p_value)

F: 1.770
p-value: 0.183

#5X2CV COMBINED F TEST (7,8)

from mlxtend.evaluate import combined_ftest_5x2cv

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time
start = time.time()

f, p = combined_ftest_5x2cv( estimator1 = classifier_lgbm_7,
                             estimator2 = classifier_lgbm_8,
                             X = X, 
                             y = Y,
                             scoring = make_scorer(matthews_corrcoef),
                             random_seed = 42 )

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 42.50 min

print('F statistic: %.3f' % f)
print('p value: %.3f' % p)

F statistic: 0.656
p value: 0.734

#5X2CV COMBINED F TEST (8,28)

from mlxtend.evaluate import combined_ftest_5x2cv

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time
start = time.time()

f, p = combined_ftest_5x2cv( estimator1 = classifier_lgbm_8,
                             estimator2 = classifier_lgbm_28,
                             X = X, 
                             y = Y,
                             scoring = make_scorer(matthews_corrcoef),
                             random_seed = 42 )

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 191.36 min

print('F statistic: %.3f' % f)
print('p value: %.3f' % p)

F statistic: 10.407
p value: 0.009

#5X2CV PAIRED T TEST (7,8)
from mlxtend.evaluate import paired_ttest_5x2cv

import time
start = time.time()

t, p = paired_ttest_5x2cv( estimator1 = classifier_lgbm_7,
                           estimator2 = classifier_lgbm_8,
                           X = X, 
                           y = Y,
                           scoring = make_scorer(matthews_corrcoef),
                           random_seed = 42)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 30.59 min

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

t statistic: 0.341
p value: 0.747

#5X2CV PAIRED T TEST (8,28)
from mlxtend.evaluate import paired_ttest_5x2cv

import time
start = time.time()

t, p = paired_ttest_5x2cv( estimator1 = classifier_lgbm_8,
                           estimator2 = classifier_lgbm_28,
                           X = X, 
                           y = Y,
                           scoring = make_scorer(matthews_corrcoef),
                           random_seed = 42)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 137.69 min

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

t statistic: -3.605
p value: 0.015

#K-FOLD CROSS-VALIDATED PAIRED T TEST (7,8)
from mlxtend.evaluate import paired_ttest_kfold_cv

import time
start = time.time()

t, p = paired_ttest_kfold_cv( estimator1 = classifier_lgbm_7,
                              estimator2 = classifier_lgbm_8,
                              X = X, 
                              y = Y,
                              scoring = make_scorer(matthews_corrcoef),
                              random_seed = 42 )

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 246.89 min

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

t statistic: 0.238
p value: 0.817

#K-FOLD CROSS-VALIDATED PAIRED T TEST (8,28)
from mlxtend.evaluate import paired_ttest_kfold_cv

import time
start = time.time()

t, p = paired_ttest_kfold_cv( estimator1 = classifier_lgbm_8,
                              estimator2 = classifier_lgbm_28,
                              X = X, 
                              y = Y,
                              scoring = make_scorer(matthews_corrcoef),
                              random_seed = 42 )

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 369.45 min

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

t statistic: -4.961
p value: 0.001

#RESAMPLED PAIRED T TEST (7,8)
from mlxtend.evaluate import paired_ttest_resampled

import time
start = time.time()

t, p = paired_ttest_resampled( estimator1 = classifier_lgbm_7,
                               estimator2 = classifier_lgbm_8,
                               X = X, 
                               y = Y,
                               scoring = make_scorer(matthews_corrcoef),
                               random_seed = 42 )

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 611.85 min

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

t statistic: -1.511
p value: 0.142

#RESAMPLED PAIRED T TEST (8, 28)
from mlxtend.evaluate import paired_ttest_resampled

import time
start = time.time()

t, p = paired_ttest_resampled( estimator1 = classifier_lgbm_8,
                               estimator2 = classifier_lgbm_28,
                               X = X, 
                               y = Y,
                               scoring = make_scorer(matthews_corrcoef),
                               random_seed = 42 )

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 604.17 min

print('t statistic: %.3f' % t)
print('p value: %.3f' % p)

t statistic: -10.416
p value: 0.000