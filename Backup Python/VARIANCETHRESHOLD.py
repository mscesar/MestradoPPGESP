#ETAPA XX: APLICAÇÃO DO VARIANCETHRESHOLD

#Bloco 01: Removing Constant Features using Variance Threshold
#Constant features are the type of features that contain only one value for all the outputs in the dataset. 
#Constant features provide no information that can help in classification of the record at hand. 
#Therefore, it is advisable to remove all the constant features from the dataset.

from sklearn.feature_selection import VarianceThreshold

constant_filter = VarianceThreshold(threshold = 0)

constant_filter.fit(X_train)

X_train_vt = constant_filter.transform(X_train)  
X_test_vt = constant_filter.transform(X_test)

print('Original number of features:', X_train.shape[1])
print('Reduced number of features:', X_train_vt.shape[1])

print('Original number of features:', X_test.shape[1])
print('Reduced number of features:', X_test_vt.shape[1])

#Bloco 02: Parametrização do Modelo

from lightgbm import LGBMClassifier

classifier_lgbm_vt = LGBMClassifier( max_depth = 500, 
                                     learning_rate = 0.01,
                                     num_leaves = 1000,
                                     min_data_in_leaf = 200,
                                     n_estimators = 2000,
                                     objective = 'binary',
                                     metric = 'binary_logloss',
                                     random_state = 42 )

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(classifier_lgbm_vt.get_params())

#Bloco 03: Fit e Predição

import time
start = time.time()

classifier_lgbm_vt.fit(X_train_vt, Y_train)

end = time.time()
print("Tempo de Execução: {} sec".format(end - start))

Y_pred_lgbm_vt = classifier_lgbm_vt.predict(X_test_vt)

#Bloco 04: Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm_vt = accuracy_score(Y_test, Y_pred_lgbm_vt)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm_vt))

#Precision Score

mtrc_precision_score_lgbm_vt = precision_score(Y_test, Y_pred_lgbm_vt)
print('Precision Score : ' + str(mtrc_precision_score_lgbm_vt))

#Recall Score

mtrc_recall_score_lgbm_vt = recall_score(Y_test, Y_pred_lgbm_vt)
print('Recall Score : ' + str(mtrc_recall_score_lgbm_vt))

#F1 Score

mtrc_f1_score_lgbm_vt = f1_score(Y_test, Y_pred_lgbm_vt)
print('F1 Score : ' + str(mtrc_f1_score_lgbm_vt))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm_vt = matthews_corrcoef(Y_test, Y_pred_lgbm_vt)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm_vt))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm_vt = cohen_kappa_score(Y_test, Y_pred_lgbm_vt)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm_vt))

#Bloco 05: Removing Quasi-Constant Features Using Variance Threshold
#Não precisou ser aplicado. Nenhuma coluna retornada.

qconstant_filter = VarianceThreshold(threshold = 0.01)

qconstant_filter.fit(X_train_vt)

X_train_vt2 = constant_filter.transform(X_train)  
X_test_vt2 = constant_filter.transform(X_test)   