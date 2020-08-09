#ETAPA XX: APLICAÇÃO DO CHI-SQUARED
#Não foi possível realizar a aplicação devido à existência de valores negativos.

#Bloco 01: Aplicação do Chi-Squared 

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import chi2

chi2_selector = SelectKBest(chi2, k = 300)

X_train_chi2 = chi2_selector.fit_transform(X_train, Y_train)

print('Original number of features:', X_train.shape[1])
print('Reduced number of features:', X_train_chi2.shape[1])

X_test_chi2 = chi2_selector.transform(X_test)

#Bloco 02: Parametrização do Modelo

from lightgbm import LGBMClassifier

classifier_lgbm_chi2 = LGBMClassifier( max_depth = 500, 
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
pprint(classifier_lgbm_chi2.get_params())

#Bloco 03: Fit e Predição

import time
start = time.time()

classifier_lgbm_chi2.fit(X_train_chi2, Y_train)

end = time.time()
print("Tempo de Execução: {} sec".format(end - start))

Y_pred_lgbm_chi2 = classifier_lgbm_chi2.predict(X_test_chi2)

#Bloco 04: Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm_chi2 = accuracy_score(Y_test, Y_pred_lgbm_chi2)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm_chi2))

#Precision Score

mtrc_precision_score_lgbm_chi2 = precision_score(Y_test, Y_pred_lgbm_chi2)
print('Precision Score : ' + str(mtrc_precision_score_lgbm_chi2))

#Recall Score

mtrc_recall_score_lgbm_chi2 = recall_score(Y_test, Y_pred_lgbm_chi2)
print('Recall Score : ' + str(mtrc_recall_score_lgbm_chi2))

#F1 Score

mtrc_f1_score_lgbm_chi2 = f1_score(Y_test, Y_pred_lgbm_chi2)
print('F1 Score : ' + str(mtrc_f1_score_lgbm_chi2))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm_chi2 = matthews_corrcoef(Y_test, Y_pred_lgbm_chi2)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm_chi2))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm_chi2 = cohen_kappa_score(Y_test, Y_pred_lgbm_chi2)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm_chi2))