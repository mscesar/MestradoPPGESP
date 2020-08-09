#ETAPA XX: APLICAÇÃO DO ANOVA F-VALUE 

#Bloco 01: Aplicação do ANOVA F-value 

from sklearn.feature_selection import SelectKBest

from sklearn.feature_selection import f_classif

fvalue_selector = SelectKBest(f_classif, k = 300)

X_train_fvalue = fvalue_selector.fit_transform(X_train, Y_train)

print('Original number of features:', X_train.shape[1])
print('Reduced number of features:', X_train_fvalue.shape[1])

X_test_fvalue = fvalue_selector.transform(X_test)

#Bloco 02: Parametrização do Modelo

from lightgbm import LGBMClassifier

classifier_lgbm_fvalue = LGBMClassifier( max_depth = 500, 
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
pprint(classifier_lgbm_fvalue.get_params())

#Bloco 03: Fit e Predição

import time
start = time.time()

classifier_lgbm_fvalue.fit(X_train_fvalue, Y_train)

end = time.time()
print("Tempo de Execução: {} sec".format(end - start))

Y_pred_lgbm_fvalue = classifier_lgbm_fvalue.predict(X_test_fvalue)

#Bloco 04: Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm_fvalue = accuracy_score(Y_test, Y_pred_lgbm_fvalue)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm_fvalue))

#Precision Score

mtrc_precision_score_lgbm_fvalue = precision_score(Y_test, Y_pred_lgbm_fvalue)
print('Precision Score : ' + str(mtrc_precision_score_lgbm_fvalue))

#Recall Score

mtrc_recall_score_lgbm_fvalue = recall_score(Y_test, Y_pred_lgbm_fvalue)
print('Recall Score : ' + str(mtrc_recall_score_lgbm_fvalue))

#F1 Score

mtrc_f1_score_lgbm_fvalue = f1_score(Y_test, Y_pred_lgbm_fvalue)
print('F1 Score : ' + str(mtrc_f1_score_lgbm_fvalue))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm_fvalue = matthews_corrcoef(Y_test, Y_pred_lgbm_fvalue)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm_fvalue))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm_fvalue = cohen_kappa_score(Y_test, Y_pred_lgbm_fvalue)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm_fvalue))

#Accuracy Score : 0.9505753034676367
#Precision Score : 0.854468085106383
#Recall Score : 0.6886145404663924
#F1 Score : 0.7626281807823775
#Matthews Correlation Coefficient : 0.7407350595579228
#Cohen’s Kappa Score : 0.7353993195990873