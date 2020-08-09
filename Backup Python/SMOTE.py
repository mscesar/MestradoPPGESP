#ETAPA XX: APLICAÇÃO DO SMOTE 

from imblearn.over_sampling import SMOTE

#Resampling

print('Before OverSampling, the shape of train_X: {}'.format(X_train.shape))
print('Before OverSampling, the shape of train_Y: {} \n'.format(Y_train.shape))

print("Before OverSampling, counts of label '1': {}".format(sum(Y_train == 1)))
print("Before OverSampling, counts of label '0': {} \n".format(sum(Y_train == 0)))

Before OverSampling, the shape of train_X: (59012, 627)
Before OverSampling, the shape of train_Y: (59012,) 

Before OverSampling, counts of label '1': 6803
Before OverSampling, counts of label '0': 52209

import time
start = time.time()

sm = SMOTE(random_state = 42, ratio = 'minority')
X_train_sm, Y_train_sm = sm.fit_sample(X_train, Y_train)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))

Tempo de Execução: 1.3642748276392618 min

print('After OverSampling, the shape of train_X: {}'.format(X_train_sm.shape))
print('After OverSampling, the shape of train_Y: {} \n'.format(Y_train_sm.shape))

print("After OverSampling, counts of label '1': {}".format(sum(Y_train_sm == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_sm == 0)))

After OverSampling, the shape of train_X: (104418, 627)
After OverSampling, the shape of train_Y: (104418,) 

After OverSampling, counts of label '1': 52209
After OverSampling, counts of label '0': 52209

#Parametrização do Modelo

from lightgbm import LGBMClassifier

#classifier_lgbm_sm = LGBMClassifier( max_depth = 1000, 
#                                     learning_rate = 0.1,
#                                     num_leaves = 2000,
#                                     min_data_in_leaf = 200,
#                                     n_estimators = 2000 )

classifier_lgbm_sm = LGBMClassifier( max_depth = 1880, 
                                     learning_rate = 0.1,
                                     num_leaves = 100,
                                     n_estimators = 4500,
                                     min_data_in_leaf = 140 )

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(classifier_lgbm_sm.get_params())

#Fit e Predição
								  
import time
start = time.time()

classifier_lgbm_sm.fit(X_train_sm, Y_train_sm)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 31.92 min

Y_pred_lgbm_sm = classifier_lgbm_sm.predict(X_test)

#Bloco 04: Análise de Métricas

from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm_sm = accuracy_score(Y_test, Y_pred_lgbm_sm)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm_sm))

#Balanced Accuracy

mtrc_balanced_accuracy_score_lgbm_sm = balanced_accuracy_score(Y_test, Y_pred_lgbm_sm)
print('Balanced Accuracy Score : ' + str(mtrc_balanced_accuracy_score_lgbm_sm))

#Precision Score

mtrc_precision_score_lgbm_sm = precision_score(Y_test, Y_pred_lgbm_sm)
print('Precision Score : ' + str(mtrc_precision_score_lgbm_sm))

#Recall Score

mtrc_recall_score_lgbm_sm = recall_score(Y_test, Y_pred_lgbm_sm)
print('Recall Score : ' + str(mtrc_recall_score_lgbm_sm))

#F1 Score

mtrc_f1_score_lgbm_sm = f1_score(Y_test, Y_pred_lgbm_sm)
print('F1 Score : ' + str(mtrc_f1_score_lgbm_sm))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm_sm = matthews_corrcoef(Y_test, Y_pred_lgbm_sm)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm_sm))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm_sm = cohen_kappa_score(Y_test, Y_pred_lgbm_sm)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm_sm))

#Area Under the Receiver Operating Characteristic Curve (ROC AUC)

from sklearn.metrics import roc_auc_score

mtrc_roc_auc_score_lgbm_sm = roc_auc_score(Y_test, Y_pred_lgbm_sm)
print('Area Under the Receiver Operating Characteristic Curve (ROC AUC) : ' + str(mtrc_roc_auc_score_lgbm_sm))

#Area Under the Precision-Recall Curve

from sklearn.metrics import average_precision_score

mtrc_average_precision_score_lgbm_sm = average_precision_score(Y_test, Y_pred_lgbm_sm)
print('Area Under the Precision-Recall Curve : ' + str(mtrc_average_precision_score_lgbm_sm))

#Classification Report

from sklearn.metrics import classification_report

mtrc_classification_report_lgbm_sm = classification_report(Y_test,Y_pred_lgbm_sm)
print('Classification Report : \n' + str(mtrc_classification_report_lgbm_sm)) 

#Confusion Matrix

from sklearn.metrics import confusion_matrix

mtrc_confusion_matrix_lgbm_sm = confusion_matrix(Y_test, Y_pred_lgbm_sm)

print('Confusion Matrix : \n' + str(mtrc_confusion_matrix_lgbm_sm))

print(pd.crosstab(Y_test, Y_pred_lgbm_sm, rownames = ['Real'], colnames = ['Predito'], margins = True))

#Confusion Matrix - Gráfico

import matplotlib.pyplot as plt

import seaborn as sns

LABELS = ['NÃO INTERNOU', 'INTERNOU']
plt.figure(figsize = (7, 7))
sns.heatmap(mtrc_confusion_matrix_lgbm_sm, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt = "d");
plt.title("Confusion Matrix")
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

#Accuracy Score : 0.9541734213751928
#Balanced Accuracy Score : 0.8498824363366055
#Precision Score : 0.8646741386467414
#Recall Score : 0.7143347050754458
#F1 Score : 0.7823474178403755
#Matthews Correlation Coefficient : 0.7613108667384527
#Cohen’s Kappa Score : 0.7569973118488829
#Area Under the Receiver Operating Characteristic Curve (ROC AUC) : 0.8498824363366057
#Area Under the Precision-Recall Curve : 0.650603363585752

Accuracy Score : 0.9557154719070025
Balanced Accuracy Score : 0.8576135289023764
Precision Score : 0.8647441104792851
Recall Score : 0.7301097393689986
F1 Score : 0.7917441428040163
Matthews Correlation Coefficient : 0.770604870162283
Cohen’s Kappa Score : 0.7671649812643972
Area Under the Receiver Operating Characteristic Curve (ROC AUC) : 0.8576135289023764
Area Under the Precision-Recall Curve : 0.6624758860596833