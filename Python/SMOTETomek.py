#ETAPA XX: APLICAÇÃO DO SMOTETomek

from imblearn.combine import SMOTETomek

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

smtk = SMOTETomek(random_state = 42, ratio = 'minority')
X_train_smtk, Y_train_smtk = smtk.fit_sample(X_train, Y_train)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))

Tempo de Execução: 202.13765750726063 min

print('After OverSampling, the shape of train_X: {}'.format(X_train_smtk.shape))
print('After OverSampling, the shape of train_Y: {} \n'.format(Y_train_smtk.shape))

print("After OverSampling, counts of label '1': {}".format(sum(Y_train_smtk == 1)))
print("After OverSampling, counts of label '0': {}".format(sum(Y_train_smtk == 0)))

After OverSampling, the shape of train_X: (104402, 627)
After OverSampling, the shape of train_Y: (104402,) 

After OverSampling, counts of label '1': 52201
After OverSampling, counts of label '0': 52201

#Parametrização do Modelo

from lightgbm import LGBMClassifier

#classifier_lgbm_smtk = LGBMClassifier( max_depth = 1000, 
#                                       learning_rate = 0.1,
#                                       num_leaves = 2000,
#                                       min_data_in_leaf = 200,
#                                       n_estimators = 2000 )

classifier_lgbm_smtk = LGBMClassifier( max_depth = 1880, 
                                       learning_rate = 0.1,
                                       num_leaves = 100,
                                       n_estimators = 4500,
                                       min_data_in_leaf = 140 )

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(classifier_lgbm_smtk.get_params())

#Fit e Predição
					
import time
start = time.time()
			  
classifier_lgbm_smtk.fit(X_train_smtk, Y_train_smtk)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Tempo de Execução: 34.12 min

Y_pred_lgbm_smtk = classifier_lgbm_smtk.predict(X_test)

#Bloco 04: Análise de Métricas

from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm_smtk = accuracy_score(Y_test, Y_pred_lgbm_smtk)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm_smtk))

#Balanced Accuracy

mtrc_balanced_accuracy_score_lgbm_smtk = balanced_accuracy_score(Y_test, Y_pred_lgbm_smtk)
print('Balanced Accuracy Score : ' + str(mtrc_balanced_accuracy_score_lgbm_smtk))

#Precision Score

mtrc_precision_score_lgbm_smtk = precision_score(Y_test, Y_pred_lgbm_smtk)
print('Precision Score : ' + str(mtrc_precision_score_lgbm_smtk))

#Recall Score

mtrc_recall_score_lgbm_smtk = recall_score(Y_test, Y_pred_lgbm_smtk)
print('Recall Score : ' + str(mtrc_recall_score_lgbm_smtk))

#F1 Score

mtrc_f1_score_lgbm_smtk = f1_score(Y_test, Y_pred_lgbm_smtk)
print('F1 Score : ' + str(mtrc_f1_score_lgbm_smtk))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm_smtk = matthews_corrcoef(Y_test, Y_pred_lgbm_smtk)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm_smtk))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm_smtk = cohen_kappa_score(Y_test, Y_pred_lgbm_smtk)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm_smtk))

#Area Under the Receiver Operating Characteristic Curve (ROC AUC)

from sklearn.metrics import roc_auc_score

mtrc_roc_auc_score_lgbm_smtk = roc_auc_score(Y_test, Y_pred_lgbm_smtk)
print('Area Under the Receiver Operating Characteristic Curve (ROC AUC) : ' + str(mtrc_roc_auc_score_lgbm_smtk))

#Area Under the Precision-Recall Curve

from sklearn.metrics import average_precision_score

mtrc_average_precision_score_lgbm_smtk = average_precision_score(Y_test, Y_pred_lgbm_smtk)
print('Area Under the Precision-Recall Curve : ' + str(mtrc_average_precision_score_lgbm_smtk))

#Classification Report

from sklearn.metrics import classification_report

mtrc_classification_report_lgbm_smtk = classification_report(Y_test,Y_pred_lgbm_smtk)
print('Classification Report : \n' + str(mtrc_classification_report_lgbm_smtk)) 

#Confusion Matrix

from sklearn.metrics import confusion_matrix

mtrc_confusion_matrix_lgbm_smtk = confusion_matrix(Y_test, Y_pred_lgbm_smtk)

print('Confusion Matrix : \n' + str(mtrc_confusion_matrix_lgbm_smtk))

print(pd.crosstab(Y_test, Y_pred_lgbm_smtk, rownames = ['Real'], colnames = ['Predito'], margins = True))

#Confusion Matrix - Gráfico

import matplotlib.pyplot as plt

import seaborn as sns

LABELS = ['NÃO INTERNOU', 'INTERNOU']
plt.figure(figsize = (7, 7))
sns.heatmap(mtrc_confusion_matrix_lgbm_smtk, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt = "d");
plt.title("Confusion Matrix")
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

#Accuracy Score : 0.9546083587046775
#Balanced Accuracy Score : 0.8510229747645431
#Precision Score : 0.8668049792531121
#Recall Score : 0.71639231824417
#F1 Score : 0.7844536237326324
#Matthews Correlation Coefficient : 0.7636507770250964
#Cohen’s Kappa Score : 0.7593424582285581
#Area Under the Receiver Operating Characteristic Curve (ROC AUC) : 0.8510229747645431
#Area Under the Precision-Recall Curve : 0.6536718077785383

Accuracy Score : 0.9554386936064213
Balanced Accuracy Score : 0.8567114973446445
Precision Score : 0.8637657584383895
Recall Score : 0.7283950617283951
F1 Score : 0.7903255813953489
Matthews Correlation Coefficient : 0.7690793361334619
Cohen’s Kappa Score : 0.7655972841119398
Area Under the Receiver Operating Characteristic Curve (ROC AUC) : 0.8567114973446445
Area Under the Precision-Recall Curve : 0.6604782006595101