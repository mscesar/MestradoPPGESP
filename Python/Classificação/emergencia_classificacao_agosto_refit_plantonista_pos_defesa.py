#Setup da Pasta de Trabalho
import os
os.chdir('C:/Users/Mauricio/Dropbox/MauricioPessoal/Pós-Graduação/PPGESP/Projeto Final/Estudo Python/Bases Cárdio Pulmonar')

import warnings
warnings.filterwarnings("ignore")

#ETAPA 03: PREDIÇÃO

Y_agosto_19_plantonista_refit_pred_lgbm = classifier_lgbm.predict(X_agosto_19_plantonista)

#Análise de Métricas

from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm = accuracy_score(Y_agosto_19_plantonista, Y_agosto_19_plantonista_refit_pred_lgbm)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm))

#Balanced Accuracy

mtrc_balanced_accuracy_score_lgbm = balanced_accuracy_score(Y_agosto_19_plantonista, Y_agosto_19_plantonista_refit_pred_lgbm)
print('Balanced Accuracy Score : ' + str(mtrc_balanced_accuracy_score_lgbm))

#Precision Score
#Número de vezes que uma classe foi predita corretamente dividida pelo número de vezes que a classe foi predita.

mtrc_precision_score_lgbm = precision_score(Y_agosto_19_plantonista, Y_agosto_19_plantonista_refit_pred_lgbm)
print('Precision Score : ' + str(mtrc_precision_score_lgbm))

#Recall Score
#Número de vezes que uma classe foi predita corretamente (TP) dividido pelo número de vezes que a classe aparece no dado de teste (TP + FN).

mtrc_recall_score_lgbm = recall_score(Y_agosto_19_plantonista, Y_agosto_19_plantonista_refit_pred_lgbm)
print('Recall Score : ' + str(mtrc_recall_score_lgbm))

#F1 Score
#Média harmônica entre precisão e revocação. Com essa informação podemos dizer a performance do classificador com um indicador apenas.

mtrc_f1_score_lgbm = f1_score(Y_agosto_19_plantonista, Y_agosto_19_plantonista_refit_pred_lgbm)
print('F1 Score : ' + str(mtrc_f1_score_lgbm))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm = matthews_corrcoef(Y_agosto_19_plantonista, Y_agosto_19_plantonista_refit_pred_lgbm)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm = cohen_kappa_score(Y_agosto_19_plantonista, Y_agosto_19_plantonista_refit_pred_lgbm)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm))

#Area Under the Receiver Operating Characteristic Curve (ROC AUC)

from sklearn.metrics import roc_auc_score

mtrc_roc_auc_score_lgbm = roc_auc_score(Y_agosto_19_plantonista, Y_agosto_19_plantonista_refit_pred_lgbm)
print('Area Under the Receiver Operating Characteristic Curve (ROC AUC) : ' + str(mtrc_roc_auc_score_lgbm))

#Area Under the Precision-Recall Curve

from sklearn.metrics import average_precision_score

mtrc_average_precision_score_lgbm = average_precision_score(Y_agosto_19_plantonista, Y_agosto_19_plantonista_refit_pred_lgbm)
print('Area Under the Precision-Recall Curve : ' + str(mtrc_average_precision_score_lgbm))

#Confusion Matrix

from sklearn.metrics import confusion_matrix

mtrc_confusion_matrix_lgbm = confusion_matrix(Y_agosto_19_plantonista, Y_agosto_19_plantonista_refit_pred_lgbm)

print('Confusion Matrix : \n' + str(mtrc_confusion_matrix_lgbm))

print(pd.crosstab(Y_agosto_19_plantonista, Y_agosto_19_plantonista_refit_pred_lgbm, rownames = ['Real'], colnames = ['Predito'], margins = True))

#Confusion Matrix - True Positives
mtrc_true_positive_lgbm = mtrc_confusion_matrix_lgbm[1, 1]

#Confusion Matrix - True Negatives
mtrc_true_negative_lgbm = mtrc_confusion_matrix_lgbm[0, 0]

#Confusion Matrix - False Positives
mtrc_false_positive_lgbm = mtrc_confusion_matrix_lgbm[0, 1]

#Confusion Matrix - False Negatives
mtrc_false_negative_lgbm = mtrc_confusion_matrix_lgbm[1, 0]

#cm_true_negative_lgbm, cm_false_positive_lgbm, cm_false_negative_lgbm, cm_true_positive_lgbm = confusion_matrix(Y_agosto_19_plantonista, Y_agosto_19_plantonista_refit_pred_lgbm).ravel()

#Confusion Matrix - Gráfico

import matplotlib.pyplot as plt

import seaborn as sns

LABELS = ['NÃO INTERNOU', 'INTERNOU']
plt.figure(figsize = (7, 7))
sns.heatmap(mtrc_confusion_matrix_lgbm, xticklabels = LABELS, yticklabels = LABELS, annot = True, fmt = "d");
plt.title("Confusion Matrix")
plt.ylabel('True Class')
plt.xlabel('Predicted Class')
plt.show()

#Resultados
Accuracy Score : 0.562039312039312
Balanced Accuracy Score : 0.7342944676524397
Precision Score : 0.22417582417582418
Recall Score : 0.966824644549763
F1 Score : 0.36396074933095446
Matthews Correlation Coefficient : 0.3169824836924515
Cohen’s Kappa Score : 0.19445539387411792
Area Under the Receiver Operating Characteristic Curve (ROC AUC) : 0.7342944676524398
Area Under the Precision-Recall Curve : 0.22103846582519568