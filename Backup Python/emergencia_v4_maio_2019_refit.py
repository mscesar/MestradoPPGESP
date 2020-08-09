#MAIO 2019

df_trat.shape
#(84303, 628)

df_abril_19_trat.shape
#(1722, 628)

frames = [df_trat, df_abril_19_trat]

df_ate_abril_19 = pd.concat(frames)

df_ate_abril_19.shape
#(86025, 628)

#Verificação dos Valores Nulos Ainda Existentes

for col in df_ate_abril_19.columns:
    if df_ate_abril_19[col].isnull().sum() > 0:
        print(col)
        
#Criação das Bases X_ate_abril_19 e Y_ate_abril_19

X_ate_abril_19 = df_ate_abril_19

Y_ate_abril_19 = df_ate_abril_19['CONT_STATUS_INT']

X_ate_abril_19 = X_ate_abril_19.drop('CONT_STATUS_INT', axis = 1)

#Estatísticas Descritivas da Base de Dados - X_ate_abril_19 - Após Remoção da Coluna Target

X_ate_abril_19.shape

X_ate_abril_19.info([''])

#del df_ate_abril_19

#Aplicação do Label Encoder na Base X_ate_abril_19

from sklearn.preprocessing import LabelEncoder

lbl_enc_X_ate_abril_19 = LabelEncoder()

X_ate_abril_19['DESC_ESTACAO_ANO'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['DESC_ESTACAO_ANO'])

X_ate_abril_19['DESC_FINAL_SEMANA'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['DESC_FINAL_SEMANA'])

X_ate_abril_19['DESC_BAIRRO'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['DESC_BAIRRO'])

X_ate_abril_19['COD_CID'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['COD_CID'])

X_ate_abril_19['DESC_GRUPO_SANGUINEO'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['DESC_GRUPO_SANGUINEO'])

X_ate_abril_19['APEL_PLANTONISTA'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['APEL_PLANTONISTA'])

X_ate_abril_19['CID_PREVALENTE'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['CID_PREVALENTE'])

X_ate_abril_19['COD_CLAC_RISCO'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['COD_CLAC_RISCO'])

X_ate_abril_19['PERGUNTA_TRIAGEM_02'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_02'])

X_ate_abril_19['PERGUNTA_TRIAGEM_03'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_03'])

X_ate_abril_19['PERGUNTA_TRIAGEM_04'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_04'])

X_ate_abril_19['PERGUNTA_TRIAGEM_05'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_05'])

X_ate_abril_19['PERGUNTA_TRIAGEM_06'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_06'])

X_ate_abril_19['PERGUNTA_TRIAGEM_07'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_07'])

X_ate_abril_19['PERGUNTA_TRIAGEM_08'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_08'])

X_ate_abril_19['PERGUNTA_TRIAGEM_09'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_09'])

X_ate_abril_19['PERGUNTA_TRIAGEM_10'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_10'])

X_ate_abril_19['PERGUNTA_TRIAGEM_11'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_11'])

X_ate_abril_19['PERGUNTA_TRIAGEM_12'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_12'])

X_ate_abril_19['PERGUNTA_TRIAGEM_13'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_13'])

X_ate_abril_19['PERGUNTA_TRIAGEM_14'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_14'])

X_ate_abril_19['PERGUNTA_TRIAGEM_15'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_15'])

X_ate_abril_19['PERGUNTA_TRIAGEM_16'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_16'])

X_ate_abril_19['PERGUNTA_TRIAGEM_24'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_24'])

X_ate_abril_19['PERGUNTA_TRIAGEM_25'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_25'])

X_ate_abril_19['PERGUNTA_TRIAGEM_26'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_26'])

X_ate_abril_19['PERGUNTA_TRIAGEM_27'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_27'])

X_ate_abril_19['PERGUNTA_TRIAGEM_28'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_28'])

X_ate_abril_19['PERGUNTA_TRIAGEM_29'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_29'])

X_ate_abril_19['PERGUNTA_TRIAGEM_30'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_30'])

X_ate_abril_19['PERGUNTA_TRIAGEM_31'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_31'])

X_ate_abril_19['PERGUNTA_TRIAGEM_32'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_32'])

X_ate_abril_19['PERGUNTA_TRIAGEM_33'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_TRIAGEM_33'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_02'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_02'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_03'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_03'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_04'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_04'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_05'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_05'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_06'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_06'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_07'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_07'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_08'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_08'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_09'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_09'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_10'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_10'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_11'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_11'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_12'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_12'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_13'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_13'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_14'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_14'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_15'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_15'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_16'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_16'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_17'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_17'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_18'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_18'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_19'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_19'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_20'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_20'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_21'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_21'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_22'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_22'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_23'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_23'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_24'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_24'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_25'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_25'])

X_ate_abril_19['PERGUNTA_FICHA_EME_PA_26'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_FICHA_EME_PA_26'])

X_ate_abril_19['COD_CID_ADM'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['COD_CID_ADM'])

X_ate_abril_19['COD_CID_ADM_PRINCIPAL'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['COD_CID_ADM_PRINCIPAL'])

X_ate_abril_19['PERGUNTA_CONSULTA_P_09'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_CONSULTA_P_09'])

X_ate_abril_19['PERGUNTA_CONSULTA_P_10'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_CONSULTA_P_10'])

X_ate_abril_19['PERGUNTA_CONSULTA_P_12'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_CONSULTA_P_12'])

X_ate_abril_19['PERGUNTA_CONSULTA_P_14'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_CONSULTA_P_14'])

X_ate_abril_19['PERGUNTA_CONSULTA_P_15'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_CONSULTA_P_15'])

X_ate_abril_19['PERGUNTA_CONSULTA_P_16'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['PERGUNTA_CONSULTA_P_16'])

X_ate_abril_19['INDICADOR_IAM_CSST'] = lbl_enc_X_ate_abril_19.fit_transform(X_ate_abril_19['INDICADOR_IAM_CSST'])

#Estatísticas Descritivas da Base de Dados - X_ate_abril_19 - Após Aplicação do Label Encoder

X_ate_abril_19.info([''])

#Verificação da Distribuição dos Tipos de Dados das Variáveis - Base X_ate_abril_19

feature_list_X_ate_abril_19 = list(X_ate_abril_19.columns)

feature_list_X_ate_abril_19_integer, feature_list_X_ate_abril_19_float, features_list_X_ate_abril_19_object = groupFeatures(feature_list_X_ate_abril_19, X_ate_abril_19)

print("# of integer feature : ", len(feature_list_X_ate_abril_19_integer))
print("# of float feature : ", len(feature_list_X_ate_abril_19_float))
print("# of object feature : ", len(features_list_X_ate_abril_19_object))

#Transformação da Base Y_ate_abril_19 em Array

import numpy as np

Y_ate_abril_19 = np.array(Y_ate_abril_19)

#Transformação da Base X_ate_abril_19 em Array

X_ate_abril_19_list = list(X_ate_abril_19.columns)

X_ate_abril_19_list_array = np.array(X_ate_abril_19_list)

X_ate_abril_19 = np.array(X_ate_abril_19)

#Escalonamento da Base X_ate_abril_19 - Opção 01 - StandardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_ate_abril_19 = scaler.fit_transform(X_ate_abril_19)        

#ETAPA 02: FIT DO MODELO COM X E Y

import time
start = time.time()

classifier_lgbm.fit(X_ate_abril_19, Y_ate_abril_19)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 30.36 min

#ETAPA 03: PREDIÇÃO

Y_maio_19_refit_pred_lgbm = classifier_lgbm.predict(X_maio_19)

#Análise de Métricas

from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm = accuracy_score(Y_maio_19, Y_maio_19_refit_pred_lgbm)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm))

#Balanced Accuracy

mtrc_balanced_accuracy_score_lgbm = balanced_accuracy_score(Y_maio_19, Y_maio_19_refit_pred_lgbm)
print('Balanced Accuracy Score : ' + str(mtrc_balanced_accuracy_score_lgbm))

#Precision Score
#Número de vezes que uma classe foi predita corretamente dividida pelo número de vezes que a classe foi predita.

mtrc_precision_score_lgbm = precision_score(Y_maio_19, Y_maio_19_refit_pred_lgbm)
print('Precision Score : ' + str(mtrc_precision_score_lgbm))

#Recall Score
#Número de vezes que uma classe foi predita corretamente (TP) dividido pelo número de vezes que a classe aparece no dado de teste (TP + FN).

mtrc_recall_score_lgbm = recall_score(Y_maio_19, Y_maio_19_refit_pred_lgbm)
print('Recall Score : ' + str(mtrc_recall_score_lgbm))

#F1 Score
#Média harmônica entre precisão e revocação. Com essa informação podemos dizer a performance do classificador com um indicador apenas.

mtrc_f1_score_lgbm = f1_score(Y_maio_19, Y_maio_19_refit_pred_lgbm)
print('F1 Score : ' + str(mtrc_f1_score_lgbm))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm = matthews_corrcoef(Y_maio_19, Y_maio_19_refit_pred_lgbm)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm = cohen_kappa_score(Y_maio_19, Y_maio_19_refit_pred_lgbm)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm))

#Area Under the Receiver Operating Characteristic Curve (ROC AUC)

from sklearn.metrics import roc_auc_score

mtrc_roc_auc_score_lgbm = roc_auc_score(Y_maio_19, Y_maio_19_refit_pred_lgbm)
print('Area Under the Receiver Operating Characteristic Curve (ROC AUC) : ' + str(mtrc_roc_auc_score_lgbm))

#Area Under the Precision-Recall Curve

from sklearn.metrics import average_precision_score

mtrc_average_precision_score_lgbm = average_precision_score(Y_maio_19, Y_maio_19_refit_pred_lgbm)
print('Area Under the Precision-Recall Curve : ' + str(mtrc_average_precision_score_lgbm))

#Confusion Matrix

from sklearn.metrics import confusion_matrix

mtrc_confusion_matrix_lgbm = confusion_matrix(Y_maio_19, Y_maio_19_refit_pred_lgbm)

print('Confusion Matrix : \n' + str(mtrc_confusion_matrix_lgbm))

print(pd.crosstab(Y_maio_19, Y_maio_19_refit_pred_lgbm, rownames = ['Real'], colnames = ['Predito'], margins = True))

#Confusion Matrix - True Positives
mtrc_true_positive_lgbm = mtrc_confusion_matrix_lgbm[1, 1]

#Confusion Matrix - True Negatives
mtrc_true_negative_lgbm = mtrc_confusion_matrix_lgbm[0, 0]

#Confusion Matrix - False Positives
mtrc_false_positive_lgbm = mtrc_confusion_matrix_lgbm[0, 1]

#Confusion Matrix - False Negatives
mtrc_false_negative_lgbm = mtrc_confusion_matrix_lgbm[1, 0]

#cm_true_negative_lgbm, cm_false_positive_lgbm, cm_false_negative_lgbm, cm_true_positive_lgbm = confusion_matrix(Y_maio_19, Y_maio_19_refit_pred_lgbm).ravel()

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
Accuracy Score : 0.9271523178807947
Balanced Accuracy Score : 0.798369061004307
Precision Score : 0.8011363636363636
Recall Score : 0.6211453744493393
F1 Score : 0.6997518610421837
Matthews Correlation Coefficient : 0.665965492900038
Cohen’s Kappa Score : 0.6590531638955757
Area Under the Receiver Operating Characteristic Curve (ROC AUC) : 0.798369061004307
Area Under the Precision-Recall Curve : 0.5493981851068965