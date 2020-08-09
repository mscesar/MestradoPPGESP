#JUNHO 2019

df_reg_trat.shape
#(8448, 629)

df_abril_19_reg_trat.shape
#(164, 629)

df_maio_19_reg_trat.shape
#(183, 629)

frames = [df_reg_trat, df_abril_19_reg_trat, df_maio_19_reg_trat]

df_ate_maio_19_reg = pd.concat(frames)

df_ate_maio_19_reg.shape
#(8795, 629)

#Verificação dos Valores Nulos Ainda Existentes

for col in df_ate_maio_19_reg.columns:
    if df_ate_maio_19_reg[col].isnull().sum() > 0:
        print(col)
        
#Criação das Bases X_ate_maio_19_reg e Y_ate_maio_19_reg 

X_ate_maio_19_reg = df_ate_maio_19_reg

Y_ate_maio_19_reg = df_ate_maio_19_reg['TEMPO_PERM_INT_POSTERIOR']

X_ate_maio_19_reg = X_ate_maio_19_reg.drop('TEMPO_PERM_INT_POSTERIOR', axis = 1)

X_ate_maio_19_reg.shape

Y_ate_maio_19_reg.shape

#Estatísticas Descritivas da Base de Dados - X_ate_maio_19_reg - Após Remoção da Coluna Target

X_ate_maio_19_reg.info([''])

#Aplicação do Label Encoder na Base X_ate_maio_19_reg

from sklearn.preprocessing import LabelEncoder

lbl_enc_X_ate_maio_19_reg = LabelEncoder()

X_ate_maio_19_reg['DESC_ESTACAO_ANO'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['DESC_ESTACAO_ANO'])

X_ate_maio_19_reg['DESC_FINAL_SEMANA'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['DESC_FINAL_SEMANA'])

X_ate_maio_19_reg['DESC_BAIRRO'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['DESC_BAIRRO'])

X_ate_maio_19_reg['COD_CID'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['COD_CID'])

X_ate_maio_19_reg['DESC_GRUPO_SANGUINEO'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['DESC_GRUPO_SANGUINEO'])

X_ate_maio_19_reg['APEL_PLANTONISTA'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['APEL_PLANTONISTA'])

X_ate_maio_19_reg['CID_PREVALENTE'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['CID_PREVALENTE'])

X_ate_maio_19_reg['COD_CLAC_RISCO'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['COD_CLAC_RISCO'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_02'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_02'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_03'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_03'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_04'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_04'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_05'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_05'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_06'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_06'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_07'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_07'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_08'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_08'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_09'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_09'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_10'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_10'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_11'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_11'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_12'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_12'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_13'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_13'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_14'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_14'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_15'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_15'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_16'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_16'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_24'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_24'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_25'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_25'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_26'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_26'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_27'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_27'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_28'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_28'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_29'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_29'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_30'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_30'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_31'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_31'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_32'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_32'])

X_ate_maio_19_reg['PERGUNTA_TRIAGEM_33'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_TRIAGEM_33'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_02'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_02'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_03'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_03'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_04'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_04'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_05'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_05'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_06'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_06'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_07'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_07'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_08'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_08'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_09'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_09'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_10'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_10'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_11'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_11'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_12'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_12'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_13'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_13'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_14'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_14'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_15'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_15'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_16'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_16'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_17'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_17'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_18'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_18'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_19'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_19'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_20'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_20'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_21'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_21'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_22'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_22'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_23'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_23'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_24'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_24'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_25'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_25'])

X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_26'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_FICHA_EME_PA_26'])

X_ate_maio_19_reg['COD_CID_ADM'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['COD_CID_ADM'])

X_ate_maio_19_reg['COD_CID_ADM_PRINCIPAL'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['COD_CID_ADM_PRINCIPAL'])

X_ate_maio_19_reg['PERGUNTA_CONSULTA_P_09'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_CONSULTA_P_09'])

X_ate_maio_19_reg['PERGUNTA_CONSULTA_P_10'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_CONSULTA_P_10'])

X_ate_maio_19_reg['PERGUNTA_CONSULTA_P_12'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_CONSULTA_P_12'])

X_ate_maio_19_reg['PERGUNTA_CONSULTA_P_14'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_CONSULTA_P_14'])

X_ate_maio_19_reg['PERGUNTA_CONSULTA_P_15'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_CONSULTA_P_15'])

X_ate_maio_19_reg['PERGUNTA_CONSULTA_P_16'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['PERGUNTA_CONSULTA_P_16'])

X_ate_maio_19_reg['INDICADOR_IAM_CSST'] = lbl_enc_X_ate_maio_19_reg.fit_transform(X_ate_maio_19_reg['INDICADOR_IAM_CSST'])

#Estatísticas Descritivas da Base de Dados - X_ate_maio_19_reg - Após Aplicação do Label Encoder

X_ate_maio_19_reg.info([''])

#Verificação da Distribuição dos Tipos de Dados das Variáveis - Base X

def groupFeatures(features, data_frame):
    features_integer = []
    features_float = []
    features_object = []
    for feature in features:
        if data_frame[feature].dtype == 'int64':
            features_integer.append(feature)
        if data_frame[feature].dtype == 'int32':
            features_integer.append(feature)            
        if data_frame[feature].dtype == 'float64':
            features_float.append(feature)
        if data_frame[feature].dtype == 'object':
            features_object.append(feature)
    return features_integer, features_float, features_object

feature_list_X_ate_maio_19_reg = list(X_ate_maio_19_reg.columns)

feature_list_X_ate_maio_19_reg_integer, feature_list_X_ate_maio_19_reg_float, features_list_X_ate_maio_19_reg_object = groupFeatures(feature_list_X_ate_maio_19_reg, X_ate_maio_19_reg)

print("# of integer feature : ", len(feature_list_X_ate_maio_19_reg_integer))
print("# of float feature : ", len(feature_list_X_ate_maio_19_reg_float))
print("# of object feature : ", len(features_list_X_ate_maio_19_reg_object))

#Transformação da Base Y_ate_maio_19_reg em Array

import numpy as np

Y_ate_maio_19_reg = np.array(Y_ate_maio_19_reg)

#Transformação da Base X_ate_maio_19_reg em Array

X_list_reg = list(X_ate_maio_19_reg.columns)

X_ate_maio_19_reg = np.array(X_ate_maio_19_reg)

#Escalonamento da Base X_ate_maio_19_reg - Opção 01 - StandardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_ate_maio_19_reg = scaler.fit_transform(X_ate_maio_19_reg)

#ETAPA 02: FIT DO MODELO COM X_ate_maio_19_reg E Y_ate_maio_19_reg

import time
start = time.time()

regressor_xgb.fit(X_ate_maio_19_reg, Y_ate_maio_19_reg)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

#ETAPA 03: PREDIÇÃO

Y_junho_19_refit_pred_reg_xgb = regressor_xgb.predict(X_junho_19_reg)

#Análise de Métricas

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_xgb = explained_variance_score(Y_junho_19_reg, Y_junho_19_refit_pred_reg_xgb)
mtrc_mae_xgb = mean_absolute_error(Y_junho_19_reg, Y_junho_19_refit_pred_reg_xgb)
mtrc_mdae_xgb = median_absolute_error(Y_junho_19_reg, Y_junho_19_refit_pred_reg_xgb)
mtrc_mse_xgb = mean_squared_error(Y_junho_19_reg, Y_junho_19_refit_pred_reg_xgb)
mtrc_rmse_xgb = np.sqrt(mean_squared_error(Y_junho_19_reg, Y_junho_19_refit_pred_reg_xgb))
mtrc_r2_xgb = r2_score(Y_junho_19_reg, Y_junho_19_refit_pred_reg_xgb)
mtrc_max_error_xgb = max_error(Y_junho_19_reg, Y_junho_19_refit_pred_reg_xgb)

print('Explained Variance Score: {:.16f}'.format(mtrc_evs_xgb))
print('Mean Absolute Error: {:.16f}'.format(mtrc_mae_xgb)) 
print('Median Absolute Error: {:.16f}'.format(mtrc_mdae_xgb)) 
print('Mean Squared Error: {:.16f}'.format(mtrc_mse_xgb))
print('Root Mean Squared Error: {:.16f}'.format(mtrc_rmse_xgb))
print('R² Score: {:.16f}'.format(mtrc_r2_xgb))
print('Max Error: {:.16f}'.format(mtrc_max_error_xgb))

print('Explained Variance Score: {:.4f}'.format(mtrc_evs_xgb))
print('Mean Absolute Error: {:.4f}'.format(mtrc_mae_xgb)) 
print('Median Absolute Error: {:.4f}'.format(mtrc_mdae_xgb)) 
print('Mean Squared Error: {:.4f}'.format(mtrc_mse_xgb))
print('Root Mean Squared Error: {:.4f}'.format(mtrc_rmse_xgb))
print('R² Score: {:.4f}'.format(mtrc_r2_xgb))
print('Max Error: {:.4f}'.format(mtrc_max_error_xgb))

Resultado:
Explained Variance Score: 0.1146955059722654
Mean Absolute Error: 1.9078624395803474
Median Absolute Error: 1.8820387125015259
Mean Squared Error: 5.2578176582606657
Root Mean Squared Error: 2.2929931657684168
R² Score: 0.0671037329666662
Max Error: 6.0449941158294678