#ETAPA 01: TRATAMENTO DOS DADOS - CHEGADA DO PACIENTE

#Importação do Pandas

import pandas as pd

#Leitura da Base de Dados

#Criação da Base df_chegada A Fim de Mapear Apenas as Informações Disponíveis na Chegada do Paciente

df_chegada = pd.read_excel('BD-EMERGENCIA-0505.xlsx')

#Análise de Valores Nulos ou Não Existentes

df_chegada.isnull().sum()

df_chegada.isna().sum()

for col in df_chegada.columns:
    if df_chegada[col].isnull().sum() > 0:
        print(col)

#Tratamento das Colunas Que Não Serão Utilizadas Nesta Etapa de Previsão
        
df_chegada['COD_CLAC_RISCO']                    = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_02']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_03']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_04']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_05']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_06']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_07']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_08']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_09']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_10']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_11']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_12']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_13']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_14']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_15']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_16']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_17']               = -1
df_chegada['PERGUNTA_TRIAGEM_18']               = -1
df_chegada['PERGUNTA_TRIAGEM_19']               = -1
df_chegada['PERGUNTA_TRIAGEM_20']               = -1
df_chegada['PERGUNTA_TRIAGEM_21']               = -1
df_chegada['PERGUNTA_TRIAGEM_22']               = -1
df_chegada['PERGUNTA_TRIAGEM_23']               = -1
df_chegada['PERGUNTA_TRIAGEM_24']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_25']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_26']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_27']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_28']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_29']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_30']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_31']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_32']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_TRIAGEM_33']               = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_02']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_03']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_04']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_05']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_06']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_07']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_08']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_09']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_10']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_11']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_12']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_13']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_14']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_15']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_16']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_17']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_18']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_19']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_20']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_21']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_22']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_23']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_24']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_25']          = 'SEM REGISTRO'
df_chegada['PERGUNTA_FICHA_EME_PA_26']          = 'SEM REGISTRO'

df_chegada['COD_CID_ADM']                       = 'SEM REGISTRO'
df_chegada['COD_CID_ADM_PRINCIPAL']             = 'SEM REGISTRO'
df_chegada['PERGUNTA_CONSULTA_P_01']            = -1
df_chegada['PERGUNTA_CONSULTA_P_02']            = -1
df_chegada['PERGUNTA_CONSULTA_P_03']            = -1
df_chegada['PERGUNTA_CONSULTA_P_04']            = -1
df_chegada['PERGUNTA_CONSULTA_P_05']            = -1
df_chegada['PERGUNTA_CONSULTA_P_06']            = -1
df_chegada['PERGUNTA_CONSULTA_P_08']            = -1
df_chegada['PERGUNTA_CONSULTA_P_09']            = 'SEM REGISTRO'
df_chegada['PERGUNTA_CONSULTA_P_10']            = 'SEM REGISTRO'
df_chegada['PERGUNTA_CONSULTA_P_11']            = -1
df_chegada['PERGUNTA_CONSULTA_P_12']            = 'SEM REGISTRO'
df_chegada['PERGUNTA_CONSULTA_P_14']            = 'SEM REGISTRO'
df_chegada['PERGUNTA_CONSULTA_P_15']            = 'SEM REGISTRO'
df_chegada['PERGUNTA_CONSULTA_P_16']            = 'SEM REGISTRO'
df_chegada['INDICADOR_IAM_CSST']                = 'SEM REGISTRO'
df_chegada['EXST_PEX_BIOQUIMICA']               = -1
df_chegada['EXST_PEX_HEMATOLOGIA']              = -1
df_chegada['EXST_PEX_MICROBIOLOGIA']            = -1
df_chegada['EXST_PEX_LAB_CLINICO']              = -1
df_chegada['EXST_PEX_ELETROCARDIOGRAFIA']       = -1
df_chegada['EXST_PEX_RADIOLOGIA']               = -1
df_chegada['EXST_PEX_TOMO_COMPUTADORIZADA']     = -1
df_chegada['EXST_PEX_IMUNOLOGIA']               = -1
df_chegada['EXST_PEX_UROANALISE']               = -1
df_chegada['EXST_PEX_HORMONIOS']                = -1
df_chegada['EXST_PEX_ULTRASSONOGRAFIA']         = -1
df_chegada['EXST_PEX_ECOCARDIOGRAFIA']          = -1
df_chegada['EXST_PEX_DOPPLER']                  = -1
df_chegada['EXST_PEX_EXM_LIQUOR']               = -1
df_chegada['EXST_PEX_PARASITOLOGIA']            = -1
df_chegada['EXST_PEX_FEZES']                    = -1
df_chegada['EXST_PEX_EXM_ESPECIAIS']            = -1
df_chegada['EXST_PEX_EXM_ENDOSCOPIA']           = -1
df_chegada['EXST_PEX_ELETROFISIOLOGICOS']       = -1
df_chegada['EXST_PEX_ERGOMETRIA']               = -1
df_chegada['EXST_PEX_MET_INTERV_IMG']           = -1
df_chegada['EXST_PEX_COLONOSCOPIA']             = -1
df_chegada['EXST_PEX_BRONCOSCOPIA']             = -1
df_chegada['EXST_PEX_HOLTER']                   = -1
df_chegada['EXST_PEX_AP_CARDIOVASCULAR']        = -1
df_chegada['EXST_PEX_AP_RESPIRATORIO']          = -1
df_chegada['EXST_PEX_AP_ANAT_PATOLOGICO']       = -1
df_chegada['EXST_PEX_AP_DIGESTIVO']             = -1
df_chegada['EXST_PEX_ESPIROMETRIA']             = -1
df_chegada['EXST_PEX_GINECOLOGIA']              = -1
df_chegada['EXST_PSC_MED_DIPIRONA']             = -1
df_chegada['EXST_PSC_MED_CLORETO_SODIO']        = -1
df_chegada['EXST_PSC_MED_BROMOPRIDA']           = -1
df_chegada['EXST_PSC_MED_DRAMIN']               = -1
df_chegada['EXST_PSC_MED_GLICOSE']              = -1
df_chegada['EXST_PSC_MED_RINGER_LACTATO']       = -1
df_chegada['EXST_PSC_MED_AAS']                  = -1
df_chegada['EXST_PSC_MED_CEFTRIAXONE']          = -1
df_chegada['EXST_PSC_MED_FUROSEMIDA']           = -1
df_chegada['EXST_PSC_MED_BUSCOPAN_COMPOSTO']    = -1
df_chegada['EXST_PSC_MED_TRAMADOL']             = -1
df_chegada['EXST_PSC_MED_MORFINA']              = -1
df_chegada['EXST_PSC_MED_PANTOPRAZOL']          = -1
df_chegada['EXST_PSC_MED_ATORVASTATINA']        = -1
df_chegada['EXST_PSC_MED_RANITIDINA']           = -1
df_chegada['EXST_PSC_MED_ENOXAPARINA']          = -1
df_chegada['EXST_PSC_MED_CLOPIDOGREL']          = -1
df_chegada['EXST_PSC_MED_CLONAZEPAN']           = -1
df_chegada['EXST_PSC_MED_METOPROLOL']           = -1
df_chegada['EXST_PSC_MED_CETOPROFENO']          = -1
df_chegada['EXST_PSC_MED_METILPREDNISOLONA']    = -1
df_chegada['EXST_PSC_MED_NITROGLICERINA']       = -1
df_chegada['EXST_PSC_MED_ISOSSORBIDA']          = -1
df_chegada['EXST_PSC_MED_LOSARTAN']             = -1
df_chegada['EXST_PSC_MED_AZITROMICINA']         = -1
df_chegada['EXST_PSC_MED_CIPROFLOXACINA']       = -1
df_chegada['EXST_PSC_MED_ONDANSETRON']          = -1
df_chegada['EXST_PSC_MED_LTIROXINA']            = -1
df_chegada['EXST_PSC_MED_ANLODIPINA']           = -1
df_chegada['EXST_PSC_MED_CLORETOSG']            = -1

df_chegada['EXST_REGISTRO_PUNCAO_VENOSA']       = -1
df_chegada['CONT_REGISTRO_PUNCAO_VENOSA']       = -1
df_chegada['EXST_REGISTRO_RESULT_CRITICO']      = -1
df_chegada['CONT_REGISTRO_RESULT_CRITICO']      = -1
df_chegada['EXST_REGISTRO_FICHA_EVOLUCAO']      = -1
df_chegada['CONT_REGISTRO_FICHA_EVOLUCAO']      = -1
df_chegada['EXST_REGISTRO_CONTROLE_DOR']        = -1
df_chegada['CONT_REGISTRO_CONTROLE_DOR']        = -1
df_chegada['EXST_REGISTRO_FICHA_EME_PA']        = -1
df_chegada['EXST_ALERTA_ALERGIA']               = -1
df_chegada['EXST_ALERTA_ALTO_RISCO_VULNERA']    = -1
df_chegada['EXST_ALERTA_PROTOCOLO_AVC']         = -1
df_chegada['EXST_ALERTA_RISCO_DESCUTRICAO']     = -1
df_chegada['EXST_ALERTA_PROTOCOLO_DT']          = -1
df_chegada['EXST_ALERTA_EXTUBACAO_ACIDNTAL']    = -1
df_chegada['EXST_ALERTA_EXTERIORIZACAO_CVC']    = -1
df_chegada['EXST_ALERTA_HIPOGLICEMIA']          = -1
df_chegada['EXST_ALERTA_PROTOCOLO_IAM']         = -1
df_chegada['EXST_ALERTA_PROTOCOLO_ICC']         = -1
df_chegada['EXST_ALERTA_INFECCAO_CVC']          = -1
df_chegada['EXST_ALERTA_ITU_ASSOC_SVD']         = -1
df_chegada['EXST_ALERTA_MOBILIDADE_1_A_5']      = -1
df_chegada['EXST_ALERTA_EXTERIORIZACAO_SEG']    = -1
df_chegada['EXST_ALERTA_RISCO_PAV']             = -1
df_chegada['EXST_ALERTA_PROTOCOLO_PNEUMO']      = -1
df_chegada['EXST_ALERTA_PRECAUCAO_CONTATO']     = -1
df_chegada['EXST_ALERTA_PROTOCOLO']             = -1
df_chegada['EXST_ALERTA_RISCO_BRONCOASPIRA']    = -1
df_chegada['EXST_ALERTA_RISCO_RESTRICAO_MC']    = -1
df_chegada['EXST_ALERTA_RISCO_FLEBITE']         = -1
df_chegada['EXST_ALERTA_RISCO_OBSTRUC_SEG']     = -1
df_chegada['EXST_ALERTA_RISCO_PSICOLOGICO']     = -1
df_chegada['EXST_ALERTA_RISCO_QUEDA']           = -1
df_chegada['EXST_ALERTA_RISCO_SANGRAMENTO']     = -1
df_chegada['EXST_ALERTA_PROTOCOLO_SEPSE']       = -1
df_chegada['EXST_ALERTA_PROTOCOLO_TEV1']        = -1
df_chegada['EXST_ALERTA_PROTOCOLO_TEV2']        = -1
df_chegada['EXST_ALERTA_PROTOCOLO_TEV3']        = -1
df_chegada['EXST_ALERTA_RISCO_LESAO_PRESS']     = -1
df_chegada['EXST_ALERTA_RISCO_ITU']             = -1
df_chegada['CSV_MIN_TEMPERATURA']               = -1
df_chegada['CSV_MAX_TEMPERATURA']               = -1
df_chegada['CSV_MEDIA_TEMPERATURA']             = -1
df_chegada['CSV_MIN_TA_SISTOLICA']              = -1
df_chegada['CSV_MAX_TA_SISTOLICA']              = -1
df_chegada['CSV_MEDIA_TA_SISTOLICA']            = -1
df_chegada['CSV_MIN_TA_DIASTOLICA']             = -1
df_chegada['CSV_MAX_TA_DIASTOLICA']             = -1
df_chegada['CSV_MEDIA_TA_DIASTOLICA']           = -1
df_chegada['CSV_MIN_FREQ_RESPIRATORIA']         = -1
df_chegada['CSV_MAX_FREQ_RESPIRATORIA']         = -1
df_chegada['CSV_MEDIA_FREQ_RESPIRATORIA']       = -1
df_chegada['CSV_MIN_GLICEMIA_DIGITAL']          = -1
df_chegada['CSV_MAX_GLICEMIA_DIGITAL']          = -1
df_chegada['CSV_MEDIA_GLICEMIA_DIGITAL']        = -1
df_chegada['CSV_MIN_SATURACAO_O2']              = -1
df_chegada['CSV_MAX_SATURACAO_O2']              = -1
df_chegada['CSV_MEDIA_SATURACAO_O2']            = -1
df_chegada['CSV_MIN_PRESSAO_ARTERIAL']          = -1
df_chegada['CSV_MAX_PRESSAO_ARTERIAL']          = -1
df_chegada['CSV_MEDIA_PRESSAO_ARTERIAL']        = -1
df_chegada['CSV_MIN_FREQ_CARDIACA']             = -1
df_chegada['CSV_MAX_FREQ_CARDIACA']             = -1
df_chegada['CSV_MEDIA_FREQ_CARDIACA']           = -1

#Verificação dos Valores Nulos Ainda Existentes

for col in df_chegada.columns:
    if df_chegada[col].isnull().sum() > 0:
        print(col)

#Preenchimento dos Valores Nulos
        
pd.value_counts(df_chegada['NUM_PESO'])
df_chegada['NUM_PESO'].count()
df_chegada['NUM_PESO'].fillna(-1, inplace = True)

pd.value_counts(df_chegada['NUM_ALTURA'])
df_chegada['NUM_ALTURA'].count()
df_chegada['NUM_ALTURA'].fillna(-1, inplace = True)

pd.value_counts(df_chegada['DESC_BAIRRO'])
df_chegada['DESC_BAIRRO'].count()
df_chegada['DESC_BAIRRO'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_chegada['COD_CID'])
df_chegada['COD_CID'].count()
df_chegada['COD_CID'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_chegada['DESC_GRUPO_SANGUINEO'])
df_chegada['DESC_GRUPO_SANGUINEO'].count()
df_chegada['DESC_GRUPO_SANGUINEO'].fillna('SEM REGISTRO', inplace = True)

#Verificação dos Valores Nulos Ainda Existentes

for col in df_chegada.columns:
    if df_chegada[col].isnull().sum() > 0:
        print(col)

#Exclusão das Colunas Que Não Serão Utilizadas

df_chegada_trat = df_chegada.drop('DESC_DIA_DA_SEMANA', axis = 1)

df_chegada_trat = df_chegada_trat.drop('REGISTRO', axis = 1)

df_chegada_trat = df_chegada_trat.drop('NOME_PACIENTE', axis = 1)

df_chegada_trat = df_chegada_trat.drop('DESC_SEXO', axis = 1)

df_chegada_trat = df_chegada_trat.drop('DESC_FAIXA_ETARIA', axis = 1)

df_chegada_trat = df_chegada_trat.drop('NUM_TRATAMENTO', axis = 1)

df_chegada_trat = df_chegada_trat.drop('DATA_ADMISSAO', axis = 1)

df_chegada_trat = df_chegada_trat.drop('DESC_DATA_ADMISSAO', axis = 1)

df_chegada_trat = df_chegada_trat.drop('DESC_CONVENIO', axis = 1)

df_chegada_trat = df_chegada_trat.drop('COD_EMPRESA', axis = 1)

df_chegada_trat = df_chegada_trat.drop('DESC_EMPRESA', axis = 1)

df_chegada_trat = df_chegada_trat.drop('COD_PLANTONISTA', axis = 1)

df_chegada_trat = df_chegada_trat.drop('NOME_PLANTONISTA', axis = 1)

df_chegada_trat = df_chegada_trat.drop('PERGUNTA_TRIAGEM_01', axis = 1)

df_chegada_trat = df_chegada_trat.drop('PERGUNTA_FICHA_EME_PA_01', axis = 1)

df_chegada_trat = df_chegada_trat.drop('PERGUNTA_CONSULTA_P_07', axis = 1)

df_chegada_trat = df_chegada_trat.drop('PERGUNTA_CONSULTA_P_13', axis = 1)

df_chegada_trat = df_chegada_trat.drop('DESC_DATA_ALTA', axis = 1)

df_chegada_trat = df_chegada_trat.drop('TEMPO_ATENDIMENTO', axis = 1)

df_chegada_trat = df_chegada_trat.drop('DESC_STATUS_INT', axis = 1)

df_chegada_trat = df_chegada_trat.drop('NUM_INT_POSTERIOR', axis = 1)

df_chegada_trat = df_chegada_trat.drop('TEMPO_PERM_INT_POSTERIOR', axis = 1)

df_chegada_trat = df_chegada_trat.drop('CONTADOR', axis = 1)

#Verificação dos Valores Nulos Ainda Existentes

for col in df_chegada_trat.columns:
    if df_chegada_trat[col].isnull().sum() > 0:
        print(col)
        
#Estatísticas Descritivas da Base de Dados - Após Tratamento dos Dados        
        
df_chegada_trat.info([''])

#Criação das Bases X_chegada e Y_chegada

X_chegada = df_chegada_trat

Y_chegada = df_chegada_trat['CONT_STATUS_INT']

X_chegada = X_chegada.drop('CONT_STATUS_INT', axis = 1)

#Estatísticas Descritivas da Base de Dados - X_chegada - Após Remoção da Coluna Target

X_chegada.info([''])

#Aplicação do Label Encoder na Base X_chegada

from sklearn.preprocessing import LabelEncoder

lbl_enc_X_chegada = LabelEncoder()

X_chegada['DESC_BAIRRO'] = lbl_enc_X_chegada.fit_transform(X_chegada['DESC_BAIRRO'])

X_chegada['COD_CID'] = lbl_enc_X_chegada.fit_transform(X_chegada['COD_CID'])

X_chegada['DESC_GRUPO_SANGUINEO'] = lbl_enc_X_chegada.fit_transform(X_chegada['DESC_GRUPO_SANGUINEO'])

X_chegada['APEL_PLANTONISTA'] = lbl_enc_X_chegada.fit_transform(X_chegada['APEL_PLANTONISTA'])

X_chegada['CID_PREVALENTE'] = lbl_enc_X_chegada.fit_transform(X_chegada['CID_PREVALENTE'])

X_chegada['COD_CLAC_RISCO'] = lbl_enc_X_chegada.fit_transform(X_chegada['COD_CLAC_RISCO'])

X_chegada['PERGUNTA_TRIAGEM_02'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_02'])

X_chegada['PERGUNTA_TRIAGEM_03'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_03'])

X_chegada['PERGUNTA_TRIAGEM_04'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_04'])

X_chegada['PERGUNTA_TRIAGEM_05'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_05'])

X_chegada['PERGUNTA_TRIAGEM_06'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_06'])

X_chegada['PERGUNTA_TRIAGEM_07'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_07'])

X_chegada['PERGUNTA_TRIAGEM_08'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_08'])

X_chegada['PERGUNTA_TRIAGEM_09'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_09'])

X_chegada['PERGUNTA_TRIAGEM_10'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_10'])

X_chegada['PERGUNTA_TRIAGEM_11'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_11'])

X_chegada['PERGUNTA_TRIAGEM_12'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_12'])

X_chegada['PERGUNTA_TRIAGEM_13'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_13'])

X_chegada['PERGUNTA_TRIAGEM_14'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_14'])

X_chegada['PERGUNTA_TRIAGEM_15'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_15'])

X_chegada['PERGUNTA_TRIAGEM_16'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_16'])

X_chegada['PERGUNTA_TRIAGEM_24'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_24'])

X_chegada['PERGUNTA_TRIAGEM_25'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_25'])

X_chegada['PERGUNTA_TRIAGEM_26'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_26'])

X_chegada['PERGUNTA_TRIAGEM_27'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_27'])

X_chegada['PERGUNTA_TRIAGEM_28'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_28'])

X_chegada['PERGUNTA_TRIAGEM_29'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_29'])

X_chegada['PERGUNTA_TRIAGEM_30'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_30'])

X_chegada['PERGUNTA_TRIAGEM_31'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_31'])

X_chegada['PERGUNTA_TRIAGEM_32'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_32'])

X_chegada['PERGUNTA_TRIAGEM_33'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_TRIAGEM_33'])

X_chegada['PERGUNTA_FICHA_EME_PA_02'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_02'])

X_chegada['PERGUNTA_FICHA_EME_PA_03'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_03'])

X_chegada['PERGUNTA_FICHA_EME_PA_04'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_04'])

X_chegada['PERGUNTA_FICHA_EME_PA_05'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_05'])

X_chegada['PERGUNTA_FICHA_EME_PA_06'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_06'])

X_chegada['PERGUNTA_FICHA_EME_PA_07'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_07'])

X_chegada['PERGUNTA_FICHA_EME_PA_08'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_08'])

X_chegada['PERGUNTA_FICHA_EME_PA_09'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_09'])

X_chegada['PERGUNTA_FICHA_EME_PA_10'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_10'])

X_chegada['PERGUNTA_FICHA_EME_PA_11'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_11'])

X_chegada['PERGUNTA_FICHA_EME_PA_12'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_12'])

X_chegada['PERGUNTA_FICHA_EME_PA_13'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_13'])

X_chegada['PERGUNTA_FICHA_EME_PA_14'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_14'])

X_chegada['PERGUNTA_FICHA_EME_PA_15'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_15'])

X_chegada['PERGUNTA_FICHA_EME_PA_16'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_16'])

X_chegada['PERGUNTA_FICHA_EME_PA_17'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_17'])

X_chegada['PERGUNTA_FICHA_EME_PA_18'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_18'])

X_chegada['PERGUNTA_FICHA_EME_PA_19'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_19'])

X_chegada['PERGUNTA_FICHA_EME_PA_20'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_20'])

X_chegada['PERGUNTA_FICHA_EME_PA_21'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_21'])

X_chegada['PERGUNTA_FICHA_EME_PA_22'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_22'])

X_chegada['PERGUNTA_FICHA_EME_PA_23'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_23'])

X_chegada['PERGUNTA_FICHA_EME_PA_24'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_24'])

X_chegada['PERGUNTA_FICHA_EME_PA_25'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_25'])

X_chegada['PERGUNTA_FICHA_EME_PA_26'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_FICHA_EME_PA_26'])

X_chegada['COD_CID_ADM'] = lbl_enc_X_chegada.fit_transform(X_chegada['COD_CID_ADM'])

X_chegada['COD_CID_ADM_PRINCIPAL'] = lbl_enc_X_chegada.fit_transform(X_chegada['COD_CID_ADM_PRINCIPAL'])

X_chegada['PERGUNTA_CONSULTA_P_09'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_CONSULTA_P_09'])

X_chegada['PERGUNTA_CONSULTA_P_10'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_CONSULTA_P_10'])

X_chegada['PERGUNTA_CONSULTA_P_12'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_CONSULTA_P_12'])

X_chegada['PERGUNTA_CONSULTA_P_14'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_CONSULTA_P_14'])

X_chegada['PERGUNTA_CONSULTA_P_15'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_CONSULTA_P_15'])

X_chegada['PERGUNTA_CONSULTA_P_16'] = lbl_enc_X_chegada.fit_transform(X_chegada['PERGUNTA_CONSULTA_P_16'])

X_chegada['INDICADOR_IAM_CSST'] = lbl_enc_X_chegada.fit_transform(X_chegada['INDICADOR_IAM_CSST'])

#Estatísticas Descritivas da Base de Dados - X_chegada - Após Aplicação do Label Encoder

X_chegada.info([''])

#Transformação da Base Y_chegada em Array

import numpy as np

Y_chegada = np.array(Y_chegada)

#Transformação da Base X_chegada em Array

X_chegada_list = list(X_chegada.columns)

X_chegada = np.array(X_chegada)

#Escalonamento da Base X_chegada - Opção 01 - StandardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_chegada = scaler.fit_transform(X_chegada)

#Escalonamento da Base X_chegada - Opção 02 - MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_chegada = scaler.fit_transform(X_chegada)

#Escalonamento da Base X_chegada - Opção 03 - Normalizer

from sklearn.preprocessing import Normalizer

scaler = Normalizer()

X_chegada = scaler.fit_transform(X_chegada)

#ETAPA 02: DIVISÃO DA BASE DE DADOS EM TESTE E TREINAMENTO

#Criação da Base de Teste e Treinamento - Sem Balanceamento em Y

from sklearn.model_selection import train_test_split

X_chegada_train, X_chegada_test, Y_chegada_train, Y_chegada_test = train_test_split(X_chegada, Y_chegada, test_size = 0.3, random_state = 42)

print('Training X Data Dimensions:', X_chegada_train.shape)
print('Training Y Data Dimensions:', Y_chegada_train.shape)
print('Testing X Data Dimensions:', X_chegada_test.shape)
print('Testing Y Data Dimensions:', Y_chegada_test.shape)

print('Labels counts in Y_chegada:', np.bincount(Y_chegada))

print('Labels counts in Y_chegada_train:', np.bincount(Y_chegada_train))

print('Labels counts in Y_chegada_test:', np.bincount(Y_chegada_test))

#Criação da Base de Teste e Treinamento - Com Balanceamento em Y

from sklearn.model_selection import train_test_split

X_chegada_train, X_chegada_test, Y_chegada_train, Y_chegada_test = train_test_split(X_chegada, Y_chegada, test_size = 0.3, random_state = 42, stratify = Y_chegada)

print('Training X Data Dimensions:', X_chegada_train.shape)
print('Training Y Data Dimensions:', Y_chegada_train.shape)
print('Testing X Data Dimensions:', X_chegada_test.shape)
print('Testing Y Data Dimensions:', Y_chegada_test.shape)

print('Labels counts in Y_chegada:', np.bincount(Y_chegada))

print('Labels counts in Y_chegada_train:', np.bincount(Y_chegada_train))

print('Labels counts in Y_chegada_test:', np.bincount(Y_chegada_test))
        
#ETAPA 03: APLICAÇÃO MODELO LIGHT GBM

#Aplicação do Modelo Construído em emergencia.py na Base X_chegada

Y_chegada_pred_lgbm = classifier_lgbm.predict(X_chegada)

#Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print('Accuracy Score : ' + str(accuracy_score(Y_chegada, Y_chegada_pred_lgbm)))
print('Precision Score : ' + str(precision_score(Y_chegada, Y_chegada_pred_lgbm)))
print('Recall Score : ' + str(recall_score(Y_chegada, Y_chegada_pred_lgbm)))
print('F1 Score : ' + str(f1_score(Y_chegada, Y_chegada_pred_lgbm)))

#Treinamento e Predição com as Bases X_chegada e Y_chegada

from lightgbm import LGBMClassifier

classifier_lgbm_chegada = LGBMClassifier( max_depth = 1000, 
                                          learning_rate = 0.01,
                                          num_leaves = 2000,
                                          min_data_in_leaf = 200,
                                          n_estimators = 5000,
                                          objective = 'binary',
                                          metric = 'binary_logloss' )

classifier_lgbm_chegada.fit(X_chegada_train, Y_chegada_train)

Y_chegada_pred_lgbm = classifier_lgbm_chegada.predict(X_chegada_test)

#Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print('Accuracy Score : ' + str(accuracy_score(Y_chegada_test, Y_chegada_pred_lgbm)))
print('Precision Score : ' + str(precision_score(Y_chegada_test, Y_chegada_pred_lgbm)))
print('Recall Score : ' + str(recall_score(Y_chegada_test, Y_chegada_pred_lgbm)))
print('F1 Score : ' + str(f1_score(Y_chegada_test, Y_chegada_pred_lgbm)))