#Setup da Pasta de Trabalho
import os
os.chdir('C:/Users/Mauricio/Dropbox/MauricioPessoal/Pós-Graduação/PPGESP/Projeto Final/Estudo Python/Bases Cárdio Pulmonar')

#Variáveis de Uso Geral

seed_reg = 42

#ETAPA 01: TRATAMENTO DOS DADOS

#Importação do Pandas

import pandas as pd

pd.options.display.float_format = '{:,.4f}'.format
pd.set_option('display.float_format', lambda x: '%.3f' % x)

#Leitura da Base de Dados

#Criação da Base df_reg com Todas as Informações Disponíveis

dfr = pd.read_excel('BD-EMERGENCIA-0505.xlsx')

df_reg = dfr.loc[(dfr["CONT_STATUS_INT"] == 1) & (dfr["TEMPO_PERM_INT_POSTERIOR"] <= 6)].copy()

#df_reg = dfr[dfr["CONT_STATUS_INT"] == 1]

#df_reg = dfr[(dfr.CONT_STATUS_INT == 1)]

#Verificação da Quantidade de Linhas e Colunas da Base de Dados

df_reg.shape

#Descrição do Index

df_reg.index

#Contagem de Dados Não Nulos

df_reg.count()

#Verificação do Nome Das Colunas da Base de Dados

df_reg.columns

df_reg.columns.values

#Visualização das n Primeiras Linhas

df_reg.head(10)

#Estatísticas Descritivas da Base de Dados

df_reg.info([''])

df_reg.describe()

#Distribution Target Variable

import matplotlib.pyplot as plt 
 
import seaborn as sns

import numpy as np

sns.set(rc = {'figure.figsize':(14, 7)})
sns.distplot(df_reg['TEMPO_PERM_INT_POSTERIOR'], bins = 30)
plt.show()

plt.figure(figsize = (14, 7))
plt.scatter(range(df_reg.shape[0]), np.sort(df_reg['TEMPO_PERM_INT_POSTERIOR'].values))
plt.xlabel('index')
plt.ylabel('Tempo de Permanência')
plt.title("Tempo de Permanência - Distribution")
plt.show();

#Target Variable Analysis

from scipy import stats

from scipy.stats import norm, skew

(mu, sigma) = norm.fit(df_reg['TEMPO_PERM_INT_POSTERIOR'])

sns.distplot(df_reg['TEMPO_PERM_INT_POSTERIOR'], fit = norm)
plt.ylabel('Frequency')
plt.title('Tempo de Permanência - Distribution')
plt.legend(['Normal Dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc = 'best')

quantile_plot = stats.probplot(df_reg['TEMPO_PERM_INT_POSTERIOR'], plot = plt)

import numpy as np

df_reg['TEMPO_PERM_INT_POSTERIOR'] = np.log1p(df_reg['TEMPO_PERM_INT_POSTERIOR'])

(mu, sigma) = norm.fit(df_reg['TEMPO_PERM_INT_POSTERIOR'])
plt.figure(figsize = (14, 7))
plt.subplot(1,2,1)
sns.distplot(df_reg['TEMPO_PERM_INT_POSTERIOR'], fit = norm)
plt.ylabel('Frequency')
plt.title('Tempo de Permanência - Distribution')
plt.subplot(1, 2, 2)
quantile_plot = stats.probplot(df_reg['TEMPO_PERM_INT_POSTERIOR'], plot = plt)

#Cálculo de Simetria dos Dados

df_reg.skew()

#Análise de Correlação das Variáveis

corr_matrix_reg = df_reg.corr()

corr_matrix_reg["TEMPO_PERM_INT_POSTERIOR"].sort_values(ascending = False)

corr_matrix_spearman_reg = df_reg.corr('spearman')

corr_matrix_spearman_reg["TEMPO_PERM_INT_POSTERIOR"].sort_values(ascending = False)

#Plot Matriz de Correlação - Método Pearson

mask = np.zeros_like(corr_matrix_reg, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
f, ax = plt.subplots(figsize = (25, 15))
sns.heatmap(corr_matrix_reg, mask = mask, vmax = 0.8, vmin = 0.05, annot = True);

#Análise de Valores Nulos ou Não Existentes

df_reg.isnull().sum()

df_reg.isna().sum()

for col in df_reg.columns:
    if df_reg[col].isnull().sum() > 0:
        print(col)
    
#Preenchimento dos Valores Nulos

pd.value_counts(df_reg['NUM_PESO'])
df_reg['NUM_PESO'].count()
df_reg['NUM_PESO'].fillna(-1, inplace = True)

pd.value_counts(df_reg['NUM_ALTURA'])
df_reg['NUM_ALTURA'].count()
df_reg['NUM_ALTURA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['DESC_BAIRRO'])
df_reg['DESC_BAIRRO'].count()
df_reg['DESC_BAIRRO'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['COD_CID'])
df_reg['COD_CID'].count()
df_reg['COD_CID'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['DESC_GRUPO_SANGUINEO'])
df_reg['DESC_GRUPO_SANGUINEO'].count()
df_reg['DESC_GRUPO_SANGUINEO'].fillna('SEM REGISTRO', inplace = True)      

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_02'])
df_reg['PERGUNTA_TRIAGEM_02'].count()
df_reg['PERGUNTA_TRIAGEM_02'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_03'])
df_reg['PERGUNTA_TRIAGEM_03'].count()
df_reg['PERGUNTA_TRIAGEM_03'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_04'])
df_reg['PERGUNTA_TRIAGEM_04'].count()
df_reg['PERGUNTA_TRIAGEM_04'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_05'])
df_reg['PERGUNTA_TRIAGEM_05'].count()
df_reg['PERGUNTA_TRIAGEM_05'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_06'])
df_reg['PERGUNTA_TRIAGEM_06'].count()
df_reg['PERGUNTA_TRIAGEM_06'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_07'])
df_reg['PERGUNTA_TRIAGEM_07'].count()
df_reg['PERGUNTA_TRIAGEM_07'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_08'])
df_reg['PERGUNTA_TRIAGEM_08'].count()
df_reg['PERGUNTA_TRIAGEM_08'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_09'])
df_reg['PERGUNTA_TRIAGEM_09'].count()
df_reg['PERGUNTA_TRIAGEM_09'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_10'])
df_reg['PERGUNTA_TRIAGEM_10'].count()
df_reg['PERGUNTA_TRIAGEM_10'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_11'])
df_reg['PERGUNTA_TRIAGEM_11'].count()
df_reg['PERGUNTA_TRIAGEM_11'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_12'])
df_reg['PERGUNTA_TRIAGEM_12'].count()
df_reg['PERGUNTA_TRIAGEM_12'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_13'])
df_reg['PERGUNTA_TRIAGEM_13'].count()
df_reg['PERGUNTA_TRIAGEM_13'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_14'])
df_reg['PERGUNTA_TRIAGEM_14'].count()
df_reg['PERGUNTA_TRIAGEM_14'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_15'])
df_reg['PERGUNTA_TRIAGEM_15'].count()
df_reg['PERGUNTA_TRIAGEM_15'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_16'])
df_reg['PERGUNTA_TRIAGEM_16'].count()
df_reg['PERGUNTA_TRIAGEM_16'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_17'])
df_reg['PERGUNTA_TRIAGEM_17'].count()
df_reg['PERGUNTA_TRIAGEM_17'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_18'])
df_reg['PERGUNTA_TRIAGEM_18'].count()
df_reg['PERGUNTA_TRIAGEM_18'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_19'])
df_reg['PERGUNTA_TRIAGEM_19'].count()
df_reg['PERGUNTA_TRIAGEM_19'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_20'])
df_reg['PERGUNTA_TRIAGEM_20'].count()
df_reg['PERGUNTA_TRIAGEM_20'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_21'])
df_reg['PERGUNTA_TRIAGEM_21'].count()
df_reg['PERGUNTA_TRIAGEM_21'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_22'])
df_reg['PERGUNTA_TRIAGEM_22'].count()
df_reg['PERGUNTA_TRIAGEM_22'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_23'])
df_reg['PERGUNTA_TRIAGEM_23'].count()
df_reg['PERGUNTA_TRIAGEM_23'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_24'])
df_reg['PERGUNTA_TRIAGEM_24'].count()
df_reg['PERGUNTA_TRIAGEM_24'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_25'])
df_reg['PERGUNTA_TRIAGEM_25'].count()
df_reg['PERGUNTA_TRIAGEM_25'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_26'])
df_reg['PERGUNTA_TRIAGEM_26'].count()
df_reg['PERGUNTA_TRIAGEM_26'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_27'])
df_reg['PERGUNTA_TRIAGEM_27'].count()
df_reg['PERGUNTA_TRIAGEM_27'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_28'])
df_reg['PERGUNTA_TRIAGEM_28'].count()
df_reg['PERGUNTA_TRIAGEM_28'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_29'])
df_reg['PERGUNTA_TRIAGEM_29'].count()
df_reg['PERGUNTA_TRIAGEM_29'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_30'])
df_reg['PERGUNTA_TRIAGEM_30'].count()
df_reg['PERGUNTA_TRIAGEM_30'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_31'])
df_reg['PERGUNTA_TRIAGEM_31'].count()
df_reg['PERGUNTA_TRIAGEM_31'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_32'])
df_reg['PERGUNTA_TRIAGEM_32'].count()
df_reg['PERGUNTA_TRIAGEM_32'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_TRIAGEM_33'])
df_reg['PERGUNTA_TRIAGEM_33'].count()
df_reg['PERGUNTA_TRIAGEM_33'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_02'])
df_reg['PERGUNTA_FICHA_EME_PA_02'].count()
df_reg['PERGUNTA_FICHA_EME_PA_02'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_03'])
df_reg['PERGUNTA_FICHA_EME_PA_03'].count()
df_reg['PERGUNTA_FICHA_EME_PA_03'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_04'])
df_reg['PERGUNTA_FICHA_EME_PA_04'].count()
df_reg['PERGUNTA_FICHA_EME_PA_04'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_05'])
df_reg['PERGUNTA_FICHA_EME_PA_05'].count()
df_reg['PERGUNTA_FICHA_EME_PA_05'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_06'])
df_reg['PERGUNTA_FICHA_EME_PA_06'].count()
df_reg['PERGUNTA_FICHA_EME_PA_06'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_07'])
df_reg['PERGUNTA_FICHA_EME_PA_07'].count()
df_reg['PERGUNTA_FICHA_EME_PA_07'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_08'])
df_reg['PERGUNTA_FICHA_EME_PA_08'].count()
df_reg['PERGUNTA_FICHA_EME_PA_08'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_09'])
df_reg['PERGUNTA_FICHA_EME_PA_09'].count()
df_reg['PERGUNTA_FICHA_EME_PA_09'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_10'])
df_reg['PERGUNTA_FICHA_EME_PA_10'].count()
df_reg['PERGUNTA_FICHA_EME_PA_10'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_11'])
df_reg['PERGUNTA_FICHA_EME_PA_11'].count()
df_reg['PERGUNTA_FICHA_EME_PA_11'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_12'])
df_reg['PERGUNTA_FICHA_EME_PA_12'].count()
df_reg['PERGUNTA_FICHA_EME_PA_12'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_13'])
df_reg['PERGUNTA_FICHA_EME_PA_13'].count()
df_reg['PERGUNTA_FICHA_EME_PA_13'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_14'])
df_reg['PERGUNTA_FICHA_EME_PA_14'].count()
df_reg['PERGUNTA_FICHA_EME_PA_14'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_15'])
df_reg['PERGUNTA_FICHA_EME_PA_15'].count()
df_reg['PERGUNTA_FICHA_EME_PA_15'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_16'])
df_reg['PERGUNTA_FICHA_EME_PA_16'].count()
df_reg['PERGUNTA_FICHA_EME_PA_16'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_17'])
df_reg['PERGUNTA_FICHA_EME_PA_17'].count()
df_reg['PERGUNTA_FICHA_EME_PA_17'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_18'])
df_reg['PERGUNTA_FICHA_EME_PA_18'].count()
df_reg['PERGUNTA_FICHA_EME_PA_18'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_19'])
df_reg['PERGUNTA_FICHA_EME_PA_19'].count()
df_reg['PERGUNTA_FICHA_EME_PA_19'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_20'])
df_reg['PERGUNTA_FICHA_EME_PA_20'].count()
df_reg['PERGUNTA_FICHA_EME_PA_20'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_21'])
df_reg['PERGUNTA_FICHA_EME_PA_21'].count()
df_reg['PERGUNTA_FICHA_EME_PA_21'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_22'])
df_reg['PERGUNTA_FICHA_EME_PA_22'].count()
df_reg['PERGUNTA_FICHA_EME_PA_22'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_23'])
df_reg['PERGUNTA_FICHA_EME_PA_23'].count()
df_reg['PERGUNTA_FICHA_EME_PA_23'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_24'])
df_reg['PERGUNTA_FICHA_EME_PA_24'].count()
df_reg['PERGUNTA_FICHA_EME_PA_24'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_25'])
df_reg['PERGUNTA_FICHA_EME_PA_25'].count()
df_reg['PERGUNTA_FICHA_EME_PA_25'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_FICHA_EME_PA_26'])
df_reg['PERGUNTA_FICHA_EME_PA_26'].count()
df_reg['PERGUNTA_FICHA_EME_PA_26'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['COD_CID_ADM'])
df_reg['COD_CID_ADM'].count()
df_reg['COD_CID_ADM'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['COD_CID_ADM_PRINCIPAL'])
df_reg['COD_CID_ADM_PRINCIPAL'].count()
df_reg['COD_CID_ADM_PRINCIPAL'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_01'])
df_reg['PERGUNTA_CONSULTA_P_01'].count()
df_reg['PERGUNTA_CONSULTA_P_01'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_02'])
df_reg['PERGUNTA_CONSULTA_P_02'].count()
df_reg['PERGUNTA_CONSULTA_P_02'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_03'])
df_reg['PERGUNTA_CONSULTA_P_03'].count()
df_reg['PERGUNTA_CONSULTA_P_03'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_04'])
df_reg['PERGUNTA_CONSULTA_P_04'].count()
df_reg['PERGUNTA_CONSULTA_P_04'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_05'])
df_reg['PERGUNTA_CONSULTA_P_05'].count()
df_reg['PERGUNTA_CONSULTA_P_05'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_06'])
df_reg['PERGUNTA_CONSULTA_P_06'].count()
df_reg['PERGUNTA_CONSULTA_P_06'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_08'])
df_reg['PERGUNTA_CONSULTA_P_08'].count()
df_reg['PERGUNTA_CONSULTA_P_08'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_09'])
df_reg['PERGUNTA_CONSULTA_P_09'].count()
df_reg['PERGUNTA_CONSULTA_P_09'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_10'])
df_reg['PERGUNTA_CONSULTA_P_10'].count()
df_reg['PERGUNTA_CONSULTA_P_10'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_11'])
df_reg['PERGUNTA_CONSULTA_P_11'].count()
df_reg['PERGUNTA_CONSULTA_P_11'].fillna(-1, inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_12'])
df_reg['PERGUNTA_CONSULTA_P_12'].count()
df_reg['PERGUNTA_CONSULTA_P_12'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_14'])
df_reg['PERGUNTA_CONSULTA_P_14'].count()
df_reg['PERGUNTA_CONSULTA_P_14'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_15'])
df_reg['PERGUNTA_CONSULTA_P_15'].count()
df_reg['PERGUNTA_CONSULTA_P_15'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['PERGUNTA_CONSULTA_P_16'])
df_reg['PERGUNTA_CONSULTA_P_16'].count()
df_reg['PERGUNTA_CONSULTA_P_16'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['INDICADOR_IAM_CSST'])
df_reg['INDICADOR_IAM_CSST'].count()
df_reg['INDICADOR_IAM_CSST'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_reg['CSV_MIN_TEMPERATURA'])
df_reg['CSV_MIN_TEMPERATURA'].count()
df_reg['CSV_MIN_TEMPERATURA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MAX_TEMPERATURA'])
df_reg['CSV_MAX_TEMPERATURA'].count()
df_reg['CSV_MAX_TEMPERATURA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MEDIA_TEMPERATURA'])
df_reg['CSV_MEDIA_TEMPERATURA'].count()
df_reg['CSV_MEDIA_TEMPERATURA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MIN_TA_SISTOLICA'])
df_reg['CSV_MIN_TA_SISTOLICA'].count()
df_reg['CSV_MIN_TA_SISTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MAX_TA_SISTOLICA'])
df_reg['CSV_MAX_TA_SISTOLICA'].count()
df_reg['CSV_MAX_TA_SISTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MEDIA_TA_SISTOLICA'])
df_reg['CSV_MEDIA_TA_SISTOLICA'].count()
df_reg['CSV_MEDIA_TA_SISTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MIN_TA_DIASTOLICA'])
df_reg['CSV_MIN_TA_DIASTOLICA'].count()
df_reg['CSV_MIN_TA_DIASTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MAX_TA_DIASTOLICA'])
df_reg['CSV_MAX_TA_DIASTOLICA'].count()
df_reg['CSV_MAX_TA_DIASTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MEDIA_TA_DIASTOLICA'])
df_reg['CSV_MEDIA_TA_DIASTOLICA'].count()
df_reg['CSV_MEDIA_TA_DIASTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MIN_FREQ_RESPIRATORIA'])
df_reg['CSV_MIN_FREQ_RESPIRATORIA'].count()
df_reg['CSV_MIN_FREQ_RESPIRATORIA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MAX_FREQ_RESPIRATORIA'])
df_reg['CSV_MAX_FREQ_RESPIRATORIA'].count()
df_reg['CSV_MAX_FREQ_RESPIRATORIA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MEDIA_FREQ_RESPIRATORIA'])
df_reg['CSV_MEDIA_FREQ_RESPIRATORIA'].count()
df_reg['CSV_MEDIA_FREQ_RESPIRATORIA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MIN_GLICEMIA_DIGITAL'])
df_reg['CSV_MIN_GLICEMIA_DIGITAL'].count()
df_reg['CSV_MIN_GLICEMIA_DIGITAL'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MAX_GLICEMIA_DIGITAL'])
df_reg['CSV_MAX_GLICEMIA_DIGITAL'].count()
df_reg['CSV_MAX_GLICEMIA_DIGITAL'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MEDIA_GLICEMIA_DIGITAL'])
df_reg['CSV_MEDIA_GLICEMIA_DIGITAL'].count()
df_reg['CSV_MEDIA_GLICEMIA_DIGITAL'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MIN_SATURACAO_O2'])
df_reg['CSV_MIN_SATURACAO_O2'].count()
df_reg['CSV_MIN_SATURACAO_O2'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MAX_SATURACAO_O2'])
df_reg['CSV_MAX_SATURACAO_O2'].count()
df_reg['CSV_MAX_SATURACAO_O2'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MEDIA_SATURACAO_O2'])
df_reg['CSV_MEDIA_SATURACAO_O2'].count()
df_reg['CSV_MEDIA_SATURACAO_O2'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MIN_PRESSAO_ARTERIAL'])
df_reg['CSV_MIN_PRESSAO_ARTERIAL'].count()
df_reg['CSV_MIN_PRESSAO_ARTERIAL'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MAX_PRESSAO_ARTERIAL'])
df_reg['CSV_MAX_PRESSAO_ARTERIAL'].count()
df_reg['CSV_MAX_PRESSAO_ARTERIAL'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MEDIA_PRESSAO_ARTERIAL'])
df_reg['CSV_MEDIA_PRESSAO_ARTERIAL'].count()
df_reg['CSV_MEDIA_PRESSAO_ARTERIAL'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MIN_FREQ_CARDIACA'])
df_reg['CSV_MIN_FREQ_CARDIACA'].count()
df_reg['CSV_MIN_FREQ_CARDIACA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MAX_FREQ_CARDIACA'])
df_reg['CSV_MAX_FREQ_CARDIACA'].count()
df_reg['CSV_MAX_FREQ_CARDIACA'].fillna(-1, inplace = True)

pd.value_counts(df_reg['CSV_MEDIA_FREQ_CARDIACA'])
df_reg['CSV_MEDIA_FREQ_CARDIACA'].count()
df_reg['CSV_MEDIA_FREQ_CARDIACA'].fillna(-1, inplace = True)

#Verificação dos Valores Nulos Ainda Existentes

for col in df_reg.columns:
    if df_reg[col].isnull().sum() > 0:
        print(col)

#Exclusão das Colunas Que Não Serão Utilizadas

df_reg_trat = df_reg.drop('DESC_DIA_DA_SEMANA', axis = 1)

df_reg_trat = df_reg_trat.drop('REGISTRO', axis = 1)

df_reg_trat = df_reg_trat.drop('NOME_PACIENTE', axis = 1)

df_reg_trat = df_reg_trat.drop('DESC_SEXO', axis = 1)

df_reg_trat = df_reg_trat.drop('DESC_FAIXA_ETARIA', axis = 1)

df_reg_trat = df_reg_trat.drop('NUM_TRATAMENTO', axis = 1)

df_reg_trat = df_reg_trat.drop('DATA_ADMISSAO', axis = 1)

df_reg_trat = df_reg_trat.drop('DESC_DATA_ADMISSAO', axis = 1)

df_reg_trat = df_reg_trat.drop('DESC_CONVENIO', axis = 1)

df_reg_trat = df_reg_trat.drop('COD_EMPRESA', axis = 1)

df_reg_trat = df_reg_trat.drop('DESC_EMPRESA', axis = 1)

df_reg_trat = df_reg_trat.drop('COD_PLANTONISTA', axis = 1)

df_reg_trat = df_reg_trat.drop('NOME_PLANTONISTA', axis = 1)

df_reg_trat = df_reg_trat.drop('PERGUNTA_TRIAGEM_01', axis = 1)

df_reg_trat = df_reg_trat.drop('PERGUNTA_FICHA_EME_PA_01', axis = 1)

df_reg_trat = df_reg_trat.drop('PERGUNTA_CONSULTA_P_07', axis = 1)

df_reg_trat = df_reg_trat.drop('PERGUNTA_CONSULTA_P_13', axis = 1)

df_reg_trat = df_reg_trat.drop('DESC_DATA_ALTA', axis = 1)

#df_reg_trat = df_reg_trat.drop('TEMPO_ATENDIMENTO', axis = 1)

df_reg_trat = df_reg_trat.drop('DESC_STATUS_INT', axis = 1)

df_reg_trat = df_reg_trat.drop('CONT_STATUS_INT', axis = 1)

df_reg_trat = df_reg_trat.drop('NUM_INT_POSTERIOR', axis = 1)

df_reg_trat = df_reg_trat.drop('CONTADOR', axis = 1)

#Verificação dos Valores Nulos Ainda Existentes

for col in df_reg_trat.columns:
    if df_reg_trat[col].isnull().sum() > 0:
        print(col)
        
#Estatísticas Descritivas da Base de Dados - Após Tratamento dos Dados
        
df_reg_trat.info([''])

#Criação das Bases X_reg e Y_reg 

X_reg = df_reg_trat

Y_reg = df_reg_trat['TEMPO_PERM_INT_POSTERIOR']

X_reg = X_reg.drop('TEMPO_PERM_INT_POSTERIOR', axis = 1)

#Estatísticas Descritivas da Base de Dados - X_reg - Após Remoção da Coluna Target

X_reg.info([''])

#Aplicação do Label Encoder na Base X_reg

from sklearn.preprocessing import LabelEncoder

lbl_enc_X_reg = LabelEncoder()

X_reg['DESC_BAIRRO'] = lbl_enc_X_reg.fit_transform(X_reg['DESC_BAIRRO'])

X_reg['COD_CID'] = lbl_enc_X_reg.fit_transform(X_reg['COD_CID'])

X_reg['DESC_GRUPO_SANGUINEO'] = lbl_enc_X_reg.fit_transform(X_reg['DESC_GRUPO_SANGUINEO'])

X_reg['APEL_PLANTONISTA'] = lbl_enc_X_reg.fit_transform(X_reg['APEL_PLANTONISTA'])

X_reg['CID_PREVALENTE'] = lbl_enc_X_reg.fit_transform(X_reg['CID_PREVALENTE'])

X_reg['COD_CLAC_RISCO'] = lbl_enc_X_reg.fit_transform(X_reg['COD_CLAC_RISCO'])

X_reg['PERGUNTA_TRIAGEM_02'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_02'])

X_reg['PERGUNTA_TRIAGEM_03'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_03'])

X_reg['PERGUNTA_TRIAGEM_04'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_04'])

X_reg['PERGUNTA_TRIAGEM_05'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_05'])

X_reg['PERGUNTA_TRIAGEM_06'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_06'])

X_reg['PERGUNTA_TRIAGEM_07'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_07'])

X_reg['PERGUNTA_TRIAGEM_08'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_08'])

X_reg['PERGUNTA_TRIAGEM_09'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_09'])

X_reg['PERGUNTA_TRIAGEM_10'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_10'])

X_reg['PERGUNTA_TRIAGEM_11'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_11'])

X_reg['PERGUNTA_TRIAGEM_12'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_12'])

X_reg['PERGUNTA_TRIAGEM_13'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_13'])

X_reg['PERGUNTA_TRIAGEM_14'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_14'])

X_reg['PERGUNTA_TRIAGEM_15'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_15'])

X_reg['PERGUNTA_TRIAGEM_16'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_16'])

X_reg['PERGUNTA_TRIAGEM_24'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_24'])

X_reg['PERGUNTA_TRIAGEM_25'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_25'])

X_reg['PERGUNTA_TRIAGEM_26'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_26'])

X_reg['PERGUNTA_TRIAGEM_27'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_27'])

X_reg['PERGUNTA_TRIAGEM_28'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_28'])

X_reg['PERGUNTA_TRIAGEM_29'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_29'])

X_reg['PERGUNTA_TRIAGEM_30'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_30'])

X_reg['PERGUNTA_TRIAGEM_31'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_31'])

X_reg['PERGUNTA_TRIAGEM_32'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_32'])

X_reg['PERGUNTA_TRIAGEM_33'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_TRIAGEM_33'])

X_reg['PERGUNTA_FICHA_EME_PA_02'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_02'])

X_reg['PERGUNTA_FICHA_EME_PA_03'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_03'])

X_reg['PERGUNTA_FICHA_EME_PA_04'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_04'])

X_reg['PERGUNTA_FICHA_EME_PA_05'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_05'])

X_reg['PERGUNTA_FICHA_EME_PA_06'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_06'])

X_reg['PERGUNTA_FICHA_EME_PA_07'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_07'])

X_reg['PERGUNTA_FICHA_EME_PA_08'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_08'])

X_reg['PERGUNTA_FICHA_EME_PA_09'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_09'])

X_reg['PERGUNTA_FICHA_EME_PA_10'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_10'])

X_reg['PERGUNTA_FICHA_EME_PA_11'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_11'])

X_reg['PERGUNTA_FICHA_EME_PA_12'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_12'])

X_reg['PERGUNTA_FICHA_EME_PA_13'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_13'])

X_reg['PERGUNTA_FICHA_EME_PA_14'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_14'])

X_reg['PERGUNTA_FICHA_EME_PA_15'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_15'])

X_reg['PERGUNTA_FICHA_EME_PA_16'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_16'])

X_reg['PERGUNTA_FICHA_EME_PA_17'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_17'])

X_reg['PERGUNTA_FICHA_EME_PA_18'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_18'])

X_reg['PERGUNTA_FICHA_EME_PA_19'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_19'])

X_reg['PERGUNTA_FICHA_EME_PA_20'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_20'])

X_reg['PERGUNTA_FICHA_EME_PA_21'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_21'])

X_reg['PERGUNTA_FICHA_EME_PA_22'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_22'])

X_reg['PERGUNTA_FICHA_EME_PA_23'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_23'])

X_reg['PERGUNTA_FICHA_EME_PA_24'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_24'])

X_reg['PERGUNTA_FICHA_EME_PA_25'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_25'])

X_reg['PERGUNTA_FICHA_EME_PA_26'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_FICHA_EME_PA_26'])

X_reg['COD_CID_ADM'] = lbl_enc_X_reg.fit_transform(X_reg['COD_CID_ADM'])

X_reg['COD_CID_ADM_PRINCIPAL'] = lbl_enc_X_reg.fit_transform(X_reg['COD_CID_ADM_PRINCIPAL'])

X_reg['PERGUNTA_CONSULTA_P_09'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_CONSULTA_P_09'])

X_reg['PERGUNTA_CONSULTA_P_10'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_CONSULTA_P_10'])

X_reg['PERGUNTA_CONSULTA_P_12'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_CONSULTA_P_12'])

X_reg['PERGUNTA_CONSULTA_P_14'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_CONSULTA_P_14'])

X_reg['PERGUNTA_CONSULTA_P_15'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_CONSULTA_P_15'])

X_reg['PERGUNTA_CONSULTA_P_16'] = lbl_enc_X_reg.fit_transform(X_reg['PERGUNTA_CONSULTA_P_16'])

X_reg['INDICADOR_IAM_CSST'] = lbl_enc_X_reg.fit_transform(X_reg['INDICADOR_IAM_CSST'])

#Estatísticas Descritivas da Base de Dados - X_reg - Após Aplicação do Label Encoder

X_reg.info([''])

#Transformação da Base Y_reg em Array

import numpy as np

Y_reg = np.array(Y_reg)

#Transformação da Base X_reg em Array

X_list_reg = list(X_reg.columns)

X_reg = np.array(X_reg)

#Escalonamento da Base X_reg - Opção 01 - StandardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_reg = scaler.fit_transform(X_reg)

#Escalonamento da Base X_reg - Opção 02 - MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X_reg = scaler.fit_transform(X_reg)

#Escalonamento da Base X_reg - Opção 03 - Normalizer

from sklearn.preprocessing import Normalizer

scaler = Normalizer()

X_reg = scaler.fit_transform(X_reg)

#Escalonamento da Base X_reg - Opção 04 - RobustScaler

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_reg = scaler.fit_transform(X_reg)

#ETAPA 02: DIVISÃO DA BASE DE DADOS EM TESTE E TREINAMENTO

#Criação da Base de Teste e Treinamento

from sklearn.model_selection import train_test_split

X_train_reg, X_test_reg, Y_train_reg, Y_test_reg = train_test_split(X_reg, Y_reg, test_size = 0.3, random_state = seed_reg)

print('Training X_reg Data Dimensions:', X_train_reg.shape)
print('Training Y_reg Data Dimensions:', Y_train_reg.shape)
print('Testing X_reg Data Dimensions:', X_test_reg.shape)
print('Testing Y_reg Data Dimensions:', Y_test_reg.shape)

#ETAPA 03: TESTE DE VÁRIOS MODELOS

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from pyearth import Earth
from rgf.sklearn import RGFRegressor
from rgf.sklearn import FastRGFRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor

from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

import matplotlib.pyplot as plt
import seaborn as sns

folds_reg   = 10
metric_reg  = "neg_mean_squared_error"

# Criação do Dicionário de Modelos

models_reg = {}
models_reg["LinearR"]          = LinearRegression()
models_reg["Lasso"]            = Lasso(random_state = seed_reg)
models_reg["LassoLarsIC"]      = LassoLarsIC()
models_reg["Ridge"]            = Ridge(random_state = seed_reg)
models_reg["KernelRidge"]      = KernelRidge()
models_reg["BayesianRidge"]    = BayesianRidge()
models_reg["ElasticNet"]       = ElasticNet(random_state = seed_reg)
models_reg["KNN"]              = KNeighborsRegressor()
models_reg["SVR"]              = SVR()
models_reg["DecisionTree"]     = DecisionTreeRegressor(random_state = seed_reg)
models_reg["ExtraTrees"]       = ExtraTreesRegressor(random_state = seed_reg)
models_reg["Earth"]            = Earth()
models_reg["RGFRegressor"]     = RGFRegressor()
models_reg["FastRGFRegressor"] = FastRGFRegressor()
models_reg["RandomForest"]     = RandomForestRegressor(random_state = seed_reg)
models_reg["AdaBoost"]         = AdaBoostRegressor(random_state = seed_reg)
models_reg["GradientBoost"]    = GradientBoostingRegressor(random_state = seed_reg)
models_reg["XGBoost"]          = XGBRegressor(random_state = seed_reg)
models_reg["LightGBM"]         = LGBMRegressor(random_state = seed_reg)
models_reg["CatBoost"]         = CatBoostRegressor(random_state = seed_reg)
models_reg["MLPRegressor"]     = MLPRegressor(random_state = seed_reg)

#Treino e Teste dos Modelos com CrossValScore + 10 KFold - X_train_reg, Y_train_reg

model_results_reg = []
model_std_reg = []
model_names_reg   = []
for model_name_reg in models_reg:
	model_reg   = models_reg[model_name_reg]
	k_fold_reg  = KFold(n_splits = folds_reg, random_state = seed_reg, shuffle = 'True')
	results_reg = np.sqrt(-cross_val_score(model_reg, X_train_reg, Y_train_reg, cv = k_fold_reg, scoring = metric_reg))
    
	score_reg = round(results_reg.mean(),3)
	std_reg = round(results_reg.std(),3)
	model_results_reg.append(score_reg)
	model_std_reg.append(std_reg)
	model_names_reg.append(model_name_reg)
	print("{}: {}, {}".format(model_reg_name, score_reg, std_reg))
    
kf_cross_val_reg = pd.DataFrame({'Name': model_reg_names, 'RMSE': model_reg_results, 'Desvio Padrão': model_reg_std})
print(kf_cross_val_reg)

#Plotagem das Acurácias com CrossValScore + 10 KFold - X_train_reg, Y_train_reg

axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val_reg)
axis.set(xlabel = 'Regressor', ylabel = 'RMSE')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha = "center") 
    
plt.figure(figsize = (14, 7))     
plt.show()

# Comparação dos Modelos com CrossValScore + 10 KFold - X_train_reg, Y_train_reg - Gráfico de BoxPlot

#plt.figure(figsize = (14, 7)) 
figure_reg = plt.figure(figsize = (14, 7))
figure_reg.suptitle('Regression Models Comparison')
axis = figure_reg.add_subplot(111)
plt.boxplot(model_reg_results)
axis.set_xticklabels(model_reg_names, rotation = 45, ha = "right")
axis.set_ylabel("Mean Squared Error (MSE)")
plt.margins(0.05, 0.1)
plt.show()

#Treino e Teste dos Modelos com CrossValidate + 10 KFold - X_train_reg, Y_train_reg

from sklearn.metrics import explained_variance_score
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

from sklearn.model_selection import cross_validate

from sklearn.model_selection import KFold

metricas_cross_validate_reg = ['explained_variance', 
                               'neg_mean_absolute_error',
                               'neg_median_absolute_error',                               
                               'neg_mean_squared_error',
                               'r2',
                               'max_error']

lst_names_reg = []
lst_fit_time_reg = []
lst_score_time_reg = []
lst_test_explained_variance_reg = []
lst_test_neg_mean_absolute_error_reg = []
lst_test_neg_median_absolute_error_reg = []
lst_test_neg_mean_squared_error_reg = []
lst_test_root_mean_squared_error_reg = []
lst_test_r2_reg = []
lst_test_max_error_reg = []

import time

start = time.time()
for model_name_reg in models_reg:
    model_reg = models_reg[model_name_reg]
    kfold_reg = KFold(n_splits = 10, random_state = seed_reg, shuffle = True ) 
    result_reg = cross_validate( estimator = model_reg, 
                                 X = X_train_reg, 
                                 y = Y_train_reg, 
                                 cv = kfold_reg, 
                                 scoring = metricas_cross_validate_reg, 
                                 return_train_score = False )
    var_name_reg = model_name_reg
    var_fit_time_reg   = result_reg['fit_time'].mean()
    var_score_time_reg = result_reg['score_time'].mean()
    var_test_explained_variance_reg = result_reg['test_explained_variance'].mean()
    var_test_neg_mean_absolute_error_reg = result_reg['test_neg_mean_absolute_error'].mean()
    var_test_neg_median_absolute_error_reg = result_reg['test_neg_median_absolute_error'].mean()    
    var_test_neg_mean_squared_error_reg = result_reg['test_neg_mean_squared_error'].mean()
    var_mean_squared_error_reg = result_reg['test_neg_mean_squared_error']*-1
    var_test_root_mean_squared_error_reg = np.sqrt(var_mean_squared_error_reg).mean()
    var_test_r2_reg = result_reg['test_r2'].mean()
    var_test_max_error_reg = result_reg['test_max_error'].mean()
    lst_names_reg.append(var_name_reg)
    lst_fit_time_reg.append(var_fit_time_reg)
    lst_score_time_reg.append(var_score_time_reg)
    lst_test_explained_variance_reg.append(var_test_explained_variance_reg)
    lst_test_neg_mean_absolute_error_reg.append(var_test_neg_mean_absolute_error_reg*-1)
    lst_test_neg_median_absolute_error_reg.append(var_test_neg_median_absolute_error_reg*-1)    
    lst_test_neg_mean_squared_error_reg.append(var_test_neg_mean_squared_error_reg*-1)
    lst_test_root_mean_squared_error_reg.append(var_test_root_mean_squared_error_reg)
    lst_test_r2_reg.append(var_test_r2_reg)
    lst_test_max_error_reg.append(var_test_max_error_reg)
end = time.time()    
kf_cross_validate_reg = pd.DataFrame({'Name': lst_names_reg, 
                                      'Fit Time': lst_fit_time_reg, 
                                      'Score Time': lst_score_time_reg, 
                                      'Test Explained Variance': lst_test_explained_variance_reg, 
                                      'Test Mean Absolute Error (MAE)': lst_test_neg_mean_absolute_error_reg, 
                                      'Test Median Absolute Error': lst_test_neg_median_absolute_error_reg,                                      
                                      'Test Mean Squared Error (MSE)': lst_test_neg_mean_squared_error_reg, 
                                      'Test Root Mean Squared Error (RMSE)': lst_test_root_mean_squared_error_reg,
                                      'Test R2 Score': lst_test_r2_reg,
                                      'Test Max Error': lst_test_max_error_reg})

print("Tempo de Execução: {:.2f} sec".format(end - start))    
print(kf_cross_validate_reg)

#EXPLAINED VARIANCE
#Best possible score is 1.0, lower values are worse.

#MEAN ABSOLUTE ERROR (MAE)
#The mean_absolute_error function computes mean absolute error, a risk metric corresponding to the expected value of the absolute error loss or -norm loss.

#MEAN SQUARED ERROR (MSE)
#The mean_squared_error function computes mean square error, a risk metric corresponding to the expected value of the squared (quadratic) error or loss.

#MEDIAN ABSOLUTE ERROR
#The median_absolute_error is particularly interesting because it is robust to outliers. 
#The loss is calculated by taking the median of all absolute differences between the target and the prediction.
#MAE output is non-negative floating point. The best value is 0.0.

#ROOT MEAN SQUARED ERROR (RMSE)
#Root Mean Square Error (RMSE) is the standard deviation of the residuals (prediction errors). 
#Residuals are a measure of how far from the regression line data points are; RMSE is a measure of how spread out these residuals are. 
#In other words, it tells you how concentrated the data is around the line of best fit.

#R² SCORE
#The r2_score function computes the coefficient of determination, usually denoted as R².
#It represents the proportion of variance (of y) that has been explained by the independent variables in the model. 
#It provides an indication of goodness of fit and therefore a measure of how well unseen samples are likely to be predicted by the model, through the proportion of explained variance.
#As such variance is dataset dependent, R² may not be meaningfully comparable across different datasets. 
#Best possible score is 1.0 and it can be negative (because the model can be arbitrarily worse). 
#A constant model that always predicts the expected value of y, disregarding the input features, would get a R² score of 0.0.

#MAX ERROR
#The max_error function computes the maximum residual error , a metric that captures the worst case error between the predicted value and the true value. 
#In a perfectly fitted single output regression model, max_error would be 0 on the training set and though this would be highly unlikely in the real world, this metric shows the extent of error that the model had when it was fitted.

#ETAPA 04: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO REGRESSÃO LINEAR

from sklearn.linear_model import LinearRegression

regressor_linear = LinearRegression()

regressor_linear.fit(X_train_reg, Y_train_reg)

#ETAPA 05: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO REGRESSÃO POLINOMIAL

#Fit/Transform das Bases X_train_reg e X_test_reg
from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 2)

X_train_poly_reg = poly.fit_transform(X_train_reg)

X_test_poly_reg = poly.fit_transform(X_test_reg)

#Fit e Predição
regressor_poly = LinearRegression()

regressor_poly.fit(X_train_poly_reg, Y_train_reg)

Y_pred_train_reg_poly = regressor_poly.predict(X_train_poly_reg)

Y_pred_test_reg_poly = regressor_poly.predict(X_test_poly_reg)

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_poly = explained_variance_score(Y_test_reg, Y_pred_test_reg_poly)
mtrc_mae_poly = mean_absolute_error(Y_test_reg, Y_pred_test_reg_poly)
mtrc_mdae_poly = median_absolute_error(Y_test_reg, Y_pred_test_reg_poly)
mtrc_mse_poly = mean_squared_error(Y_test_reg, Y_pred_test_reg_poly)
mtrc_rmse_poly = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_poly))
#mtrc_msle_poly = mean_squared_log_error(Y_test_reg, Y_pred_test_reg_poly)
mtrc_r2_poly = r2_score(Y_test_reg, Y_pred_test_reg_poly)
mtrc_max_error_poly = max_error(Y_test_reg, Y_pred_test_reg_poly)

print('Explained Variance Score:', mtrc_evs_poly)
print('Mean Absolute Error:', mtrc_mae_poly)
print('Median Absolute Error:', mtrc_mdae_poly)
print('Mean Squared Error:', mtrc_mse_poly)  
print('Root Mean Squared Error:', mtrc_rmse_poly)
#print('Mean Squared Logarithmic Error:', mtrc_msle_poly)
print('R² Score:', mtrc_r2_poly)
print('Max Error:', mtrc_max_error_poly)

print('Explained Variance Score: {:.4f}'.format(mtrc_evs_poly))
print('Mean Absolute Error: {:.4f}'.format(mtrc_mae_poly)) 
print('Median Absolute Error: {:.4f}'.format(mtrc_mdae_poly)) 
print('Mean Squared Error: {:.4f}'.format(mtrc_mse_poly))
print('Root Mean Squared Error: {:.4f}'.format(mtrc_rmse_poly))
#print('Mean Squared Logarithmic Error: {:.4f}'.format(mtrc_msle_poly))
print('R² Score: {:.4f}'.format(mtrc_r2_poly))
print('Max Error: {:.4f}'.format(mtrc_max_error_poly))

#LOG
#Explained Variance Score: -0.0361
#Mean Absolute Error: 0.6378
#Median Absolute Error: 0.5046
#Mean Squared Error: 0.7220
#Root Mean Squared Error: 0.8497
#R² Score: -0.0379
#Max Error: 8.3000

#Plot - Y_test_reg vs Y_pred_test_reg_poly

#Ideally Should Have Been a Straight Line
plt.figure(figsize = (14, 7)) 
plt.scatter(Y_test_reg, Y_pred_test_reg_poly)
plt.show()

#Plot Residuals

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_poly, Y_pred_train_reg_poly - Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_poly, Y_pred_test_reg_poly - Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("Polynomial Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

#Plot Predictions

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_poly, Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_poly, Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("Polynomial Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Real Values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

#ETAPA 06: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO LASSO

from sklearn.linear_model import Lasso

regressor_lasso = Lasso(alpha = 0.0005, random_state = seed_reg)

#ETAPA 07: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO LASSOLARSIC

from sklearn.linear_model import LassoLarsIC

regressor_llic = LassoLarsIC(criterion = 'bic')

regressor_llic = LassoLarsIC(criterion = 'aic')

regressor_llic.fit(X_train_reg, Y_train_reg)

Y_pred_train_reg_llic = regressor_llic.predict(X_train_reg)

Y_pred_test_reg_llic = regressor_llic.predict(X_test_reg)

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_llic = explained_variance_score(Y_test_reg, Y_pred_test_reg_llic)
mtrc_mae_llic = mean_absolute_error(Y_test_reg, Y_pred_test_reg_llic)
mtrc_mdae_llic = median_absolute_error(Y_test_reg, Y_pred_test_reg_llic)
mtrc_mse_llic = mean_squared_error(Y_test_reg, Y_pred_test_reg_llic)
mtrc_rmse_llic = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_llic))
#mtrc_msle_llic = mean_squared_log_error(Y_test_reg, Y_pred_test_reg_llic)
mtrc_r2_llic = r2_score(Y_test_reg, Y_pred_test_reg_llic)
mtrc_max_error_llic = max_error(Y_test_reg, Y_pred_test_reg_llic)

print('Explained Variance Score:', mtrc_evs_llic)
print('Mean Absolute Error:', mtrc_mae_llic)
print('Median Absolute Error:', mtrc_mdae_llic)
print('Mean Squared Error:', mtrc_mse_llic)  
print('Root Mean Squared Error:', mtrc_rmse_llic)
print('Mean Squared Logarithmic Error:', mtrc_msle_llic)
print('R² Score:', mtrc_r2_llic)
print('Max Error:', mtrc_max_error_llic)

print('Explained Variance Score: {:.4f}'.format(mtrc_evs_llic))
print('Mean Absolute Error: {:.4f}'.format(mtrc_mae_llic)) 
print('Median Absolute Error: {:.4f}'.format(mtrc_mdae_llic)) 
print('Mean Squared Error: {:.4f}'.format(mtrc_mse_llic))
print('Root Mean Squared Error: {:.4f}'.format(mtrc_rmse_llic))
#print('Mean Squared Logarithmic Error: {:.4f}'.format(mtrc_msle_llic))
print('R² Score: {:.4f}'.format(mtrc_r2_llic))
print('Max Error: {:.4f}'.format(mtrc_max_error_llic))

#BIC
#Explained Variance Score: 0.1791
#Mean Absolute Error: 4.0601
#Median Absolute Error: 2.2751
#Mean Squared Error: 66.0540
#Root Mean Squared Error: 8.1274
#Mean Squared Logarithmic Error: 0.5755
#R² Score: 0.1775
#Max Error: 105.2851

#AIC
#Explained Variance Score: 0.2179
#Mean Absolute Error: 4.0191
#Median Absolute Error: 2.3375
#Mean Squared Error: 62.9472
#Root Mean Squared Error: 7.9339
#R² Score: 0.2162
#Max Error: 102.7636

#AIC - Log
#Explained Variance Score: 0.3356
#Mean Absolute Error: 0.5249
#Median Absolute Error: 0.4213
#Mean Squared Error: 0.4627
#Root Mean Squared Error: 0.6802
#R² Score: 0.3349
#Max Error: 2.9847

#Plot - Y_test_reg vs Y_pred_test_reg_llic

#Ideally Should Have Been a Straight Line
plt.figure(figsize = (14, 7)) 
plt.scatter(Y_test_reg, Y_pred_test_reg_llic)
plt.show()

#Plot Residuals

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_llic, Y_pred_train_reg_llic - Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_llic, Y_pred_test_reg_llic - Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("LassoLarsIC Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

#Plot Predictions

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_llic, Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_llic, Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("LassoLarsIC Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Real Values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

#ETAPA 08: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO RIDGE

from sklearn.linear_model import Ridge

regressor_ridge = Ridge()

#ETAPA 09: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO KERNELRIDGE

from sklearn.kernel_ridge import KernelRidge

regressor_kridge = KernelRidge(alpha = 0.6, kernel = 'polynomial', degree = 2, coef0 = 2.5, random_state = seed_reg)

#ETAPA 10: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO BAYESIANRIDGE

from sklearn.linear_model import BayesianRidge

regressor_bridge = BayesianRidge()

#ETAPA 11: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO ELASTICNET

from sklearn.linear_model import ElasticNet

regressor_elnet = ElasticNet(alpha = 0.0005, l1_ratio = .9, random_state = seed_reg)

#ETAPA 12: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO KNN

from sklearn.neighbors import KNeighborsRegressor

regressor_knn = KNeighborsRegressor()

#ETAPA 13: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO SVM

from sklearn.svm import SVR

regressor_svr = SVR()

#ETAPA 14: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO DECISION TREE

from sklearn.tree import DecisionTreeRegressor

regressor_dtr = DecisionTreeRegressor()

#ETAPA 15: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO EXTRA TREES

from sklearn.ensemble import ExtraTreesRegressor

regressor_etr = ExtraTreesRegressor()

#ETAPA 16: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO MARS

from pyearth import Earth

regressor_mars = Earth( max_degree = 1, 
                        penalty = 1.0, 
                        endspan = 5 )

#ETAPA 17: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO RGFREGRESSOR

from rgf.sklearn import RGFRegressor

regressor_rgf = RGFRegressor( max_leaf = 300,
                              algorithm = "RGF_Sib",
                              test_interval = 100,
                              loss = "LS",
                              verbose = False )

regressor_rgf = RGFRegressor( max_leaf = 300,
                             test_interval = 100,
                             algorithm = 'RGF_Sib',
                             loss = 'Log',
                             reg_depth = 1.1,
                             l2 = 0.1,
                             sl2 = None,
                             normalize = False,
                             min_samples_leaf = 0.4,
                             n_iter = None,
                             n_tree_search = 2,
                             opt_interval = 100,
                             learning_rate = 0.4,
                             memory_policy = 'conservative',
                             verbose = True )

#ETAPA 18: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO FASTRGFREGRESSOR

from rgf.sklearn import FastRGFRegressor

regressor_frgf = FastRGFRegressor( n_estimators = 50,
                                   max_depth = 3,
                                   max_leaf = 20,
                                   tree_gain_ratio = 0.3,
                                   min_samples_leaf = 0.5,
                                   l1 = 0.6,
                                   l2 = 100.0,
                                   opt_algorithm = 'rgf',
                                   learning_rate = 0.05,
                                   max_bin = 150,
                                   min_child_weight = 9.0,
                                   data_l2 = 9.0,
                                   sparse_max_features = 1000,
                                   sparse_min_occurences = 2,
                                   n_jobs = -1,
                                   verbose = True )

#ETAPA 19: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO RANDOM FOREST

from sklearn.ensemble import RandomForestRegressor

regressor_rf = RandomForestRegressor(n_estimators = 3000)

regressor_rf.fit(X_train_reg, Y_train_reg)

Y_pred_train_reg_rf = regressor_rf.predict(X_train_reg)

Y_pred_test_reg_rf = regressor_rf.predict(X_test_reg)

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_rf = explained_variance_score(Y_test_reg, Y_pred_test_reg_rf)
mtrc_mae_rf = mean_absolute_error(Y_test_reg, Y_pred_test_reg_rf)
mtrc_mdae_rf = median_absolute_error(Y_test_reg, Y_pred_test_reg_rf)
mtrc_mse_rf = mean_squared_error(Y_test_reg, Y_pred_test_reg_rf)
mtrc_rmse_rf = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_rf))
mtrc_msle_rf = mean_squared_log_error(Y_test_reg, Y_pred_test_reg_rf)
mtrc_r2_rf = r2_score(Y_test_reg, Y_pred_test_reg_rf)
mtrc_max_error_rf = max_error(Y_test_reg, Y_pred_test_reg_rf)

print('Explained Variance Score:', mtrc_evs_rf)
print('Mean Absolute Error:', mtrc_mae_rf)
print('Median Absolute Error:', mtrc_mdae_rf)
print('Mean Squared Error:', mtrc_mse_rf)  
print('Root Mean Squared Error:', mtrc_rmse_rf)
print('Mean Squared Logarithmic Error:', mtrc_msle_rf)
print('R² Score:', mtrc_r2_rf)
print('Max Error:', mtrc_max_error_rf)

print('Explained Variance Score: {:.4f}'.format(mtrc_evs_rf))
print('Mean Absolute Error: {:.4f}'.format(mtrc_mae_rf)) 
print('Median Absolute Error: {:.4f}'.format(mtrc_mdae_rf)) 
print('Mean Squared Error: {:.4f}'.format(mtrc_mse_rf))
print('Root Mean Squared Error: {:.4f}'.format(mtrc_rmse_rf))
print('Mean Squared Logarithmic Error: {:.4f}'.format(mtrc_msle_rf))
print('R² Score: {:.4f}'.format(mtrc_r2_rf))
print('Max Error: {:.4f}'.format(mtrc_max_error_rf))

#Mean Absolute Error: 4.0332
#Mean Squared Error: 61.7493
#Root Mean Squared Error: 7.8581
#Explained Variance Score: 0.2311
#Mean Squared Logarithmic Error: 0.5446
#Median Absolute Error: 2.2723
#R² Score: 0.2311
#Max Error: 98.9110

#Plot - Y_test_reg vs Y_pred_test_reg_rf

#Ideally Should Have Been a Straight Line
plt.figure(figsize = (14, 7)) 
plt.scatter(Y_test_reg, Y_pred_test_reg_rf)
plt.show()

#Plot Residuals

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_rf, Y_pred_train_reg_rf - Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_rf, Y_pred_test_reg_rf - Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("Random Forest Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

#Plot Predictions

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_rf, Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_rf, Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("Random Forest Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Real Values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

#RandomForest Feature Importance

fea_imp = pd.DataFrame({'imp': regressor_rf.feature_importances_, 'col': X_list_reg})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending = [True, False]).iloc[-30:]
fea_imp.plot(kind = 'barh', x = 'col', y = 'imp', figsize = (14, 7), legend = None)
plt.title('RandomForest - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance')

#ETAPA 20: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO ADA BOOST

from sklearn.ensemble import AdaBoostRegressor

regressor_ab = AdaBoostRegressor(random_state = seed_reg)

#ETAPA 21: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO GRADIENT BOOST

from sklearn.ensemble import GradientBoostingRegressor

regressor_gb = GradientBoostingRegressor( n_estimators = 3000, 
                                          learning_rate = 0.05,
                                          max_depth = 4, 
                                          max_features = 'sqrt',
                                          min_samples_leaf = 15, 
                                          min_samples_split = 10, 
                                          loss = 'huber', 
                                          random_state = seed_reg )

#ETAPA 22: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO XGBOOST

from xgboost import XGBRegressor

regressor_xgb = XGBRegressor( colsample_bytree = 0.4603, 
                              gamma = 0.0468, 
                              learning_rate = 0.04, 
                              max_depth = 3, 
                              min_child_weight = 1.7817, 
                              n_estimators = 2200,
                              reg_alpha = 0.4640, 
                              reg_lambda = 0.8571,
                              subsample = 0.5213, 
                              silent = 1,
                              random_state = seed_reg, 
                              nthread = -1 )

#ETAPA 23: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO LIGHT GBM

from lightgbm import LGBMRegressor

regressor_lgbm = LGBMRegressor()

#LightGBM Model Training
#num_leaves is the main parameter to control the complexity of the tree model. when trying to tune the num_leaves, we should let it be smaller than 2^(max_depth) (225).
#We use max_depth to limit growing deep tree.
#for better accuracy, we us small learning_rate with large num_iterations.
#To speed up training and deal with overfitting, we set feature_fraction=0.6, that is, selecting 60% features before training each tree.
#Set verbosity = -1, eval metric on the eval set is printed at every verbose boosting stage.
#early_stopping_rounds = 500, the model will train until the validation score stops improving. Validation score needs to improve at least every 500 round(s) to continue training.
#verbose_eval = 500, an evaluation metric is printed every 500 boosting stages.

regressor_lgbm = LGBMRegressor( lcolsample_bytree = 0.4603, 
                                gamma = 0.0468, 
                                learning_rate = 0.04, 
                                max_depth = 3, 
                                min_child_weight = 1.7817, 
                                n_estimators = 2200,
                                reg_alpha = 0.4640, 
                                reg_lambda = 0.8571,
                                subsample = 0.5213, 
                                silent = 1,
                                random_state = seed_reg, 
                                nthread = -1)


regressor_lgbm = LGBMRegressor ( objective = 'regression',
                                 metric = 'RMSE',
                                 num_leaves = 200,
                                 max_depth = 15,
                                 learning_rate = 0.01,
                                 feature_fraction = 0.6,
                                 verbosity = -1,
                                 num_iterations = 20000,
                                 verbose_eval = 500 )

regressor_lgbm.fit(X_train_reg, Y_train_reg)

Y_pred_train_reg_lgbm = regressor_lgbm.predict(X_train_reg)

Y_pred_test_reg_lgbm = regressor_lgbm.predict(X_test_reg)

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_lgbm = explained_variance_score(Y_test_reg, Y_pred_test_reg_lgbm)
mtrc_mae_lgbm = mean_absolute_error(Y_test_reg, Y_pred_test_reg_lgbm)
mtrc_mdae_lgbm = median_absolute_error(Y_test_reg, Y_pred_test_reg_lgbm)
mtrc_mse_lgbm = mean_squared_error(Y_test_reg, Y_pred_test_reg_lgbm)
mtrc_rmse_lgbm = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_lgbm))
#mtrc_msle_lgbm = mean_squared_log_error(Y_test_reg, Y_pred_test_reg_lgbm)
mtrc_r2_lgbm = r2_score(Y_test_reg, Y_pred_test_reg_lgbm)
mtrc_max_error_lgbm = max_error(Y_test_reg, Y_pred_test_reg_lgbm)

print('Explained Variance Score:', mtrc_evs_lgbm)
print('Mean Absolute Error:', mtrc_mae_lgbm)
print('Median Absolute Error:', mtrc_mdae_lgbm)
print('Mean Squared Error:', mtrc_mse_lgbm)  
print('Root Mean Squared Error:', mtrc_rmse_lgbm)
#print('Mean Squared Logarithmic Error:', mtrc_msle_lgbm)
print('R² Score:', mtrc_r2_lgbm)
print('Max Error:', mtrc_max_error_lgbm)

print('Explained Variance Score: {:.4f}'.format(mtrc_evs_lgbm))
print('Mean Absolute Error: {:.4f}'.format(mtrc_mae_lgbm)) 
print('Median Absolute Error: {:.4f}'.format(mtrc_mdae_lgbm)) 
print('Mean Squared Error: {:.4f}'.format(mtrc_mse_lgbm))
print('Root Mean Squared Error: {:.4f}'.format(mtrc_rmse_lgbm))
#print('Mean Squared Logarithmic Error: {:.4f}'.format(mtrc_msle_lgbm))
print('R² Score: {:.4f}'.format(mtrc_r2_lgbm))
print('Max Error: {:.4f}'.format(mtrc_max_error_lgbm))

#Explained Variance Score: 0.2373
#Mean Absolute Error: 3.9213
#Median Absolute Error: 2.0385
#Mean Squared Error: 61.2976
#Root Mean Squared Error: 7.8293
#R² Score: 0.2367
#Max Error: 95.7378

#Plot - Y_test_reg vs Y_pred_test_reg_lgbm

#Ideally Should Have Been a Straight Line
plt.figure(figsize = (14, 7)) 
plt.scatter(Y_test_reg, Y_pred_test_reg_lgbm)
plt.show()

#Plot Residuals

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_lgbm, Y_pred_train_reg_lgbm - Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_lgbm, Y_pred_test_reg_lgbm - Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("CatBoost Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

#Plot Predictions

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_lgbm, Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_lgbm, Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("CatBoost Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Real Values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

#LightGBM Feature Importance

fea_imp = pd.DataFrame({'imp': regressor_lgbm.feature_importances_, 'col': X_list_reg})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending = [True, False]).iloc[-30:]
fea_imp.plot(kind = 'barh', x = 'col', y = 'imp', figsize = (14, 7), legend = None)
plt.title('LightGBM Regressor - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance')

#ETAPA 24: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO CATBOOST

from catboost import CatBoostRegressor

regressor_cbr = CatBoostRegressor()

#CatBoost Model Tuning
#iterations is maximum number of trees that can be built when solving machine learning problems.
#learning_rate is used for reducing the gradient step.
#depth is the depth of the tree. Any integer up to 16 when using CPU.
#We calculate RMSE as metric.
#bagging_temperature defines the settings of the Bayesian bootstrap, the higher the value the more aggressive the bagging is. We do not want it high.
#We will use the overfitting detector, so, if overfitting occurs, CatBoost can stop the training earlier than the training parameters dictate. And the type of the overfitting detector is “Iter”.
#metric_period is the frequency of iterations to calculate the values of objectives and metrics.
#od_wait, consider the model overfitted and stop training after the specified number of iterations (100) since the iteration with the optimal metric value.
#eval_set is the validation dataset for overfitting detector, best iteration selection and monitoring metrics’ changes.
#use_best_model=True if a validation set is input (the eval_setparameter is defined) and at least one of the label values of objects in this set differs from the others.

regressor_cbr = CatBoostRegressor( iterations = 50, 
                                   depth = 3, 
                                   learning_rate = 0.1, 
                                   loss_function = 'RMSE',
                                   random_seed = seed_reg )

regressor_cbr = CatBoostRegressor( iterations = 200,
                                   learning_rate = 0.01,
                                   depth = 16,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.2,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

regressor_cbr.fit(X_train_reg, Y_train_reg)

Y_pred_train_reg_cbr = regressor_cbr.predict(X_train_reg)

Y_pred_test_reg_cbr = regressor_cbr.predict(X_test_reg)

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import mean_squared_log_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_cbr = explained_variance_score(Y_test_reg, Y_pred_test_reg_cbr)
mtrc_mae_cbr = mean_absolute_error(Y_test_reg, Y_pred_test_reg_cbr)
mtrc_mdae_cbr = median_absolute_error(Y_test_reg, Y_pred_test_reg_cbr)
mtrc_mse_cbr = mean_squared_error(Y_test_reg, Y_pred_test_reg_cbr)
mtrc_rmse_cbr = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_cbr))
mtrc_msle_cbr = mean_squared_log_error(Y_test_reg, Y_pred_test_reg_cbr)
mtrc_r2_cbr = r2_score(Y_test_reg, Y_pred_test_reg_cbr)
mtrc_max_error_cbr = max_error(Y_test_reg, Y_pred_test_reg_cbr)

print('Explained Variance Score:', mtrc_evs_cbr)
print('Mean Absolute Error:', mtrc_mae_cbr)
print('Median Absolute Error:', mtrc_mdae_cbr)
print('Mean Squared Error:', mtrc_mse_cbr)  
print('Root Mean Squared Error:', mtrc_rmse_cbr)
print('Mean Squared Logarithmic Error:', mtrc_msle_cbr)
print('R² Score:', mtrc_r2_cbr)
print('Max Error:', mtrc_max_error_cbr)

print('Explained Variance Score: {:.4f}'.format(mtrc_evs_cbr))
print('Mean Absolute Error: {:.4f}'.format(mtrc_mae_cbr)) 
print('Median Absolute Error: {:.4f}'.format(mtrc_mdae_cbr)) 
print('Mean Squared Error: {:.4f}'.format(mtrc_mse_cbr))
print('Root Mean Squared Error: {:.4f}'.format(mtrc_rmse_cbr))
print('Mean Squared Logarithmic Error: {:.4f}'.format(mtrc_msle_cbr))
print('R² Score: {:.4f}'.format(mtrc_r2_cbr))
print('Max Error: {:.4f}'.format(mtrc_max_error_cbr))

#Explained Variance Score: 0.0540
#Mean Absolute Error: 4.0413
#Median Absolute Error: 1.5084
#Mean Squared Error: 84.8436
#Root Mean Squared Error: 9.2111
#Mean Squared Logarithmic Error: 0.6309
#R² Score: -0.0565
#Max Error: 113.5833

#Plot - Y_test_reg vs Y_pred_test_reg_cbr

#Ideally Should Have Been a Straight Line
plt.figure(figsize = (14, 7)) 
plt.scatter(Y_test_reg, Y_pred_test_reg_cbr)
plt.show()

#Plot Residuals

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_cbr, Y_pred_train_reg_cbr - Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_cbr, Y_pred_test_reg_cbr - Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("CatBoost Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

#Plot Predictions

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_cbr, Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_cbr, Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("CatBoost Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Real Values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

#CatBoost Feature Importance

fea_imp = pd.DataFrame({'imp': regressor_cbr.feature_importances_, 'col': X_list_reg})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending = [True, False]).iloc[-30:]
fea_imp.plot(kind = 'barh', x = 'col', y = 'imp', figsize = (14, 7), legend = None)
plt.title('CatBoost - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance')

#ETAPA 25: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO MLPREGRESSOR

from sklearn.neural_network import MLPRegressor

regressor_mlp = MLPRegressor()
