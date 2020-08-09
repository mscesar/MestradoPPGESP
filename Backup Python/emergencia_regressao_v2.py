#Setup da Pasta de Trabalho
import os
os.chdir('C:/Users/Mauricio/Dropbox/MauricioPessoal/Pós-Graduação/PPGESP/Projeto Final/Estudo Python/Bases Cárdio Pulmonar')

import warnings
warnings.filterwarnings("ignore")

#Variáveis de Uso Geral

seed_reg = 42

#ETAPA 01: TRATAMENTO DOS DADOS

#Importação do Pandas

import pandas as pd

pd.set_option('display.float_format', lambda x: '%.3f' % x)

#Importação dos Dados e Seleção das Linhas de Pacientes Internados

import time

start = time.time()
dfr = pd.read_excel('TB_IND_ASSIST_EME_0115_0319.xlsx')
end = time.time()

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 6.56 min

dfr_temp = dfr.loc[(dfr['CONT_STATUS_INT'] == 1)].copy()

#Análise Descritiva da Variável Target para Detecção de Outliers

dfr_temp['TEMPO_PERM_INT_POSTERIOR'].describe()
#count   9719.000
#mean       5.687
#std        8.546
#min        0.000
#25%        1.000
#50%        3.000
#75%        6.000
#max      127.000

dfr_temp['TEMPO_PERM_INT_POSTERIOR'].median()
#3.0

dfr_temp['TEMPO_PERM_INT_POSTERIOR'].var()
#73.03673490571809

dfr_temp['TEMPO_PERM_INT_POSTERIOR'].mad()
#4.818920165763722

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (14,7))
plt.hist(dfr_temp['TEMPO_PERM_INT_POSTERIOR'])

plt.figure(figsize = (14,7))
plt.hist(dfr_temp['TEMPO_PERM_INT_POSTERIOR'], bins = 100)

plt.figure(figsize = (14,7))
sns.distplot(dfr_temp['TEMPO_PERM_INT_POSTERIOR'])

print("Skewness: %f" % dfr_temp['TEMPO_PERM_INT_POSTERIOR'].skew())
print("Kurtosis: %f" % dfr_temp['TEMPO_PERM_INT_POSTERIOR'].kurt())

plt.figure(figsize = (14,7))
sns.boxplot(y = dfr_temp['TEMPO_PERM_INT_POSTERIOR'], palette = 'Blues', orient = 'h', linewidth=  2)

#Criação da Base df_reg com Todas as Informações Disponíveis

df_reg = dfr_temp.loc[(dfr_temp['CONT_STATUS_INT'] == 1) & (dfr_temp['TEMPO_PERM_INT_POSTERIOR'] <= 10)].copy()

#df_reg = dfr_temp[dfr_temp["CONT_STATUS_INT"] == 1]

#df_reg = dfr_temp[(dfr_temp.CONT_STATUS_INT == 1)]

del dfr

del dfr_temp

#Verificação da Quantidade de Linhas e Colunas da Base de Dados

df_reg.shape
#(9006, 404)
#(8448, 404)

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

df_reg['TEMPO_PERM_INT_POSTERIOR'].describe()

df_reg['TEMPO_PERM_INT_POSTERIOR'].median()

df_reg['TEMPO_PERM_INT_POSTERIOR'].var()

df_reg['TEMPO_PERM_INT_POSTERIOR'].mad()

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

import seaborn as sns

import matplotlib.pyplot as plt 

(mu, sigma) = norm.fit(df_reg['TEMPO_PERM_INT_POSTERIOR'])

plt.figure(figsize = (14, 7))
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

#Análise de Correlação das Variáveis - Pearson

import numpy as np

corr_matrix_pearson_reg = df_reg.corr('pearson')

corr_matrix_pearson_reg['TEMPO_PERM_INT_POSTERIOR'].sort_values(ascending = False)

k_reg = 20

corr_matrix_pearson_reg.nlargest(k_reg, 'TEMPO_PERM_INT_POSTERIOR')['TEMPO_PERM_INT_POSTERIOR']

cols_reg = corr_matrix_pearson_reg.nlargest(k_reg, 'TEMPO_PERM_INT_POSTERIOR')['TEMPO_PERM_INT_POSTERIOR'].index
cm_reg = np.corrcoef(df_reg[cols_reg].values.T)
sns.set(font_scale = 1.2)
plt.figure(figsize = (10, 10))
hm_reg = sns.heatmap(cm_reg, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols_reg.values, xticklabels = cols_reg.values)
plt.show()

corr_matrix_pearson_reg_new = df_reg[corr_matrix_pearson_reg['TEMPO_PERM_INT_POSTERIOR'].nlargest(20).index].corr('pearson')

#Análise de Correlação das Variáveis - Spearman

import numpy as np

corr_matrix_spearman_reg = df_reg.corr('spearman')

corr_matrix_spearman_reg['TEMPO_PERM_INT_POSTERIOR'].sort_values(ascending = False)

k_reg = 20

corr_matrix_spearman_reg.nlargest(k_reg, 'TEMPO_PERM_INT_POSTERIOR')['TEMPO_PERM_INT_POSTERIOR']

#cols_reg = corr_matrix_spearman_reg.nlargest(k_reg, 'TEMPO_PERM_INT_POSTERIOR')['TEMPO_PERM_INT_POSTERIOR'].index
#cm_reg = np.corrcoef(df_reg[cols_reg].values.T)
#sns.set(font_scale = 1.2)
#plt.figure(figsize = (10, 10))
#hm_reg = sns.heatmap(cm_reg, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols_reg.values, xticklabels = cols_reg.values)
#plt.show()

corr_matrix_spearman_reg_new = df_reg[corr_matrix_spearman_reg['TEMPO_PERM_INT_POSTERIOR'].nlargest(20).index].corr('spearman')

cols_reg = corr_matrix_spearman_reg_new.index
sns.set(font_scale = 1.2)
plt.figure(figsize = (10, 10))
hm_reg = sns.heatmap(corr_matrix_spearman_reg_new, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols_reg.values, xticklabels = cols_reg.values)
plt.show()

#Análise de Correlação das Variáveis - Kendall

import numpy as np

corr_matrix_kendall_reg = df_reg.corr('kendall')

corr_matrix_kendall_reg['TEMPO_PERM_INT_POSTERIOR'].sort_values(ascending = False)

k_reg = 20

corr_matrix_kendall_reg.nlargest(k_reg, 'TEMPO_PERM_INT_POSTERIOR')['TEMPO_PERM_INT_POSTERIOR']

#cols_reg = corr_matrix_kendall_reg.nlargest(k_reg, 'TEMPO_PERM_INT_POSTERIOR')['TEMPO_PERM_INT_POSTERIOR'].index
#cm_reg = np.corrcoef(df_reg[cols_reg].values.T)
#sns.set(font_scale = 1.2)
#plt.figure(figsize = (10, 10))
#hm_reg = sns.heatmap(cm_reg, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols_reg.values, xticklabels = cols_reg.values)
#plt.show()

corr_matrix_kendall_reg_new = df_reg[corr_matrix_kendall_reg['TEMPO_PERM_INT_POSTERIOR'].nlargest(20).index].corr('kendall')

cols_reg = corr_matrix_kendall_reg_new.index
sns.set(font_scale = 1.2)
plt.figure(figsize = (10, 10))
hm_reg = sns.heatmap(corr_matrix_kendall_reg_new, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols_reg.values, xticklabels = cols_reg.values)
plt.show()

#Análise de Correlação das Variáveis - Gráficos - Pearson

import matplotlib.pyplot as plt

import seaborn as sns

#Heat Map - Pearson Correlation - Op 1

print("PEARSON CORRELATION")
print(corr_matrix_pearson_reg_new)
plt.figure(figsize = (56, 28))
sns.heatmap(corr_matrix_pearson_reg_new)
plt.savefig("heatmap_pearson.png", dpi = 480, orientation = 'landscape')
plt.clf()
plt.close()

#Heat Map - Pearson Correlation - Op 2

fig = plt.figure(figsize = (12,9))
sns.heatmap(corr_matrix_pearson_reg_new, vmax = 0.8, square = True)
plt.show()

#Heat Map - Pearson Correlation - Op 3

import numpy as np

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix_pearson_reg_new, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix_pearson_reg_new, mask = mask, cmap = cmap, vmax = .3, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})

#Heat Map - Pearson Correlation - Op 4

plt.figure(figsize = (28,28))
sns.heatmap(corr_matrix_pearson_reg_new, vmax = 1, square = True, annot = True, cmap = 'viridis')
plt.title('Correlation Between Different Fearures')

#Heat Map - Pearson Correlation - Op 5

plt.figure(figsize = (28,28))
sns.heatmap(corr_matrix_pearson_reg_new, vmax = 0.8, square = True)
plt.xticks(rotation = 45, fontsize = 7); plt.yticks(rotation = 45, fontsize = 7)

#Análise de Correlação das Variáveis - Gráficos - Spearman

import matplotlib.pyplot as plt

import seaborn as sns

#Heat Map - Spearman Correlation - Op 1

print("SPEARMAN CORRELATION")
print(corr_matrix_spearman_reg_new)
plt.figure(figsize = (56, 28))
sns.heatmap(corr_matrix_spearman_reg_new)
plt.savefig("heatmap_spearman.png", dpi = 480, orientation = 'landscape')
plt.clf()
plt.close()

#Heat Map - Spearman Correlation - Op 2

fig = plt.figure(figsize = (12,9))
sns.heatmap(corr_matrix_spearman_reg_new, vmax = 0.8, square = True)
plt.show()

#Heat Map - Spearman Correlation - Op 3

import numpy as np

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix_spearman_reg_new, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix_spearman_reg_new, mask = mask, cmap = cmap, vmax = .3, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})

#Heat Map - Spearman Correlation - Op 4

plt.figure(figsize = (28,28))
sns.heatmap(corr_matrix_spearman_reg_new, vmax = 1, square = True, annot = True, cmap = 'viridis')
plt.title('Correlation Between Different Fearures')

#Heat Map - Spearman Correlation - Op 5

plt.figure(figsize = (28,28))
sns.heatmap(corr_matrix_spearman_reg_new, vmax = 0.8, square = True)
plt.xticks(rotation = 45, fontsize = 7); plt.yticks(rotation = 45, fontsize = 7)

#Análise de Correlação das Variáveis - Gráficos - Kendall

import matplotlib.pyplot as plt

import seaborn as sns

#Heat Map - Kendall Correlation - Op 1

print("KENDALL CORRELATION")
print(corr_matrix_kendall_reg_new)
plt.figure(figsize = (56, 28))
sns.heatmap(corr_matrix_kendall_reg_new)
plt.savefig("heatmap_kendall.png", dpi = 480, orientation = 'landscape')
plt.clf()
plt.close()

#Heat Map - Kendall Correlation - Op 2

fig = plt.figure(figsize = (12,9))
sns.heatmap(corr_matrix_kendall_reg_new, vmax = 0.8, square = True)
plt.show()

#Heat Map - Kendall Correlation - Op 3

import numpy as np

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix_kendall_reg_new, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix_kendall_reg_new, mask = mask, cmap = cmap, vmax = .3, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})

#Heat Map - Kendall Correlation - Op 4

plt.figure(figsize = (28,28))
sns.heatmap(corr_matrix_kendall_reg_new, vmax = 1, square = True, annot = True, cmap = 'viridis')
plt.title('Correlation Between Different Fearures')

#Heat Map - Kendall Correlation - Op 5

plt.figure(figsize = (28,28))
sns.heatmap(corr_matrix_kendall_reg_new, vmax = 0.8, square = True)
plt.xticks(rotation = 45, fontsize = 7); plt.yticks(rotation = 45, fontsize = 7)

#Análise de Valores Nulos ou Não Existentes

df_reg.isnull().sum()

df_reg.isna().sum()

for col in df_reg.columns:
    if df_reg[col].isnull().sum() > 0:
        print(col)
        
#Análise de Valores Nulos ou Não Existentes - Data Frame        
        
total_reg = df_reg.isnull().sum().sort_values(ascending = False)

percent_1_reg = df_reg.isnull().sum()/df_reg.isnull().count()*100

percent_2_reg = (round(percent_1_reg, 2)).sort_values(ascending = False)

missing_data_reg = pd.concat([total_reg, percent_2_reg], axis = 1, keys = ['Total', '%'])

missing_data_reg[missing_data_reg.Total > 0].sort_values(ascending = False, by = 'Total').head(50)        
    
#Preenchimento dos Valores Nulos

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

pd.value_counts(df_reg['TEMPO_ATENDIMENTO'])
df_reg['TEMPO_ATENDIMENTO'].count()
df_reg['TEMPO_ATENDIMENTO'].fillna(-1, inplace = True)

pd.value_counts(df_reg['TEMPO_ATD_ULT_ATENDIMENTO'])
df_reg['TEMPO_ATD_ULT_ATENDIMENTO'].count()
df_reg['TEMPO_ATD_ULT_ATENDIMENTO'].fillna(-1, inplace = True)

pd.value_counts(df_reg['MEDIA_TEMPO_ATD_ATD_ANTERIOR'])
df_reg['MEDIA_TEMPO_ATD_ATD_ANTERIOR'].count()
df_reg['MEDIA_TEMPO_ATD_ATD_ANTERIOR'].fillna(-1, inplace = True)

pd.value_counts(df_reg['TEMPO_PERM_ULT_INTERNAMENTO'])
df_reg['TEMPO_PERM_ULT_INTERNAMENTO'].count()
df_reg['TEMPO_PERM_ULT_INTERNAMENTO'].fillna(-1, inplace = True)

pd.value_counts(df_reg['MEDIA_TEMPO_PERM_INT_ANT'])
df_reg['MEDIA_TEMPO_PERM_INT_ANT'].count()
df_reg['MEDIA_TEMPO_PERM_INT_ANT'].fillna(-1, inplace = True)

#Verificação dos Valores Nulos Ainda Existentes

for col in df_reg.columns:
    if df_reg[col].isnull().sum() > 0:
        print(col)

#Exclusão das Colunas Que Não Serão Utilizadas
        
df_reg_trat = df_reg.drop('DATA_ADM', axis = 1)

df_reg_trat = df_reg_trat.drop('DESC_DIA_DA_SEMANA', axis = 1)

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

df_reg_trat = df_reg_trat.drop('CONT_STATUS_INT', axis = 1)

df_reg_trat = df_reg_trat.drop('DESC_STATUS_INT', axis = 1)

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

X_reg['DESC_ESTACAO_ANO'] = lbl_enc_X_reg.fit_transform(X_reg['DESC_ESTACAO_ANO'])

X_reg['DESC_FINAL_SEMANA'] = lbl_enc_X_reg.fit_transform(X_reg['DESC_FINAL_SEMANA'])

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

feature_list_X_reg = list(X_reg.columns)

feature_list_X_reg_integer, feature_list_X_reg_float, features_list_X_reg_object = groupFeatures(feature_list_X_reg, X_reg)

print("# of integer feature : ", len(feature_list_X_reg_integer))
print("# of float feature : ", len(feature_list_X_reg_float))
print("# of object feature : ", len(features_list_X_reg_object))

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

#Training X_reg Data Dimensions: (6304, 380)
#Training Y_reg Data Dimensions: (6304,)
#Testing X_reg Data Dimensions: (2702, 380)
#Testing Y_reg Data Dimensions: (2702,)

#Training X_reg Data Dimensions: (5913, 380)
#Training Y_reg Data Dimensions: (5913,)
#Testing X_reg Data Dimensions: (2535, 380)
#Testing Y_reg Data Dimensions: (2535,)

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

#TESTE01: Treino e Teste dos Modelos com CrossValScore + 10 KFold(shuffle = False) - X_train_reg, Y_train_reg

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

import time

model_results_reg = []
model_std_reg = []
model_names_reg   = []

start = time.time()
for model_name_reg in models_reg:
	model_reg   = models_reg[model_name_reg]
	k_fold_reg  = KFold(n_splits = folds_reg, random_state = seed_reg, shuffle = False)
	results_reg = np.sqrt(-cross_val_score(model_reg, X_train_reg, Y_train_reg, cv = k_fold_reg, scoring = metric_reg))    
	
	score_reg = round(results_reg.mean(),3)
	std_reg = round(results_reg.std(),3)
	model_results_reg.append(score_reg)
	model_std_reg.append(std_reg)
	model_names_reg.append(model_name_reg)
	#print("{}: {}, {}".format(model_name_reg, score_reg, std_reg))
end = time.time()
kf_cross_val_reg = pd.DataFrame({'Name': model_names_reg, 'RMSE': model_results_reg, 'Desvio Padrão': model_std_reg})

print("Tempo de Execução: {:.2f} sec".format(end - start))
print(kf_cross_val_reg)

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_regressor = pd.ExcelWriter('cross_val_score_regressor_X_train_Y_train_kfold.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val_reg.to_excel(cross_val_score_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_regressor.save()

#TESTE02: Treino e Teste dos Modelos com CrossValScore + 10 KFold(shuffle = True) - X_train_reg, Y_train_reg

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

import time

model_results_reg = []
model_std_reg = []
model_names_reg   = []

start = time.time()
for model_name_reg in models_reg:
	model_reg   = models_reg[model_name_reg]
	k_fold_reg  = KFold(n_splits = folds_reg, random_state = seed_reg, shuffle = True)
	results_reg = np.sqrt(-cross_val_score(model_reg, X_train_reg, Y_train_reg, cv = k_fold_reg, scoring = metric_reg))
    
	score_reg = round(results_reg.mean(),3)
	std_reg = round(results_reg.std(),3)
	model_results_reg.append(score_reg)
	model_std_reg.append(std_reg)
	model_names_reg.append(model_name_reg)
	#print("{}: {}, {}".format(model_name_reg, score_reg, std_reg))  
end = time.time()
kf_cross_val_reg = pd.DataFrame({'Name': model_names_reg, 'RMSE': model_results_reg, 'Desvio Padrão': model_std_reg})

print("Tempo de Execução: {:.2f} sec".format(end - start))
print(kf_cross_val_reg)

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_regressor = pd.ExcelWriter('cross_val_score_regressor_X_train_Y_train_kfold_shuffle.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val_reg.to_excel(cross_val_score_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_regressor.save()

#TESTE03: Treino e Teste dos Modelos com CrossValScore + 10 KFold(shuffle = False) - X_reg, Y_reg

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

import time

model_results_reg = []
model_std_reg = []
model_names_reg   = []

start = time.time()
for model_name_reg in models_reg:
	model_reg   = models_reg[model_name_reg]
	k_fold_reg  = KFold(n_splits = folds_reg, random_state = seed_reg, shuffle = False)
	results_reg = np.sqrt(-cross_val_score(model_reg, X_reg, Y_reg, cv = k_fold_reg, scoring = metric_reg))
    
	score_reg = round(results_reg.mean(),3)
	std_reg = round(results_reg.std(),3)
	model_results_reg.append(score_reg)
	model_std_reg.append(std_reg)
	model_names_reg.append(model_name_reg)
	#print("{}: {}, {}".format(model_name_reg, score_reg, std_reg))
end = time.time()
kf_cross_val_reg = pd.DataFrame({'Name': model_names_reg, 'RMSE': model_results_reg, 'Desvio Padrão': model_std_reg})

print("Tempo de Execução: {:.2f} sec".format(end - start))
print(kf_cross_val_reg)

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_regressor = pd.ExcelWriter('cross_val_score_regressor_X_Y_kfold.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val_reg.to_excel(cross_val_score_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_regressor.save()

#TESTE04: Treino e Teste dos Modelos com CrossValScore + 10 KFold(shuffle = True) - X_reg, Y_reg

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

import time

model_results_reg = []
model_std_reg = []
model_names_reg   = []

start = time.time()
for model_name_reg in models_reg:
	model_reg   = models_reg[model_name_reg]
	k_fold_reg  = KFold(n_splits = folds_reg, random_state = seed_reg, shuffle = True)
	results_reg = np.sqrt(-cross_val_score(model_reg, X_reg, Y_reg, cv = k_fold_reg, scoring = metric_reg))
    
	score_reg = round(results_reg.mean(),3)
	std_reg = round(results_reg.std(),3)
	model_results_reg.append(score_reg)
	model_std_reg.append(std_reg)
	model_names_reg.append(model_name_reg)
	#print("{}: {}, {}".format(model_name_reg, score_reg, std_reg))
end = time.time()
kf_cross_val_reg = pd.DataFrame({'Name': model_names_reg, 'RMSE': model_results_reg, 'Desvio Padrão': model_std_reg})

print("Tempo de Execução: {:.2f} sec".format(end - start))
print(kf_cross_val_reg)

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_regressor = pd.ExcelWriter('cross_val_score_regressor_X_Y_kfold_shuffle.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val_reg.to_excel(cross_val_score_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_regressor.save()

#Plotagem das Acurácias com CrossValScore + 10 KFold

import seaborn as sns

import matplotlib.pyplot as plt

axis = sns.barplot(kf_cross_val_reg['Name'], kf_cross_val_reg['RMSE'])
axis.set(xlabel = 'Regressor', ylabel = 'RMSE')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha = "center") 
    
plt.figure(figsize = (14, 7))     
plt.show()

#TESTE05: Treino e Teste dos Modelos com CrossValScore + ShuffleSplit(n_splits = 10) - X_train_reg, Y_train_reg

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

import time

model_results_reg = []
model_std_reg = []
model_names_reg   = []

start = time.time()
for model_name_reg in models_reg:
	model_reg   = models_reg[model_name_reg]
	shuffle_reg = ShuffleSplit(n_splits = folds_reg, test_size = 0.3, random_state = seed_reg)
	results_reg = np.sqrt(-cross_val_score(model_reg, X_train_reg, Y_train_reg, cv = shuffle_reg, scoring = metric_reg))
    
	score_reg = round(results_reg.mean(),3)
	std_reg = round(results_reg.std(),3)
	model_results_reg.append(score_reg)
	model_std_reg.append(std_reg)
	model_names_reg.append(model_name_reg)
	#print("{}: {}, {}".format(model_name_reg, score_reg, std_reg))
end = time.time()
kf_cross_val_reg = pd.DataFrame({'Name': model_names_reg, 'RMSE': model_results_reg, 'Desvio Padrão': model_std_reg})

print("Tempo de Execução: {:.2f} sec".format(end - start))
print(kf_cross_val_reg)

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_regressor = pd.ExcelWriter('cross_val_score_regressor_shuffle_X_train_Y_train.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val_reg.to_excel(cross_val_score_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_regressor.save()

#TESTE06: Treino e Teste dos Modelos com CrossValScore + ShuffleSplit(n_splits = 10) - X_reg, Y_reg

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.metrics import mean_squared_error

import time

model_results_reg = []
model_std_reg = []
model_names_reg   = []

start = time.time()
for model_name_reg in models_reg:
	model_reg   = models_reg[model_name_reg]
	shuffle_reg = ShuffleSplit(n_splits = folds_reg, test_size = 0.3, random_state = seed_reg)
	results_reg = np.sqrt(-cross_val_score(model_reg, X_reg, Y_reg, cv = shuffle_reg, scoring = metric_reg))
    
	score_reg = round(results_reg.mean(),3)
	std_reg = round(results_reg.std(),3)
	model_results_reg.append(score_reg)
	model_std_reg.append(std_reg)
	model_names_reg.append(model_name_reg)
	#print("{}: {}, {}".format(model_name_reg, score_reg, std_reg))
end = time.time()
kf_cross_val_reg = pd.DataFrame({'Name': model_names_reg, 'RMSE': model_results_reg, 'Desvio Padrão': model_std_reg})

print("Tempo de Execução: {:.2f} sec".format(end - start))
print(kf_cross_val_reg)

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_regressor = pd.ExcelWriter('cross_val_score_regressor_shuffle_X_Y.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val_reg.to_excel(cross_val_score_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_regressor.save()

#Plotagem das Acurácias com CrossValScore + ShuffleSplit(n_splits = 10)

import seaborn as sns

import matplotlib.pyplot as plt

axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val_reg)
axis.set(xlabel = 'Classifier', ylabel = 'RMSE')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha = "center") 
    
plt.figure(figsize = (14, 7))    
plt.show()

#Treino e Teste dos Modelos com CrossValidate + 10 KFold(shuffle = False) + Várias Métricas - X_train_reg, Y_train_reg

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate

from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import median_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.metrics import max_error

import time

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

start = time.time()
for model_name_reg in models_reg:
    model_reg = models_reg[model_name_reg]
    kfold_reg = KFold(n_splits = 10, random_state = seed_reg, shuffle = False ) 
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

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_validate_regressor = pd.ExcelWriter('cross_validate_regressor_X_train_Y_train_kfold.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_validate_reg.to_excel(cross_validate_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_validate_regressor.save()

#Treino e Teste dos Modelos com CrossValidate + 10 KFold(shuffle = True) + Várias Métricas - X_train_reg, Y_train_reg

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate

from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import median_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.metrics import max_error

import time

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

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_validate_regressor = pd.ExcelWriter('cross_validate_regressor_X_train_Y_train_kfold_shuffle.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_validate_reg.to_excel(cross_validate_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_validate_regressor.save()

#Treino e Teste dos Modelos com CrossValidate + 10 KFold(shuffle = False) + Várias Métricas - X_reg, Y_reg

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate

from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import median_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.metrics import max_error

import time

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

start = time.time()
for model_name_reg in models_reg:
    model_reg = models_reg[model_name_reg]
    kfold_reg = KFold(n_splits = 10, random_state = seed_reg, shuffle = False ) 
    result_reg = cross_validate( estimator = model_reg, 
                                 X = X_reg, 
                                 y = Y_reg, 
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

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_validate_regressor = pd.ExcelWriter('cross_validate_regressor_X_Y_kfold.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_validate_reg.to_excel(cross_validate_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_validate_regressor.save()

#Treino e Teste dos Modelos com CrossValidate + 10 KFold(shuffle = True) + Várias Métricas - X_reg, Y_reg

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate

from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import median_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.metrics import max_error

import time

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

start = time.time()
for model_name_reg in models_reg:
    model_reg = models_reg[model_name_reg]
    kfold_reg = KFold(n_splits = 10, random_state = seed_reg, shuffle = True ) 
    result_reg = cross_validate( estimator = model_reg, 
                                 X = X_reg, 
                                 y = Y_reg, 
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

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_validate_regressor = pd.ExcelWriter('cross_validate_regressor_X_Y_kfold_shuffle.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_validate_reg.to_excel(cross_validate_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_validate_regressor.save()

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

#Parametrização do Modelo

from sklearn.linear_model import LinearRegression

#1° Teste

regressor_linear = LinearRegression()

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_linear.get_params())

#ETAPA 05: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO REGRESSÃO POLINOMIAL

#Preparação da Base de Dados

from sklearn.preprocessing import PolynomialFeatures

poly = PolynomialFeatures(degree = 2)

X_train_poly_reg = poly.fit_transform(X_train_reg)

X_test_poly_reg = poly.fit_transform(X_test_reg)

#Parametrização do Modelo

from sklearn.linear_model import LinearRegression

#1° Teste

regressor_poly = LinearRegression()

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_poly.get_params())

# Fit

import time

start = time.time()

regressor_poly.fit(X_train_reg, Y_train_reg)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

#Predição

Y_pred_train_reg_poly = regressor_poly.predict(X_train_poly_reg)

Y_pred_test_reg_poly = regressor_poly.predict(X_test_poly_reg)

#Análise de Métricas

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_poly = explained_variance_score(Y_test_reg, Y_pred_test_reg_poly)
mtrc_mae_poly = mean_absolute_error(Y_test_reg, Y_pred_test_reg_poly)
mtrc_mdae_poly = median_absolute_error(Y_test_reg, Y_pred_test_reg_poly)
mtrc_mse_poly = mean_squared_error(Y_test_reg, Y_pred_test_reg_poly)
mtrc_rmse_poly = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_poly))
mtrc_r2_poly = r2_score(Y_test_reg, Y_pred_test_reg_poly)
mtrc_max_error_poly = max_error(Y_test_reg, Y_pred_test_reg_poly)

print('Explained Variance Score: {:.4f}'.format(mtrc_evs_poly))
print('Mean Absolute Error: {:.4f}'.format(mtrc_mae_poly)) 
print('Median Absolute Error: {:.4f}'.format(mtrc_mdae_poly)) 
print('Mean Squared Error: {:.4f}'.format(mtrc_mse_poly))
print('Root Mean Squared Error: {:.4f}'.format(mtrc_rmse_poly))
print('R² Score: {:.4f}'.format(mtrc_r2_poly))
print('Max Error: {:.4f}'.format(mtrc_max_error_poly))

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

#Parametrização do Modelo

from sklearn.linear_model import Lasso

#1° Teste

regressor_lasso = Lasso(alpha = 0.0005, random_state = seed_reg)

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_lasso.get_params())

#ETAPA 07: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO LASSOLARSIC

#Parametrização do Modelo

from sklearn.linear_model import LassoLarsIC

#1° Teste

#Criterion BIC
regressor_llic = LassoLarsIC(criterion = 'bic')

#Criterion AIC
regressor_llic = LassoLarsIC(criterion = 'aic')

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_llic.get_params())

# Fit

import time

start = time.time()

regressor_llic.fit(X_train_reg, Y_train_reg)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

#Predição

Y_pred_train_reg_llic = regressor_llic.predict(X_train_reg)

Y_pred_test_reg_llic = regressor_llic.predict(X_test_reg)

#Análise de Métricas

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_llic = explained_variance_score(Y_test_reg, Y_pred_test_reg_llic)
mtrc_mae_llic = mean_absolute_error(Y_test_reg, Y_pred_test_reg_llic)
mtrc_mdae_llic = median_absolute_error(Y_test_reg, Y_pred_test_reg_llic)
mtrc_mse_llic = mean_squared_error(Y_test_reg, Y_pred_test_reg_llic)
mtrc_rmse_llic = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_llic))
mtrc_r2_llic = r2_score(Y_test_reg, Y_pred_test_reg_llic)
mtrc_max_error_llic = max_error(Y_test_reg, Y_pred_test_reg_llic)

print('Explained Variance Score: {:.4f}'.format(mtrc_evs_llic))
print('Mean Absolute Error: {:.4f}'.format(mtrc_mae_llic)) 
print('Median Absolute Error: {:.4f}'.format(mtrc_mdae_llic)) 
print('Mean Squared Error: {:.4f}'.format(mtrc_mse_llic))
print('Root Mean Squared Error: {:.4f}'.format(mtrc_rmse_llic))
print('R² Score: {:.4f}'.format(mtrc_r2_llic))
print('Max Error: {:.4f}'.format(mtrc_max_error_llic))

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

#Parametrização do Modelo

from sklearn.linear_model import Ridge

#1° Teste

regressor_ridge = Ridge()

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_ridge.get_params())

#ETAPA 09: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO KERNELRIDGE

#Parametrização do Modelo

from sklearn.kernel_ridge import KernelRidge

#1° Teste

regressor_kridge = KernelRidge(alpha = 0.6, kernel = 'polynomial', degree = 2, coef0 = 2.5, random_state = seed_reg)

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_kridge.get_params())

#ETAPA 10: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO BAYESIANRIDGE

#Parametrização do Modelo

from sklearn.linear_model import BayesianRidge

#1° Teste

regressor_bridge = BayesianRidge()

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_bridge.get_params())

#ETAPA 11: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO ELASTICNET

#Parametrização do Modelo

from sklearn.linear_model import ElasticNet

#1° Teste

regressor_elnet = ElasticNet(alpha = 0.0005, l1_ratio = .9, random_state = seed_reg)

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_elnet.get_params())

#ETAPA 12: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO KNN

#Parametrização do Modelo

from sklearn.neighbors import KNeighborsRegressor

#1° Teste

regressor_knn = KNeighborsRegressor()

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_knn.get_params())

#ETAPA 13: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO SVM

#Parametrização do Modelo

from sklearn.svm import SVR

#1° Teste

regressor_svr = SVR()

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_svr.get_params())

#ETAPA 14: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO DECISION TREE

#Parametrização do Modelo

from sklearn.tree import DecisionTreeRegressor

#1° Teste

regressor_dtr = DecisionTreeRegressor()

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_dtr.get_params())

#ETAPA 15: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO EXTRA TREES

#Parametrização do Modelo

from sklearn.ensemble import ExtraTreesRegressor

#1° Teste

regressor_etr = ExtraTreesRegressor()

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_etr.get_params())

#ETAPA 16: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO MARS

#Parametrização do Modelo

from pyearth import Earth

#1° Teste

regressor_mars = Earth( max_degree = 1, 
                        penalty = 1.0, 
                        endspan = 5 )

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_mars.get_params())

#ETAPA 17: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO RGFREGRESSOR

#Parametrização do Modelo

from rgf.sklearn import RGFRegressor

#1° Teste

regressor_rgf = RGFRegressor( max_leaf = 300,
                              algorithm = "RGF_Sib",
                              test_interval = 100,
                              loss = "LS",
                              verbose = False )

#2° Teste

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

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_rgf.get_params())

#ETAPA 18: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO FASTRGFREGRESSOR

#Parametrização do Modelo

from rgf.sklearn import FastRGFRegressor

#1° Teste

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

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_frgf.get_params())

#ETAPA 19: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO RANDOM FOREST

#Parametrização do Modelo

from sklearn.ensemble import RandomForestRegressor

#1° Teste

regressor_rf = RandomForestRegressor(n_estimators = 3000)

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_rf.get_params())

# Fit

import time
start = time.time()

regressor_rf.fit(X_train_reg, Y_train_reg)

end = time.time()
print("Tempo de Execução: {} sec".format(end - start))

#Predição

Y_pred_train_reg_rf = regressor_rf.predict(X_train_reg)

Y_pred_test_reg_rf = regressor_rf.predict(X_test_reg)

#Análise de Métricas

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_rf = explained_variance_score(Y_test_reg, Y_pred_test_reg_rf)
mtrc_mae_rf = mean_absolute_error(Y_test_reg, Y_pred_test_reg_rf)
mtrc_mdae_rf = median_absolute_error(Y_test_reg, Y_pred_test_reg_rf)
mtrc_mse_rf = mean_squared_error(Y_test_reg, Y_pred_test_reg_rf)
mtrc_rmse_rf = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_rf))
mtrc_r2_rf = r2_score(Y_test_reg, Y_pred_test_reg_rf)
mtrc_max_error_rf = max_error(Y_test_reg, Y_pred_test_reg_rf)

print('Explained Variance Score: {:.4f}'.format(mtrc_evs_rf))
print('Mean Absolute Error: {:.4f}'.format(mtrc_mae_rf)) 
print('Median Absolute Error: {:.4f}'.format(mtrc_mdae_rf)) 
print('Mean Squared Error: {:.4f}'.format(mtrc_mse_rf))
print('Root Mean Squared Error: {:.4f}'.format(mtrc_rmse_rf))
print('R² Score: {:.4f}'.format(mtrc_r2_rf))
print('Max Error: {:.4f}'.format(mtrc_max_error_rf))

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
plt.title('Random Forest - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance')

#ETAPA 20: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO ADA BOOST

#Parametrização do Modelo

from sklearn.ensemble import AdaBoostRegressor

#1° Teste

regressor_ab = AdaBoostRegressor(random_state = seed_reg)

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_ab.get_params())

#ETAPA 21: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO GRADIENT BOOST

#Parametrização do Modelo

from sklearn.ensemble import GradientBoostingRegressor

#1° Teste

regressor_gb = GradientBoostingRegressor( n_estimators = 3000, 
                                          learning_rate = 0.05,
                                          max_depth = 4, 
                                          max_features = 'sqrt',
                                          min_samples_leaf = 15, 
                                          min_samples_split = 10, 
                                          loss = 'huber', 
                                          random_state = seed_reg )

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_gb.get_params())

#ETAPA 22: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO XGBOOST

#Parametrização do Modelo

from xgboost import XGBRegressor

#1° Teste

regressor_xgb = XGBRegressor()

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1872
#Mean Absolute Error: 2.2486
#Median Absolute Error: 1.7623
#Mean Squared Error: 8.8729
#Root Mean Squared Error: 2.9787
#Mean Squared Logarithmic Error: 0.4018
#R² Score: 0.1869
#Max Error: 12.4269

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1897
#Mean Absolute Error: 1.7589
#Median Absolute Error: 1.4178
#Mean Squared Error: 5.1352
#Root Mean Squared Error: 2.2661
#R² Score: 0.1895
#Max Error: 8.8021

#2° Teste

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

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1465
#Mean Absolute Error: 2.2838
#Median Absolute Error: 1.7435
#Mean Squared Error: 9.3187
#Root Mean Squared Error: 3.0526
#R² Score: 0.1461
#Max Error: 12.3634

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1567
#Mean Absolute Error: 1.7762
#Median Absolute Error: 1.4079
#Mean Squared Error: 5.3433
#Root Mean Squared Error: 2.3116
#R² Score: 0.1567
#Max Error: 8.7195

#3° Teste

regressor_xgb = XGBRegressor( max_depth = 500, 
                              learning_rate = 0.01,
                              min_child_weight = 20,
                              n_estimators = 2000 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1621
#Mean Absolute Error: 2.2833
#Median Absolute Error: 1.8002
#Mean Squared Error: 9.1493
#Root Mean Squared Error: 3.0248
#R² Score: 0.1616
#Max Error: 12.4911

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1875
#Mean Absolute Error: 1.7501
#Median Absolute Error: 1.3947
#Mean Squared Error: 5.1482
#Root Mean Squared Error: 2.2690
#R² Score: 0.1875
#Max Error: 8.6697

#4° Teste

regressor_xgb = XGBRegressor( max_depth = 50, 
                              learning_rate = 0.1,
                              min_child_weight = 20,
                              n_estimators = 2000 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1435
#Mean Absolute Error: 2.3140
#Median Absolute Error: 1.8381
#Mean Squared Error: 9.3550
#Root Mean Squared Error: 3.0586
#R² Score: 0.1428
#Max Error: 13.6268

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1729
#Mean Absolute Error: 1.7645
#Median Absolute Error: 1.3912
#Mean Squared Error: 5.2408
#Root Mean Squared Error: 2.2893
#R² Score: 0.1729
#Max Error: 8.9468

#5° Teste - Randomized Search

regressor_xgb = XGBRegressor( max_depth = 300, 
                              learning_rate = 0.05,
                              min_child_weight = 9,
                              n_estimators = 1000 )

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1791
#Mean Absolute Error: 1.7592
#Median Absolute Error: 1.3853
#Mean Squared Error: 5.2014
#Root Mean Squared Error: 2.2807
#R² Score: 0.1791
#Max Error: 8.2279

#6° Teste - Randomized Search - Melhor Resultado

regressor_xgb = XGBRegressor( n_estimators = 500,
                              min_child_weight = 70,
                              max_depth = 45,
                              learning_rate = 0.01 )

#TMP 10 Dias de Corte
#Explained Variance Score: 0.2041
#Mean Absolute Error: 1.7325
#Median Absolute Error: 1.4148
#Mean Squared Error: 5.0455
#Root Mean Squared Error: 2.2462
#R² Score: 0.2037
#Max Error: 8.4773

#7° Teste - Randomized Search

regressor_xgb = XGBRegressor( n_estimators = 400,
                              min_child_weight = 60,
                              max_depth = 50,
                              learning_rate = 0.01 )

#TMP 10 Dias de Corte
#Explained Variance Score: 0.2009
#Mean Absolute Error: 1.7329
#Median Absolute Error: 1.3886
#Mean Squared Error: 5.0694
#Root Mean Squared Error: 2.2515
#R² Score: 0.1999
#Max Error: 8.3941

#8° Teste - Randomized Search

regressor_xgb = XGBRegressor( n_estimators = 4000,
                              min_child_weight = 40,
                              max_depth = 450,
                              learning_rate = 0.001 )

#TMP 10 Dias de Corte
#Explained Variance Score: 0.2027
#Mean Absolute Error: 1.7274
#Median Absolute Error: 1.3805
#Mean Squared Error: 5.0594
#Root Mean Squared Error: 2.2493
#R² Score: 0.2015
#Max Error: 8.3463

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_xgb.get_params())

# Fit

import time
start = time.time()

regressor_xgb.fit(X_train_reg, Y_train_reg)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))

#Predição

Y_pred_train_reg_xgb = regressor_xgb.predict(X_train_reg)

Y_pred_test_reg_xgb = regressor_xgb.predict(X_test_reg)

#Análise de Métricas

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_xgb = explained_variance_score(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_mae_xgb = mean_absolute_error(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_mdae_xgb = median_absolute_error(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_mse_xgb = mean_squared_error(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_rmse_xgb = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_xgb))
mtrc_r2_xgb = r2_score(Y_test_reg, Y_pred_test_reg_xgb)
mtrc_max_error_xgb = max_error(Y_test_reg, Y_pred_test_reg_xgb)

print('Explained Variance Score: {:.4f}'.format(mtrc_evs_xgb))
print('Mean Absolute Error: {:.4f}'.format(mtrc_mae_xgb)) 
print('Median Absolute Error: {:.4f}'.format(mtrc_mdae_xgb)) 
print('Mean Squared Error: {:.4f}'.format(mtrc_mse_xgb))
print('Root Mean Squared Error: {:.4f}'.format(mtrc_rmse_xgb))
print('R² Score: {:.4f}'.format(mtrc_r2_xgb))
print('Max Error: {:.4f}'.format(mtrc_max_error_xgb))

#Plot - Y_test_reg vs Y_pred_test_reg_xgb

#Ideally Should Have Been a Straight Line
plt.figure(figsize = (14, 7)) 
plt.scatter(Y_test_reg, Y_pred_test_reg_xgb)
plt.show()

#Plot Residuals

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_xgb, Y_pred_train_reg_xgb - Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_xgb, Y_pred_test_reg_xgb - Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("XGBoost Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

#Plot Predictions

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_xgb, Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_xgb, Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("XGBoost Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Real Values")
plt.legend(loc = "upper left")
plt.plot([10.5, 13.5], [10.5, 13.5], c = "red")
plt.show()

#XGBoost Regressor Feature Importance

fea_imp = pd.DataFrame({'imp': regressor_xgb.feature_importances_, 'col': X_list_reg})
fea_imp = fea_imp.sort_values(['imp', 'col'], ascending = [True, False]).iloc[-30:]
fea_imp.plot(kind = 'barh', x = 'col', y = 'imp', figsize = (7, 7), legend = None)
plt.title('XGBoost - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance')

#Learning Curve - X_train_reg, Y_train_reg

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import learning_curve

import time

start = time.time()
train_sizes_lc_xgb, train_scores_lc_xgb, test_scores_lc_xgb = learning_curve( estimator = regressor_xgb,
                                                                              X = X_train_reg,
                                                                              y = Y_train_reg,
                                                                              train_sizes = np.linspace(0.1, 1.0, 5),
                                                                              cv = 10,
                                                                              scoring = 'neg_mean_squared_error',
                                                                              shuffle = True,
                                                                              random_state = 42,
                                                                              n_jobs = -1 )
end = time.time()
print("Tempo de Execução: {} sec".format(end - start))

train_mean_lc_xgb = np.sqrt(np.mean(train_scores_lc_xgb, axis = 1)*-1)
train_std_lc_xgb = np.sqrt(np.std(train_scores_lc_xgb, axis = 1)*-1)
test_mean_lc_xgb = np.sqrt(np.mean(test_scores_lc_xgb, axis = 1)*-1)
test_std_lc_xgb = np.sqrt(np.std(test_scores_lc_xgb, axis = 1)*-1)

plt.figure(figsize = (14, 7))
plt.plot( train_sizes_lc_xgb, 
          train_mean_lc_xgb,
          color = 'blue', 
          marker = 'o',
          markersize = 5, 
          label = 'Training RMSE' )

plt.fill_between( train_sizes_lc_xgb,
                  train_mean_lc_xgb + train_std_lc_xgb,
                  train_mean_lc_xgb - train_std_lc_xgb,
                  alpha = 0.15, 
                  color ='blue' )

plt.plot( train_sizes_lc_xgb, 
          test_mean_lc_xgb,
          color = 'green', 
          linestyle = '--',
          marker = 's', 
          markersize = 5,
          label = 'Validation RMSE')

plt.fill_between(train_sizes_lc_xgb,
                 test_mean_lc_xgb + test_std_lc_xgb,
                 test_mean_lc_xgb - test_std_lc_xgb,
                 alpha = 0.15, 
                 color = 'green')

plt.grid()
plt.xlabel('Number of Training Samples')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.legend(loc = 'lower right')
plt.ylim([1.0, 3.0])
plt.tight_layout()
#plt.savefig('images/06_05.png', dpi = 300)
plt.show()

#Validation Curve - X_train_reg, Y_train_reg

from sklearn.model_selection import validation_curve

import time

param_range_vc = [100, 200, 300, 400, 500, 1000, 2000, 3000, 4000, 5000]

start = time.time()
train_scores_vc_xgb, test_scores_vc_xgb = validation_curve( estimator = regressor_xgb, 
                                                            X = X_train_reg, 
                                                            y = Y_train_reg, 
                                                            param_name = 'n_estimators', 
                                                            param_range = param_range_vc,
                                                            scoring = 'neg_mean_squared_error',
                                                            cv = 10,
                                                            n_jobs = -1)
end = time.time()
print("Tempo de Execução: {} sec".format(end - start))

train_mean_vc_xgb = np.sqrt(np.mean(train_scores_vc_xgb, axis = 1)*-1)
train_std_vc_xgb = np.sqrt(np.std(train_scores_vc_xgb, axis = 1)*-1)
test_mean_vc_xgb = np.sqrt(np.mean(test_scores_vc_xgb, axis = 1)*-1)
test_std_vc_xgb = np.sqrt(np.std(test_scores_vc_xgb, axis = 1)*-1)

plt.figure(figsize = (14, 7))
plt.plot( param_range_vc, 
          train_mean_vc_xgb, 
          color = 'blue', 
          marker = 'o', 
          markersize = 5, 
          label = 'Training RMSE' )

plt.fill_between( param_range_vc, 
                  train_mean_vc_xgb + train_std_vc_xgb,
                  train_mean_vc_xgb - train_std_vc_xgb, 
                  alpha = 0.15,
                  color = 'blue' )

plt.plot( param_range_vc, 
          test_mean_vc_xgb, 
          color = 'green', 
          linestyle = '--', 
          marker = 's', 
          markersize = 5, 
          label = 'Validation RMSE' )

plt.fill_between( param_range_vc, 
                  test_mean_vc_xgb + test_std_vc_xgb,
                  test_mean_vc_xgb - test_std_vc_xgb, 
                  alpha = 0.15, 
                  color = 'green')

plt.grid()
#plt.xscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('Parameter n Estimator')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.ylim([0.0, 3.0])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi = 300)
plt.show()

#OTIMIZAÇÃO DE HIPERPARAMETROS - XGBOOST

#TESTE 01: Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor

#Criação das Variáveis dos Parâmetros a Serem Testados

#max_depth [default=6]: The maximum depth of a tree, same as GBM.
#Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
#Should be tuned using CV. Typical values: 3-10.
max_depth_xgb = [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)]

#learning_rate:
learning_rate_xgb = [0.01, 0.03, 0.05, 0.1, 0.15]

#min_child_weight [default=1]: Defines the minimum sum of weights of all observations required in a child.
#This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
#Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
#Too high values can lead to under-fitting hence, it should be tuned using CV.
min_child_weight_xgb = [int(x) for x in np.linspace(1, 9, num = 5)]

#n_estimators:
n_estimators_xgb = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Criação e Fit do Random Grid

random_grid_xgb = { 'max_depth': max_depth_xgb,
                    'learning_rate': learning_rate_xgb,
                    'min_child_weight': min_child_weight_xgb,
                    'n_estimators': n_estimators_xgb }

regressor_xgb_rscv = XGBRegressor()

regressor_xgb_rscv_random = RandomizedSearchCV( estimator = regressor_xgb_rscv, 
                                                param_distributions = random_grid_xgb, 
                                                n_iter = 20, 
                                                cv = 3, 
                                                verbose = 2, 
                                                random_state = 42,
                                                scoring = 'neg_mean_squared_error' )

import time
start = time.time()

regressor_xgb_rscv_random.fit(X_train_reg, Y_train_reg)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))

regressor_xgb_rscv_random.best_params_

#TMP 10 Dias de Corte
#{'n_estimators': 1000,
# 'min_child_weight': 9,
# 'max_depth': 300,
# 'learning_rate': 0.05}

regressor_xgb_rscv_random.best_score_

#TMP 10 Dias de Corte
#-5.379508637776577

regressor_xgb_rscv_random.best_estimator_

regressor_xgb_rscv_random.best_index_

#TESTE 02: Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor

#Criação das Variáveis dos Parâmetros a Serem Testados

#max_depth [default=6]: The maximum depth of a tree, same as GBM.
#Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
#Should be tuned using CV. Typical values: 3-10.
max_depth_xgb = [int(x) for x in np.linspace(start = 5, stop = 500, num = 100)]

#learning_rate:
learning_rate_xgb = [0.01, 0.03, 0.05, 0.1, 0.15]

#min_child_weight [default=1]: Defines the minimum sum of weights of all observations required in a child.
#This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
#Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
#Too high values can lead to under-fitting hence, it should be tuned using CV.
min_child_weight_xgb = [int(x) for x in np.linspace(10, 100, num = 10)]

#n_estimators:
n_estimators_xgb = [int(x) for x in np.linspace(start = 500, stop = 5000, num = 10)]

# Criação e Fit do Random Grid

random_grid_xgb = { 'max_depth': max_depth_xgb,
                    'learning_rate': learning_rate_xgb,
                    'min_child_weight': min_child_weight_xgb,
                    'n_estimators': n_estimators_xgb }

regressor_xgb_rscv = XGBRegressor()

regressor_xgb_rscv_random = RandomizedSearchCV( estimator = regressor_xgb_rscv, 
                                                param_distributions = random_grid_xgb, 
                                                n_iter = 30, 
                                                cv = 3, 
                                                verbose = 2, 
                                                random_state = 42,
                                                scoring = 'neg_mean_squared_error' )

import time

start = time.time()
regressor_xgb_rscv_random.fit(X_train_reg, Y_train_reg)
end = time.time()

print("Tempo de Execução: {} min".format((end - start)/60))

regressor_xgb_rscv_random.best_params_

#TMP 10 Dias de Corte
#{'n_estimators': 500,
# 'min_child_weight': 70,
# 'max_depth': 45,
# 'learning_rate': 0.01}

regressor_xgb_rscv_random.best_score_

#-5.134859484122991

regressor_xgb_rscv_random.best_estimator_

#TMP 10 Dias de Corte
#XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#             colsample_bynode=1, colsample_bytree=1, gamma=0,
#             importance_type='gain', learning_rate=0.01, max_delta_step=0,
#             max_depth=45, min_child_weight=70, missing=None, n_estimators=500,
#             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#             silent=None, subsample=1, verbosity=1)

regressor_xgb = regressor_xgb_rscv_random.best_estimator_

regressor_xgb_rscv_random.best_index_

#TESTE 03: Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor

#Criação das Variáveis dos Parâmetros a Serem Testados

#max_depth [default=6]: The maximum depth of a tree, same as GBM.
#Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
#Should be tuned using CV. Typical values: 3-10.
max_depth_xgb = [30, 40, 45, 50, 60]

#learning_rate:
learning_rate_xgb = [0.01, 0.02, 0.03]

#min_child_weight [default=1]: Defines the minimum sum of weights of all observations required in a child.
#This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
#Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
#Too high values can lead to under-fitting hence, it should be tuned using CV.
min_child_weight_xgb = [60, 70, 80]

#n_estimators:
n_estimators_xgb = [400, 500, 600]

# Criação e Fit do Random Grid

random_grid_xgb = { 'max_depth': max_depth_xgb,
                    'learning_rate': learning_rate_xgb,
                    'min_child_weight': min_child_weight_xgb,
                    'n_estimators': n_estimators_xgb }

regressor_xgb_rscv = XGBRegressor()

regressor_xgb_rscv_random = RandomizedSearchCV( estimator = regressor_xgb_rscv, 
                                                param_distributions = random_grid_xgb, 
                                                n_iter = 90, 
                                                cv = 3,
                                                verbose = 2,
                                                random_state = 42,
                                                scoring = 'neg_mean_squared_error',
                                                n_jobs = -1 )

import time

start = time.time()
regressor_xgb_rscv_random.fit(X_train_reg, Y_train_reg)
end = time.time()

print("Tempo de Execução: {} min".format((end - start)/60))

regressor_xgb_rscv_random.best_params_

#TMP 10 Dias de Corte
#{'n_estimators': 400,
# 'min_child_weight': 60,
# 'max_depth': 50,
# 'learning_rate': 0.01}

regressor_xgb_rscv_random.best_score_

#TMP 10 Dias de Corte
#-5.1208284637229715

regressor_xgb_rscv_random.best_estimator_

#TMP 10 Dias de Corte
#XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#             colsample_bynode=1, colsample_bytree=1, gamma=0,
#             importance_type='gain', learning_rate=0.01, max_delta_step=0,
#             max_depth=50, min_child_weight=60, missing=None, n_estimators=400,
#             n_jobs=1, nthread=None, objective='reg:linear', random_state=0,
#             reg_alpha=0, reg_lambda=1, scale_pos_weight=1, seed=None,
#             silent=None, subsample=1, verbosity=1)

regressor_xgb = regressor_xgb_rscv_random.best_estimator_

regressor_xgb_rscv_random.best_index_

#TESTE 04: Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBRegressor

#Criação das Variáveis dos Parâmetros a Serem Testados

#max_depth [default=6]: The maximum depth of a tree, same as GBM.
#Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
#Should be tuned using CV. Typical values: 3-10.
max_depth_xgb = [300, 400, 450, 500, 600]

#learning_rate:
learning_rate_xgb = [0.001, 0.01, 0.02, 0.03, 0.05, 0.1]

#min_child_weight [default=1]: Defines the minimum sum of weights of all observations required in a child.
#This is similar to min_child_leaf in GBM but not exactly. This refers to min “sum of weights” of observations while GBM has min “number of observations”.
#Used to control over-fitting. Higher values prevent a model from learning relations which might be highly specific to the particular sample selected for a tree.
#Too high values can lead to under-fitting hence, it should be tuned using CV.
min_child_weight_xgb = [20, 30, 40, 50, 60, 70, 80]

#n_estimators:
n_estimators_xgb = [3000, 4000, 5000, 6000]

# Criação e Fit do Random Grid

random_grid_xgb = { 'max_depth': max_depth_xgb,
                    'learning_rate': learning_rate_xgb,
                    'min_child_weight': min_child_weight_xgb,
                    'n_estimators': n_estimators_xgb }

regressor_xgb_rscv = XGBRegressor()

regressor_xgb_rscv_random = RandomizedSearchCV( estimator = regressor_xgb_rscv, 
                                                param_distributions = random_grid_xgb, 
                                                n_iter = 20, 
                                                cv = 3,
                                                verbose = 2,
                                                random_state = 42,
                                                scoring = 'neg_mean_squared_error',
                                                n_jobs = -1 )

import time

start = time.time()
regressor_xgb_rscv_random.fit(X_train_reg, Y_train_reg)
end = time.time()

print("Tempo de Execução: {} min".format((end - start)/60))

regressor_xgb_rscv_random.best_params_

#TMP 10 Dias de Corte
#{'n_estimators': 4000,
# 'min_child_weight': 40,
# 'max_depth': 450,
# 'learning_rate': 0.001}

regressor_xgb_rscv_random.best_score_

#TMP 10 Dias de Corte
#-5.140922071630635

regressor_xgb_rscv_random.best_estimator_

#XGBRegressor(base_score=0.5, booster='gbtree', colsample_bylevel=1,
#             colsample_bynode=1, colsample_bytree=1, gamma=0,
#             importance_type='gain', learning_rate=0.001, max_delta_step=0,
#             max_depth=450, min_child_weight=40, missing=None,
#             n_estimators=4000, n_jobs=1, nthread=None, objective='reg:linear',
#             random_state=0, reg_alpha=0, reg_lambda=1, scale_pos_weight=1,
#             seed=None, silent=None, subsample=1, verbosity=1)

regressor_xgb = regressor_xgb_rscv_random.best_estimator_

regressor_xgb_rscv_random.best_index_

#ETAPA 23: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO LIGHT GBM

#Parametrização do Modelo

from lightgbm import LGBMRegressor

#1° Teste

regressor_lgbm = LGBMRegressor()

#LightGBM Model Training
#num_leaves is the main parameter to control the complexity of the tree model. when trying to tune the num_leaves, we should let it be smaller than 2^(max_depth) (225).
#We use max_depth to limit growing deep tree.
#for better accuracy, we us small learning_rate with large num_iterations.
#To speed up training and deal with overfitting, we set feature_fraction=0.6, that is, selecting 60% features before training each tree.
#Set verbosity = -1, eval metric on the eval set is printed at every verbose boosting stage.
#early_stopping_rounds = 500, the model will train until the validation score stops improving. Validation score needs to improve at least every 500 round(s) to continue training.
#verbose_eval = 500, an evaluation metric is printed every 500 boosting stages.

#2° Teste

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

#3° Teste

regressor_lgbm = LGBMRegressor ( objective = 'regression',
                                 metric = 'RMSE',
                                 num_leaves = 200,
                                 max_depth = 15,
                                 learning_rate = 0.01,
                                 feature_fraction = 0.6,
                                 verbosity = -1,
                                 num_iterations = 20000,
                                 verbose_eval = 500 )

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_lgbm.get_params())

# Fit

import time

start = time.time()

regressor_lgbm.fit(X_train_reg, Y_train_reg)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

#Predição

Y_pred_train_reg_lgbm = regressor_lgbm.predict(X_train_reg)

Y_pred_test_reg_lgbm = regressor_lgbm.predict(X_test_reg)

#Análise de Métricas

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_lgbm = explained_variance_score(Y_test_reg, Y_pred_test_reg_lgbm)
mtrc_mae_lgbm = mean_absolute_error(Y_test_reg, Y_pred_test_reg_lgbm)
mtrc_mdae_lgbm = median_absolute_error(Y_test_reg, Y_pred_test_reg_lgbm)
mtrc_mse_lgbm = mean_squared_error(Y_test_reg, Y_pred_test_reg_lgbm)
mtrc_rmse_lgbm = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_lgbm))
mtrc_r2_lgbm = r2_score(Y_test_reg, Y_pred_test_reg_lgbm)
mtrc_max_error_lgbm = max_error(Y_test_reg, Y_pred_test_reg_lgbm)

print('Explained Variance Score: {:.4f}'.format(mtrc_evs_lgbm))
print('Mean Absolute Error: {:.4f}'.format(mtrc_mae_lgbm)) 
print('Median Absolute Error: {:.4f}'.format(mtrc_mdae_lgbm)) 
print('Mean Squared Error: {:.4f}'.format(mtrc_mse_lgbm))
print('Root Mean Squared Error: {:.4f}'.format(mtrc_rmse_lgbm))
print('R² Score: {:.4f}'.format(mtrc_r2_lgbm))
print('Max Error: {:.4f}'.format(mtrc_max_error_lgbm))

#Plot - Y_test_reg vs Y_pred_test_reg_lgbm

#Ideally Should Have Been a Straight Line
plt.figure(figsize = (14, 7)) 
plt.scatter(Y_test_reg, Y_pred_test_reg_lgbm)
plt.show()

#Plot Residuals

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_lgbm, Y_pred_train_reg_lgbm - Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_lgbm, Y_pred_test_reg_lgbm - Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("LightGBM Regressor")
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.legend(loc = "upper left")
plt.hlines(y = 0, xmin = 10.5, xmax = 13.5, color = "red")
plt.show()

#Plot Predictions

plt.figure(figsize = (14, 7)) 
plt.scatter(Y_pred_train_reg_lgbm, Y_train_reg, c = "blue", marker = "s", label = "Training Data")
plt.scatter(Y_pred_test_reg_lgbm, Y_test_reg, c = "lightgreen", marker = "s", label = "Validation Data")
plt.title("LightGBM Regressor")
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

#Parametrização do Modelo

from catboost import CatBoostRegressor

#1° Teste

regressor_cbr = CatBoostRegressor()

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1963
#Mean Absolute Error: 2.2343
#Median Absolute Error: 1.7361
#Mean Squared Error: 8.7730
#Root Mean Squared Error: 2.9619
#R² Score: 0.1961
#Max Error: 12.4565

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1938
#Mean Absolute Error: 1.7471
#Median Absolute Error: 1.3996
#Mean Squared Error: 5.1093
#Root Mean Squared Error: 2.2604
#R² Score: 0.1936
#Max Error: 8.6229

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

#2° Teste

regressor_cbr = CatBoostRegressor( iterations = 50, 
                                   depth = 3, 
                                   learning_rate = 0.1, 
                                   loss_function = 'RMSE',
                                   random_seed = seed_reg )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1692
#Mean Absolute Error: 2.2936
#Median Absolute Error: 1.8183
#Mean Squared Error: 9.0670
#Root Mean Squared Error: 3.0111
#R² Score: 0.1692
#Max Error: 12.3931

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1675
#Mean Absolute Error: 1.8082
#Median Absolute Error: 1.4763
#Mean Squared Error: 5.2766
#Root Mean Squared Error: 2.2971
#R² Score: 0.1672
#Max Error: 8.1909

#3° Teste

regressor_cbr = CatBoostRegressor( iterations = 2000,
                                   learning_rate = 0.01,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.2,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1952
#Mean Absolute Error: 2.2322
#Median Absolute Error: 1.7602
#Mean Squared Error: 8.7829
#Root Mean Squared Error: 2.9636
#Mean Squared Logarithmic Error: 0.3952
#R² Score: 0.1952
#Max Error: 12.4764

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1971
#Mean Absolute Error: 1.7499
#Median Absolute Error: 1.4009
#Mean Squared Error: 5.0893
#Root Mean Squared Error: 2.2559
#R² Score: 0.1968
#Max Error: 8.7948

#4° Teste

regressor_cbr = CatBoostRegressor( iterations = 2000,
                                   learning_rate = 0.1,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.2,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1614
#Mean Absolute Error: 2.2675
#Median Absolute Error: 1.7581
#Mean Squared Error: 9.1537
#Root Mean Squared Error: 3.0255
#R² Score: 0.1612
#Max Error: 12.5788

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1666
#Mean Absolute Error: 1.7698
#Median Absolute Error: 1.4024
#Mean Squared Error: 5.2813
#Root Mean Squared Error: 2.2981
#R² Score: 0.1665
#Max Error: 8.6035

#5° Teste

regressor_cbr = CatBoostRegressor( iterations = 2000,
                                   learning_rate = 0.1,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1498
#Mean Absolute Error: 2.2782
#Median Absolute Error: 1.7407
#Mean Squared Error: 9.2811
#Root Mean Squared Error: 3.0465
#R² Score: 0.1495
#Max Error: 12.6518

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1723
#Mean Absolute Error: 1.7632
#Median Absolute Error: 1.3727
#Mean Squared Error: 5.2463
#Root Mean Squared Error: 2.2905
#R² Score: 0.1720
#Max Error: 9.0140

#6° Teste - 10 Dias

regressor_cbr = CatBoostRegressor( iterations = 4000,
                                   learning_rate = 0.01,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.2,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1945
#Mean Absolute Error: 2.2288
#Median Absolute Error: 1.7153
#Mean Squared Error: 8.7915
#Root Mean Squared Error: 2.9651
#R² Score: 0.1944
#Max Error: 12.5005

#TMP 10 Dias de Corte
#Explained Variance Score: 0.2009
#Mean Absolute Error: 1.7412
#Median Absolute Error: 1.3808
#Mean Squared Error: 5.0654
#Root Mean Squared Error: 2.2507
#R² Score: 0.2006
#Max Error: 8.8925

#7° Teste

regressor_cbr = CatBoostRegressor( iterations = 4000,
                                   learning_rate = 0.01,
                                   depth = 5,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.2,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1875
#Mean Absolute Error: 2.2428
#Median Absolute Error: 1.7511
#Mean Squared Error: 8.8681
#Root Mean Squared Error: 2.9779
#R² Score: 0.1874
#Max Error: 12.4920

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1918
#Mean Absolute Error: 1.7544
#Median Absolute Error: 1.3994
#Mean Squared Error: 5.1215
#Root Mean Squared Error: 2.2631
#R² Score: 0.1917
#Max Error: 8.8706

#8° Teste

regressor_cbr = CatBoostRegressor( iterations = 10000,
                                   learning_rate = 0.01,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.2,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1822
#Mean Absolute Error: 2.2361
#Median Absolute Error: 1.7234
#Mean Squared Error: 8.9270
#Root Mean Squared Error: 2.9878
#R² Score: 0.1820
#Max Error: 12.5372

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1967
#Mean Absolute Error: 1.7396
#Median Absolute Error: 1.3697
#Mean Squared Error: 5.0913
#Root Mean Squared Error: 2.2564
#R² Score: 0.1965
#Max Error: 8.9128

#9° Teste

regressor_cbr = CatBoostRegressor( iterations = 2000,
                                   learning_rate = 0.01,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.1,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1972
#Mean Absolute Error: 2.2306
#Median Absolute Error: 1.7585
#Mean Squared Error: 8.7615
#Root Mean Squared Error: 2.9600
#R² Score: 0.1971
#Max Error: 12.3811

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1979
#Mean Absolute Error: 1.7486
#Median Absolute Error: 1.4096
#Mean Squared Error: 5.0845
#Root Mean Squared Error: 2.2549
#R² Score: 0.1975
#Max Error: 8.7870

#10° Teste

regressor_cbr = CatBoostRegressor( learning_rate = 0.01,
                                   l2_leaf_reg = 9,
                                   iterations = 1000,
                                   depth = 9,
                                   bagging_temperature = 0.3 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1917
#Mean Absolute Error: 2.2464
#Median Absolute Error: 1.7699
#Mean Squared Error: 8.8216
#Root Mean Squared Error: 2.9701
#R² Score: 0.1916
#Max Error: 12.4459

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1964
#Mean Absolute Error: 1.7574
#Median Absolute Error: 1.4158
#Mean Squared Error: 5.0937
#Root Mean Squared Error: 2.2569
#R² Score: 0.1961
#Max Error: 8.6291

#11° Teste

regressor_cbr = CatBoostRegressor( iterations = 2000,
                                   learning_rate = 0.01,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.01,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1975
#Mean Absolute Error: 2.2322
#Median Absolute Error: 1.7468
#Mean Squared Error: 8.7583
#Root Mean Squared Error: 2.9594
#R² Score: 0.1974
#Max Error: 12.3290

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1946
#Mean Absolute Error: 1.7535
#Median Absolute Error: 1.4017
#Mean Squared Error: 5.1051
#Root Mean Squared Error: 2.2595
#R² Score: 0.1943
#Max Error: 8.7744

#12° Teste - 15 Dias

regressor_cbr = CatBoostRegressor( iterations = 2000,
                                   learning_rate = 0.01,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.001,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1980
#Mean Absolute Error: 2.2330
#Median Absolute Error: 1.7347
#Mean Squared Error: 8.7525
#Root Mean Squared Error: 2.9585
#R² Score: 0.1980
#Max Error: 12.3913

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1956
#Mean Absolute Error: 1.7499
#Median Absolute Error: 1.3986
#Mean Squared Error: 5.0991
#Root Mean Squared Error: 2.2581
#R² Score: 0.1952
#Max Error: 8.7362

#13° Teste

regressor_cbr = CatBoostRegressor( iterations = 2000,
                                   learning_rate = 0.01,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.001,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 50 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1980
#Mean Absolute Error: 2.2330
#Median Absolute Error: 1.7347
#Mean Squared Error: 8.7525
#Root Mean Squared Error: 2.9585
#R² Score: 0.1980
#Max Error: 12.3913

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1956
#Mean Absolute Error: 1.7499
#Median Absolute Error: 1.3986
#Mean Squared Error: 5.0991
#Root Mean Squared Error: 2.2581
#R² Score: 0.1952
#Max Error: 8.7362

#14° Teste

regressor_cbr = CatBoostRegressor( iterations = 2000,
                                   learning_rate = 0.01,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.001,
                                   od_type = 'Iter',
                                   metric_period = 25,
                                   od_wait = 50 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1980
#Mean Absolute Error: 2.2330
#Median Absolute Error: 1.7347
#Mean Squared Error: 8.7525
#Root Mean Squared Error: 2.9585
#R² Score: 0.1980
#Max Error: 12.3913

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1956
#Mean Absolute Error: 1.7499
#Median Absolute Error: 1.3986
#Mean Squared Error: 5.0991
#Root Mean Squared Error: 2.2581
#R² Score: 0.1952
#Max Error: 8.7362

#15° Teste

regressor_cbr = CatBoostRegressor( iterations = 2000,
                                   learning_rate = 0.01,
                                   depth = 8,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.001,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1975
#Mean Absolute Error: 2.2337
#Median Absolute Error: 1.7398
#Mean Squared Error: 8.7583
#Root Mean Squared Error: 2.9595
#R² Score: 0.1974
#Max Error: 12.3645

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1976
#Mean Absolute Error: 1.7503
#Median Absolute Error: 1.4267
#Mean Squared Error: 5.0859
#Root Mean Squared Error: 2.2552
#R² Score: 0.1973
#Max Error: 8.7382

#16° Teste

regressor_cbr = CatBoostRegressor( iterations = 4000,
                                   learning_rate = 0.01,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.001,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1945
#Mean Absolute Error: 2.2319
#Median Absolute Error: 1.7382
#Mean Squared Error: 8.7921
#Root Mean Squared Error: 2.9651
#R² Score: 0.1943
#Max Error: 12.3390

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1984
#Mean Absolute Error: 1.7420
#Median Absolute Error: 1.3819
#Mean Squared Error: 5.0808
#Root Mean Squared Error: 2.2541
#R² Score: 0.1981
#Max Error: 8.7662

#17° Teste

regressor_cbr = CatBoostRegressor( iterations = 1000,
                                   learning_rate = 0.01,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.001,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1956
#Mean Absolute Error: 2.2390
#Median Absolute Error: 1.7574
#Mean Squared Error: 8.7785
#Root Mean Squared Error: 2.9629
#R² Score: 0.1956
#Max Error: 12.2653

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1990
#Mean Absolute Error: 1.7529
#Median Absolute Error: 1.4144
#Mean Squared Error: 5.0775
#Root Mean Squared Error: 2.2533
#R² Score: 0.1987
#Max Error: 8.6568

#18° Teste

regressor_cbr = CatBoostRegressor( iterations = 1500,
                                   learning_rate = 0.01,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.001,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1973
#Mean Absolute Error: 2.2342
#Median Absolute Error: 1.7448
#Mean Squared Error: 8.7597
#Root Mean Squared Error: 2.9597
#R² Score: 0.1973
#Max Error: 12.3298

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1964
#Mean Absolute Error: 1.7518
#Median Absolute Error: 1.4086
#Mean Squared Error: 5.0942
#Root Mean Squared Error: 2.2570
#R² Score: 0.1960
#Max Error: 8.6824


#19° Teste

regressor_cbr = CatBoostRegressor( iterations = 2000,
                                   learning_rate = 0.001,
                                   depth = 10,
                                   eval_metric = 'RMSE',
                                   random_seed = seed_reg,
                                   bagging_temperature = 0.001,
                                   od_type = 'Iter',
                                   metric_period = 75,
                                   od_wait = 100 )

#TMP 15 Dias de Corte
#Explained Variance Score: 0.1429
#Mean Absolute Error: 2.2437
#Median Absolute Error: 1.6444
#Mean Squared Error: 9.6405
#Root Mean Squared Error: 3.1049
#R² Score: 0.1166
#Max Error: 12.7436

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1376
#Mean Absolute Error: 1.7985
#Median Absolute Error: 1.3607
#Mean Squared Error: 5.7115
#Root Mean Squared Error: 2.3899
#R² Score: 0.0986
#Max Error: 8.2665

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_cbr.get_params())

# Fit

import time

start = time.time()

regressor_cbr.fit(X_train_reg, Y_train_reg)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

#Predição

Y_pred_train_reg_cbr = regressor_cbr.predict(X_train_reg)

Y_pred_test_reg_cbr = regressor_cbr.predict(X_test_reg)

#Análise de Métricas

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_cbr = explained_variance_score(Y_test_reg, Y_pred_test_reg_cbr)
mtrc_mae_cbr = mean_absolute_error(Y_test_reg, Y_pred_test_reg_cbr)
mtrc_mdae_cbr = median_absolute_error(Y_test_reg, Y_pred_test_reg_cbr)
mtrc_mse_cbr = mean_squared_error(Y_test_reg, Y_pred_test_reg_cbr)
mtrc_rmse_cbr = np.sqrt(mean_squared_error(Y_test_reg, Y_pred_test_reg_cbr))
mtrc_r2_cbr = r2_score(Y_test_reg, Y_pred_test_reg_cbr)
mtrc_max_error_cbr = max_error(Y_test_reg, Y_pred_test_reg_cbr)

print('Explained Variance Score: {:.4f}'.format(mtrc_evs_cbr))
print('Mean Absolute Error: {:.4f}'.format(mtrc_mae_cbr)) 
print('Median Absolute Error: {:.4f}'.format(mtrc_mdae_cbr)) 
print('Mean Squared Error: {:.4f}'.format(mtrc_mse_cbr))
print('Root Mean Squared Error: {:.4f}'.format(mtrc_rmse_cbr))
print('R² Score: {:.4f}'.format(mtrc_r2_cbr))
print('Max Error: {:.4f}'.format(mtrc_max_error_cbr))

#Plot - Y_test_reg vs Y_pred_test_reg_cbr

import matplotlib.pyplot as plt

#Ideally Should Have Been a Straight Line
plt.figure(figsize = (14, 7)) 
plt.scatter(Y_test_reg, Y_pred_test_reg_cbr)
plt.show()

#Plot Residuals

import matplotlib.pyplot as plt

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

import matplotlib.pyplot as plt

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
fea_imp.plot(kind = 'barh', x = 'col', y = 'imp', figsize = (7, 7), legend = None)
plt.title('CatBoost - Feature Importance')
plt.ylabel('Features')
plt.xlabel('Importance')

#OTIMIZAÇÃO DE HIPERPARAMETROS - CATBOOST

#TESTE 01: Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error

#Criação das Variáveis dos Parâmetros a Serem Testados

#depth: Defines the depth of the trees.
depth_cbc = [int(x) for x in np.linspace(start = 1, stop = 9, num = 5)]

#iterations: The maximum number of trees that can be built.
# The final number of trees may be less than or equal to this number.
iterations_cbc = [int(x) for x in np.linspace(1000, 10000, num = 10)]

#l2_leaf_reg_cbc
l2_leaf_reg_cbc = [int(x) for x in np.linspace(1, 9, num = 5)]

#learning_rate: Defines the learning rate.
#Used for reducing the gradient step.
learning_rate_cbc = [0.01, 0.03, 0.05, 0.1, 0.15]

#Defines the settings of the Bayesian bootstrap. It is used by default in classification and regression modes.
#Use the Bayesian bootstrap to assign random weights to objects.
#The weights are sampled from exponential distribution if the value of this parameter is set to “1”. 
#All weights are equal to 1 if the value of this parameter is set to “0”.
bagging_temperature_cbc = [0.01, 0.03, 0.05, 0.1, 0.2, 0.3]

# Criação e Fit do Random Grid

random_grid_cbc = { 'depth': depth_cbc,
                    'iterations': iterations_cbc,
                    'l2_leaf_reg': l2_leaf_reg_cbc,
                    'learning_rate': learning_rate_cbc,
                    'bagging_temperature': bagging_temperature_cbc }

regressor_cbc_rscv = CatBoostRegressor()

regressor_cbc_rscv_random = RandomizedSearchCV( estimator = regressor_cbc_rscv, 
                                                param_distributions = random_grid_cbc, 
                                                n_iter = 20, 
                                                cv = 3, 
                                                verbose = 2, 
                                                random_state = 42,
                                                scoring = 'neg_mean_squared_error' )

import time
start = time.time()

regressor_cbc_rscv_random.fit(X_train_reg, Y_train_reg)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))

regressor_cbc_rscv_random.best_params_

#TMP 15 Dias de Corte
#{'learning_rate': 0.01,
# 'l2_leaf_reg': 9,
# 'iterations': 1000,
# 'depth': 9,
# 'bagging_temperature': 0.3}

#TMP 10 Dias de Corte
#{'learning_rate': 0.01,
# 'l2_leaf_reg': 9,
# 'iterations': 1000,
# 'depth': 9,
# 'bagging_temperature': 0.3}

regressor_cbc_rscv_random.best_score_

#TMP 15 Dias de Corte
#-9.186041096704711

#TMP 10 Dias de Corte
#-5.138291665087496

regressor_cbc_rscv_random.best_estimator_

regressor_cbr = regressor_cbc_rscv_random.best_estimator_

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1964
#Mean Absolute Error: 1.7574
#Median Absolute Error: 1.4158
#Mean Squared Error: 5.0937
#Root Mean Squared Error: 2.2569
#R² Score: 0.1961
#Max Error: 8.6291

regressor_cbc_rscv_random.best_index_

#TESTE 02: Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from catboost import CatBoostRegressor

from sklearn.metrics import mean_squared_error

#Criação das Variáveis dos Parâmetros a Serem Testados

#depth: Defines the depth of the trees.
depth_cbc = [int(x) for x in np.linspace(start = 2, stop = 10, num = 5)]

#iterations: The maximum number of trees that can be built.
# The final number of trees may be less than or equal to this number.
iterations_cbc = [int(x) for x in np.linspace(1000, 10000, num = 10)]

#l2_leaf_reg_cbc
l2_leaf_reg_cbc = [int(x) for x in np.linspace(1, 9, num = 5)]

#learning_rate: Defines the learning rate.
#Used for reducing the gradient step.
learning_rate_cbc = [0.001, 0.01, 0.03, 0.05, 0.1]

#Defines the settings of the Bayesian bootstrap. It is used by default in classification and regression modes.
#Use the Bayesian bootstrap to assign random weights to objects.
#The weights are sampled from exponential distribution if the value of this parameter is set to “1”. 
#All weights are equal to 1 if the value of this parameter is set to “0”.
bagging_temperature_cbc = [0.001, 0.01, 0.03, 0.05, 0.1]

# Criação e Fit do Random Grid

random_grid_cbc = { 'depth': depth_cbc,
                    'iterations': iterations_cbc,
                    'l2_leaf_reg': l2_leaf_reg_cbc,
                    'learning_rate': learning_rate_cbc,
                    'bagging_temperature': bagging_temperature_cbc }

regressor_cbc_rscv = CatBoostRegressor()

regressor_cbc_rscv_random = RandomizedSearchCV( estimator = regressor_cbc_rscv, 
                                                param_distributions = random_grid_cbc, 
                                                n_iter = 60, 
                                                cv = 3, 
                                                verbose = 2, 
                                                random_state = 42,
                                                scoring = 'neg_mean_squared_error' )

import time
start = time.time()

regressor_cbc_rscv_random.fit(X_train_reg, Y_train_reg)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))

regressor_cbc_rscv_random.best_params_

#TMP 15 Dias de Corte
#{'learning_rate': 0.001,
# 'l2_leaf_reg': 5,
# 'iterations': 10000,
# 'depth': 10,
# 'bagging_temperature': 0.1}

#TMP 10 Dias de Corte
#{'learning_rate': 0.001,
# 'l2_leaf_reg': 5,
# 'iterations': 10000,
# 'depth': 10,
# 'bagging_temperature': 0.1}

regressor_cbc_rscv_random.best_score_

#TMP 15 Dias de Corte
#-9.187495322442704

#TMP 10 Dias de Corte
#-5.1349953320184385

regressor_cbc_rscv_random.best_estimator_

regressor_cbr = regressor_cbc_rscv_random.best_estimator_

#TMP 10 Dias de Corte
#Explained Variance Score: 0.1968
#Mean Absolute Error: 1.7578
#Median Absolute Error: 1.4216
#Mean Squared Error: 5.0912
#Root Mean Squared Error: 2.2564
#R² Score: 0.1965
#Max Error: 8.5824

regressor_cbc_rscv_random.best_index_

#ETAPA 25: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO MLPREGRESSOR

#Parametrização do Modelo

from sklearn.neural_network import MLPRegressor

#1° Teste

regressor_mlp = MLPRegressor()

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_mlp.get_params())