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
dfr = pd.read_excel('TB_IND_ASSIST_EME_0115_0819_V6.xlsx')
end = time.time()

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 12.79 min

dfr_temp = dfr.loc[(dfr['CONT_STATUS_INT'] == 1)].copy()

del dfr

#Análise Descritiva da Variável Target para Detecção de Outliers

dfr_temp['TEMPO_PERM_INT_POSTERIOR'].describe()
#count    10774.000000
#mean         5.562187
#std          8.340397
#min          0.000000
#25%          1.000000
#50%          3.000000
#75%          6.000000
#max        127.000000

dfr_temp['TEMPO_PERM_INT_POSTERIOR'].median()
#3.0

dfr_temp['TEMPO_PERM_INT_POSTERIOR'].var()
#69.56222360347789

dfr_temp['TEMPO_PERM_INT_POSTERIOR'].mad()
#4.705802189535003

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
#(9413, 652)

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

del df_reg

del df_reg_trat

for col in X_reg.columns:
    if X_reg[col].isnull().sum() > 0:
        print(col)

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

X_reg.shape

Y_reg.shape

#Escalonamento da Base X_reg - Opção 01 - StandardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_reg = scaler.fit_transform(X_reg)

#ETAPA 22: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO XGBOOST

#Parametrização do Modelo

from xgboost import XGBRegressor

#6° Teste - Novo - Best Result

regressor_xgb = XGBRegressor( n_estimators = 500,
                              min_child_weight = 70,
                              max_depth = 45,
                              learning_rate = 0.01,
                              n_jobs = 6 )

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(regressor_xgb.get_params())

# Fit

import time
start = time.time()

regressor_xgb.fit(X_reg, Y_reg)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 2.53 min

#Learning Curve - RMSE - X_reg, Y_reg

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import learning_curve

import time

start = time.time()
train_sizes_lc_xgb, train_scores_lc_xgb, test_scores_lc_xgb = learning_curve( estimator = regressor_xgb,
                                                                              X = X_reg,
                                                                              y = Y_reg,
                                                                              train_sizes = np.linspace(0.1, 1.0, 10),
                                                                              cv = 10,
                                                                              scoring = 'neg_mean_squared_error',
                                                                              shuffle = True,
                                                                              random_state = 42,
                                                                              n_jobs = 1 )
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 107.41 min

train_mean_lc_xgb = np.sqrt(np.mean(train_scores_lc_xgb, axis = 1)*-1)
train_std_lc_xgb = np.sqrt(np.std(train_scores_lc_xgb*-1, axis = 1))
test_mean_lc_xgb = np.sqrt(np.mean(test_scores_lc_xgb, axis = 1)*-1)
test_std_lc_xgb = np.sqrt(np.std(test_scores_lc_xgb*-1, axis = 1))

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

plt.minorticks_on()
plt.grid(b = True, which = 'both', color = '#666666', linestyle = '-', alpha = 0.2, axis = 'both', linewidth = 1)

#plt.grid()
plt.xlabel('Number of Training Samples')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.legend(loc = 'lower right')
plt.ylim([1.0, 3.0])
plt.tight_layout()
#plt.savefig('images/06_05.png', dpi = 300)
plt.show()

#Learning Curve - R2 SCORE - X_reg, Y_reg

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import learning_curve

import time

start = time.time()
train_sizes_lc_xgb, train_scores_lc_xgb, test_scores_lc_xgb = learning_curve( estimator = regressor_xgb,
                                                                              X = X_reg,
                                                                              y = Y_reg,
                                                                              train_sizes = np.linspace(0.1, 1.0, 10),
                                                                              cv = 10,
                                                                              scoring = 'r2',
                                                                              shuffle = True,
                                                                              random_state = 42,
                                                                              n_jobs = 1 )
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 115.68 min

train_mean_lc_xgb = np.mean(train_scores_lc_xgb, axis = 1)
train_std_lc_xgb = np.std(train_scores_lc_xgb, axis = 1)
test_mean_lc_xgb = np.mean(test_scores_lc_xgb, axis = 1)
test_std_lc_xgb = np.std(test_scores_lc_xgb, axis = 1)

plt.figure(figsize = (14, 7))
plt.plot( train_sizes_lc_xgb, 
          train_mean_lc_xgb,
          color = 'blue', 
          marker = 'o',
          markersize = 5, 
          label = 'Training R2 Score' )

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
          label = 'Validation R2 Score')

plt.fill_between(train_sizes_lc_xgb,
                 test_mean_lc_xgb + test_std_lc_xgb,
                 test_mean_lc_xgb - test_std_lc_xgb,
                 alpha = 0.15, 
                 color = 'green')

plt.minorticks_on()
plt.grid(b = True, which = 'both', color = '#666666', linestyle = '-', alpha = 0.2, axis = 'both', linewidth = 1)

#plt.grid()
plt.xlabel('Number of Training Samples')
plt.ylabel('R2 Score')
plt.legend(loc = 'lower right')
plt.ylim([0.0, 1.0])
plt.tight_layout()
#plt.savefig('images/06_05.png', dpi = 300)
plt.show()

#Validation Curve - RMSE - X_reg, Y_reg

from sklearn.model_selection import validation_curve

import time

param_range_vc = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

start = time.time()
train_scores_vc_xgb, test_scores_vc_xgb = validation_curve( estimator = regressor_xgb, 
                                                            X = X_reg, 
                                                            y = Y_reg, 
                                                            param_name = 'n_estimators', 
                                                            param_range = param_range_vc,
                                                            scoring = 'neg_mean_squared_error',
                                                            cv = 10,
                                                            n_jobs = 1)
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 219.00 min

train_mean_vc_xgb = np.sqrt(np.mean(train_scores_vc_xgb, axis = 1)*-1)
train_std_vc_xgb = np.sqrt(np.std(train_scores_vc_xgb*-1, axis = 1))
test_mean_vc_xgb = np.sqrt(np.mean(test_scores_vc_xgb, axis = 1)*-1)
test_std_vc_xgb = np.sqrt(np.std(test_scores_vc_xgb*-1, axis = 1))

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

plt.minorticks_on()
plt.grid(b = True, which = 'both', color = '#666666', linestyle = '-', alpha = 0.2, axis = 'both', linewidth = 1)

#plt.grid()
#plt.grid(b = True, which = 'both', color = 'b', linestyle = '-', alpha = 0.2)
#plt.grid(b = True, which = 'major', color = 'b', linestyle = '-')
#plt.xscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('Parameter n Estimator')
plt.ylabel('Root Mean Squared Error (RMSE)')
plt.ylim([0.0, 3.0])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi = 300)
plt.show()

#Validation Curve - R2 SCORE - X_reg, Y_reg

from sklearn.model_selection import validation_curve

import time

param_range_vc = [200, 300, 400, 500, 600, 700, 800, 900, 1000]

start = time.time()
train_scores_vc_xgb, test_scores_vc_xgb = validation_curve( estimator = regressor_xgb, 
                                                            X = X_reg, 
                                                            y = Y_reg, 
                                                            param_name = 'n_estimators', 
                                                            param_range = param_range_vc,
                                                            scoring = 'r2',
                                                            cv = 10,
                                                            n_jobs = 1)
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 208.63 min

train_mean_vc_xgb = np.mean(train_scores_vc_xgb, axis = 1)
train_std_vc_xgb = np.std(train_scores_vc_xgb, axis = 1)
test_mean_vc_xgb = np.mean(test_scores_vc_xgb, axis = 1)
test_std_vc_xgb = np.std(test_scores_vc_xgb, axis = 1)

plt.figure(figsize = (14, 7))
plt.plot( param_range_vc, 
          train_mean_vc_xgb, 
          color = 'blue', 
          marker = 'o', 
          markersize = 5, 
          label = 'Training R2 Score' )

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
          label = 'Validation R2 Score' )

plt.fill_between( param_range_vc, 
                  test_mean_vc_xgb + test_std_vc_xgb,
                  test_mean_vc_xgb - test_std_vc_xgb, 
                  alpha = 0.15, 
                  color = 'green')

plt.minorticks_on()
plt.grid(b = True, which = 'both', color = '#666666', linestyle = '-', alpha = 0.2, axis = 'both', linewidth = 1)

plt.grid()
#plt.xscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('Parameter n Estimator')
plt.ylabel('R2 Score')
plt.ylim([0.0, 1.0])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi = 300)
plt.show()

#CrossValidate - 10 KFolds - X_reg, Y_reg - Múltiplas Métricas

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

k_fold_reg = KFold(n_splits = 10, shuffle = True, random_state = seed_reg)

start = time.time()
result_reg_xgb_cross_validate = cross_validate( estimator = regressor_xgb, 
                                                X = X_reg, 
                                                y = Y_reg, 
                                                cv = k_fold_reg, 
                                                scoring = metricas_cross_validate_reg, 
                                                return_train_score = False )
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 19.52 min

sorted(result_reg_xgb_cross_validate.keys())

pd.set_option('display.float_format', lambda x: '%.3f' % x)

cv_fit_time_reg_xgb = result_reg_xgb_cross_validate['fit_time'].mean() 
print('Fit Time XGB: ' + str(cv_fit_time_reg_xgb))

cv_score_time_reg_xgb = result_reg_xgb_cross_validate['score_time'].mean()
print('Score Time XGB: ' + str(cv_score_time_reg_xgb))

cv_test_explained_variance_reg_xgb = result_reg_xgb_cross_validate['test_explained_variance'].mean()
print('Test Explained Variance XGB: ' + str(cv_test_explained_variance_reg_xgb))

cv_test_explained_variance_std_reg_xgb = result_reg_xgb_cross_validate['test_explained_variance'].std()
print('STD Test Explained Variance XGB: ' + str(cv_test_explained_variance_std_reg_xgb))

cv_test_mean_absolute_error_reg_xgb = result_reg_xgb_cross_validate['test_neg_mean_absolute_error'].mean()
print('Test Mean Absolute Error (MAE) XGB: ' + str(cv_test_mean_absolute_error_reg_xgb))

cv_test_mean_absolute_error_std_reg_xgb = result_reg_xgb_cross_validate['test_neg_mean_absolute_error'].std()
print('STD Test Mean Absolute Error (MAE) XGB: ' + str(cv_test_mean_absolute_error_std_reg_xgb))

cv_test_median_absolute_error_reg_xgb = result_reg_xgb_cross_validate['test_neg_median_absolute_error'].mean()
print('Test Median Absolute Error XGB: ' + str(cv_test_median_absolute_error_reg_xgb))

cv_test_median_absolute_error_std_reg_xgb = result_reg_xgb_cross_validate['test_neg_median_absolute_error'].std()
print('STD Test Median Absolute Error XGB: ' + str(cv_test_median_absolute_error_std_reg_xgb))

cv_test_mean_squared_error_reg_xgb = result_reg_xgb_cross_validate['test_neg_mean_squared_error'].mean()
print('Test Mean Squared Error (MSE) XGB: ' + str(cv_test_mean_squared_error_reg_xgb))

cv_test_mean_squared_error_std_reg_xgb = result_reg_xgb_cross_validate['test_neg_mean_squared_error'].std()
print('STD Test Mean Squared Error (MSE) XGB: ' + str(cv_test_mean_squared_error_std_reg_xgb))

cv_test_root_mean_squared_error_reg_xgb = np.sqrt(-result_reg_xgb_cross_validate['test_neg_mean_squared_error']).mean()
print('Test Root Mean Squared Error (RMSE) XGB: ' + str(cv_test_root_mean_squared_error_reg_xgb))

cv_test_root_mean_squared_error_std_reg_xgb = np.sqrt(-result_reg_xgb_cross_validate['test_neg_mean_squared_error']).std()
print('STD Test Root Mean Squared Error (RMSE) XGB: ' + str(cv_test_root_mean_squared_error_std_reg_xgb))

cv_test_r2_reg_xgb = result_reg_xgb_cross_validate['test_r2'].mean()
print('Test R2 Score XGB: ' + str(cv_test_r2_reg_xgb))

cv_test_r2_std_reg_xgb = result_reg_xgb_cross_validate['test_r2'].std()
print('STD Test R2 Score XGB: ' + str(cv_test_r2_std_reg_xgb))

cv_test_max_error_reg_xgb = result_reg_xgb_cross_validate['test_max_error'].mean()
print('Test Max Error XGB: ' + str(cv_test_max_error_reg_xgb))

cv_test_max_error_std_reg_xgb = result_reg_xgb_cross_validate['test_max_error'].std()
print('STD Test Max Error XGB: ' + str(cv_test_max_error_std_reg_xgb))

#X_train_reg, Y_train_reg
Fit Time XGB: 64.23052244186401
Score Time XGB: 0.09630956649780273
Test Explained Variance XGB: 0.2065360459863625
STD Test Explained Variance XGB: 0.03669193775295775
Test Mean Absolute Error (MAE) XGB: -1.7253060775605538
STD Test Mean Absolute Error (MAE) XGB: 0.04996861573085539
Test Median Absolute Error XGB: -1.3837092399597168
STD Test Median Absolute Error XGB: 0.027255046836289824
Test Mean Squared Error (MSE) XGB: -4.977231770322375
STD Test Mean Squared Error (MSE) XGB: 0.31771406594697965
Test Root Mean Squared Error (RMSE) XGB: 2.229804588709669
STD Test Root Mean Squared Error (RMSE) XGB: 0.07213367099820715
Test R2 Score XGB: 0.2039081671406883
STD Test R2 Score XGB: 0.036070096374850764
Test Max Error XGB: -7.491297709941864
STD Test Max Error XGB: 0.48094008810212063

#X_reg, Y_reg
Fit Time XGB: 116.73183226585388
Score Time XGB: 0.2581011295318604
Test Explained Variance XGB: 0.216304568564938
STD Test Explained Variance XGB: 0.01772733582095083
Test Mean Absolute Error (MAE) XGB: -1.7032711494582102
STD Test Mean Absolute Error (MAE) XGB: 0.03314452538409922
Test Median Absolute Error XGB: -1.341743814945221
STD Test Median Absolute Error XGB: 0.05439640242805105
Test Mean Squared Error (MSE) XGB: -4.887712971460003
STD Test Mean Squared Error (MSE) XGB: 0.16234530335992317
Test Root Mean Squared Error (RMSE) XGB: 2.210518851096094
STD Test Root Mean Squared Error (RMSE) XGB: 0.036323276405161614
Test R2 Score XGB: 0.2159275650571511
STD Test R2 Score XGB: 0.01787045010740871
Test Max Error XGB: -8.02560658454895
STD Test Max Error XGB: 0.505836812583131

Base Antiga
Fit Time XGB: 116.73183226585388
Score Time XGB: 0.2581011295318604
Test Explained Variance XGB: 0.216304568564938
STD Test Explained Variance XGB: 0.01772733582095083
Test Mean Absolute Error (MAE) XGB: -1.7032711494582102
STD Test Mean Absolute Error (MAE) XGB: 0.03314452538409922
Test Median Absolute Error XGB: -1.341743814945221
STD Test Median Absolute Error XGB: 0.05439640242805105
Test Mean Squared Error (MSE) XGB: -4.887712971460003
STD Test Mean Squared Error (MSE) XGB: 0.16234530335992317
Test Root Mean Squared Error (RMSE) XGB: 2.210518851096094
STD Test Root Mean Squared Error (RMSE) XGB: 0.036323276405161614
Test R2 Score XGB: 0.2159275650571511
STD Test R2 Score XGB: 0.01787045010740871
Test Max Error XGB: -8.02560658454895
STD Test Max Error XGB: 0.505836812583131

Base Nova
Fit Time XGB: 118.65805592536927
Score Time XGB: 0.16866085529327393
Test Explained Variance XGB: 0.2176860429152097
STD Test Explained Variance XGB: 0.02271560398956888
Test Mean Absolute Error (MAE) XGB: -1.7058556381129162
STD Test Mean Absolute Error (MAE) XGB: 0.0427529032527247
Test Median Absolute Error XGB: -1.3631755352020263
STD Test Median Absolute Error XGB: 0.06576732971129609
Test Mean Squared Error (MSE) XGB: -4.889621428602412
STD Test Mean Squared Error (MSE) XGB: 0.18852677372803658
Test Root Mean Squared Error (RMSE) XGB: 2.210840680208908
STD Test Root Mean Squared Error (RMSE) XGB: 0.04248429516685629
Test R2 Score XGB: 0.216912799258591
STD Test R2 Score XGB: 0.022665875016068836
Test Max Error XGB: -7.981030076742172
STD Test Max Error XGB: 0.6074572561128684

