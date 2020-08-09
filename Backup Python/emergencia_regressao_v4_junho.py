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
dfr = pd.read_excel('TB_IND_ASSIST_EME_0419_0819_V6.xlsx')
end = time.time()

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 1.13 min

dfr.shape
#(8253, 652)

#Criação da Base df_junho_19_reg com Todas as Informações Disponíveis

df_junho_19_reg = dfr.loc[(dfr['CONT_STATUS_INT'] == 1) & (dfr['NUM_MES'] == 6) & (dfr['TEMPO_PERM_INT_POSTERIOR'] != 0) & (dfr['TEMPO_PERM_INT_POSTERIOR'] <= 10)].copy()

#Verificação da Quantidade de Linhas e Colunas da Base de Dados

df_abril_19_reg.shape
#(164, 652)
df_maio_19_reg.shape
#(183, 652)
df_junho_19_reg.shape
#(174, 652)
df_julho_19_reg.shape
#(178, 652)
df_agosto_19_reg.shape
#(163, 652)

#Descrição do Index

df_junho_19_reg.index

#Contagem de Dados Não Nulos

df_junho_19_reg.count()

#Verificação do Nome Das Colunas da Base de Dados

df_junho_19_reg.columns

df_junho_19_reg.columns.values

#Visualização das n Primeiras Linhas

df_junho_19_reg.head(10)

#Estatísticas Descritivas da Base de Dados

df_junho_19_reg.info([''])

df_junho_19_reg.describe()

df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'].describe()

df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'].median()

df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'].var()

df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'].mad()

#Distribution Target Variable

import matplotlib.pyplot as plt 
 
import seaborn as sns

import numpy as np

sns.set(rc = {'figure.figsize':(14, 7)})
sns.distplot(df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'], bins = 30)
plt.show()

plt.figure(figsize = (14, 7))
plt.scatter(range(df_junho_19_reg.shape[0]), np.sort(df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'].values))
plt.xlabel('index')
plt.ylabel('Tempo de Permanência')
plt.title("Tempo de Permanência - Distribution")
plt.show();

#Target Variable Analysis

from scipy import stats

from scipy.stats import norm, skew

import seaborn as sns

import matplotlib.pyplot as plt 

(mu, sigma) = norm.fit(df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'])

plt.figure(figsize = (14, 7))
sns.distplot(df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'], fit = norm)
plt.ylabel('Frequency')
plt.title('Tempo de Permanência - Distribution')
plt.legend(['Normal Dist. ($\mu=$ {:.2f} and $\sigma=$ {:.2f} )'.format(mu, sigma)], loc = 'best')

quantile_plot = stats.probplot(df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'], plot = plt)

import numpy as np

df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'] = np.log1p(df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'])

(mu, sigma) = norm.fit(df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'])
plt.figure(figsize = (14, 7))
plt.subplot(1,2,1)
sns.distplot(df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'], fit = norm)
plt.ylabel('Frequency')
plt.title('Tempo de Permanência - Distribution')
plt.subplot(1, 2, 2)
quantile_plot = stats.probplot(df_junho_19_reg['TEMPO_PERM_INT_POSTERIOR'], plot = plt)

#Cálculo de Simetria dos Dados

df_junho_19_reg.skew()

#Análise de Correlação das Variáveis - Pearson

import numpy as np

corr_matrix_pearson_reg = df_junho_19_reg.corr('pearson')

corr_matrix_pearson_reg['TEMPO_PERM_INT_POSTERIOR'].sort_values(ascending = False)

k_reg = 20

corr_matrix_pearson_reg.nlargest(k_reg, 'TEMPO_PERM_INT_POSTERIOR')['TEMPO_PERM_INT_POSTERIOR']

cols_reg = corr_matrix_pearson_reg.nlargest(k_reg, 'TEMPO_PERM_INT_POSTERIOR')['TEMPO_PERM_INT_POSTERIOR'].index
cm_reg = np.corrcoef(df_junho_19_reg[cols_reg].values.T)
sns.set(font_scale = 1.2)
plt.figure(figsize = (10, 10))
hm_reg = sns.heatmap(cm_reg, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols_reg.values, xticklabels = cols_reg.values)
plt.show()

corr_matrix_pearson_reg_new = df_junho_19_reg[corr_matrix_pearson_reg['TEMPO_PERM_INT_POSTERIOR'].nlargest(20).index].corr('pearson')

#Análise de Correlação das Variáveis - Spearman

import numpy as np

corr_matrix_spearman_reg = df_junho_19_reg.corr('spearman')

corr_matrix_spearman_reg['TEMPO_PERM_INT_POSTERIOR'].sort_values(ascending = False)

k_reg = 20

corr_matrix_spearman_reg.nlargest(k_reg, 'TEMPO_PERM_INT_POSTERIOR')['TEMPO_PERM_INT_POSTERIOR']

#cols_reg = corr_matrix_spearman_reg.nlargest(k_reg, 'TEMPO_PERM_INT_POSTERIOR')['TEMPO_PERM_INT_POSTERIOR'].index
#cm_reg = np.corrcoef(df_junho_19_reg[cols_reg].values.T)
#sns.set(font_scale = 1.2)
#plt.figure(figsize = (10, 10))
#hm_reg = sns.heatmap(cm_reg, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols_reg.values, xticklabels = cols_reg.values)
#plt.show()

corr_matrix_spearman_reg_new = df_junho_19_reg[corr_matrix_spearman_reg['TEMPO_PERM_INT_POSTERIOR'].nlargest(20).index].corr('spearman')

cols_reg = corr_matrix_spearman_reg_new.index
sns.set(font_scale = 1.2)
plt.figure(figsize = (10, 10))
hm_reg = sns.heatmap(corr_matrix_spearman_reg_new, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols_reg.values, xticklabels = cols_reg.values)
plt.show()

#Análise de Correlação das Variáveis - Kendall

import numpy as np

corr_matrix_kendall_reg = df_junho_19_reg.corr('kendall')

corr_matrix_kendall_reg['TEMPO_PERM_INT_POSTERIOR'].sort_values(ascending = False)

k_reg = 20

corr_matrix_kendall_reg.nlargest(k_reg, 'TEMPO_PERM_INT_POSTERIOR')['TEMPO_PERM_INT_POSTERIOR']

#cols_reg = corr_matrix_kendall_reg.nlargest(k_reg, 'TEMPO_PERM_INT_POSTERIOR')['TEMPO_PERM_INT_POSTERIOR'].index
#cm_reg = np.corrcoef(df_junho_19_reg[cols_reg].values.T)
#sns.set(font_scale = 1.2)
#plt.figure(figsize = (10, 10))
#hm_reg = sns.heatmap(cm_reg, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols_reg.values, xticklabels = cols_reg.values)
#plt.show()

corr_matrix_kendall_reg_new = df_junho_19_reg[corr_matrix_kendall_reg['TEMPO_PERM_INT_POSTERIOR'].nlargest(20).index].corr('kendall')

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

df_junho_19_reg.isnull().sum()

df_junho_19_reg.isna().sum()

for col in df_junho_19_reg.columns:
    if df_junho_19_reg[col].isnull().sum() > 0:
        print(col)
        
#Análise de Valores Nulos ou Não Existentes - Data Frame        
        
total_reg = df_junho_19_reg.isnull().sum().sort_values(ascending = False)

percent_1_reg = df_junho_19_reg.isnull().sum()/df_junho_19_reg.isnull().count()*100

percent_2_reg = (round(percent_1_reg, 2)).sort_values(ascending = False)

missing_data_reg = pd.concat([total_reg, percent_2_reg], axis = 1, keys = ['Total', '%'])

missing_data_reg[missing_data_reg.Total > 0].sort_values(ascending = False, by = 'Total').head(50)        
    
#Preenchimento dos Valores Nulos

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_02'])
df_junho_19_reg['PERGUNTA_TRIAGEM_02'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_02'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_03'])
df_junho_19_reg['PERGUNTA_TRIAGEM_03'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_03'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_04'])
df_junho_19_reg['PERGUNTA_TRIAGEM_04'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_04'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_05'])
df_junho_19_reg['PERGUNTA_TRIAGEM_05'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_05'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_06'])
df_junho_19_reg['PERGUNTA_TRIAGEM_06'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_06'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_07'])
df_junho_19_reg['PERGUNTA_TRIAGEM_07'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_07'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_08'])
df_junho_19_reg['PERGUNTA_TRIAGEM_08'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_08'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_09'])
df_junho_19_reg['PERGUNTA_TRIAGEM_09'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_09'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_10'])
df_junho_19_reg['PERGUNTA_TRIAGEM_10'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_10'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_11'])
df_junho_19_reg['PERGUNTA_TRIAGEM_11'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_11'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_12'])
df_junho_19_reg['PERGUNTA_TRIAGEM_12'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_12'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_13'])
df_junho_19_reg['PERGUNTA_TRIAGEM_13'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_13'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_14'])
df_junho_19_reg['PERGUNTA_TRIAGEM_14'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_14'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_15'])
df_junho_19_reg['PERGUNTA_TRIAGEM_15'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_15'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_16'])
df_junho_19_reg['PERGUNTA_TRIAGEM_16'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_16'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_17'])
df_junho_19_reg['PERGUNTA_TRIAGEM_17'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_17'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_18'])
df_junho_19_reg['PERGUNTA_TRIAGEM_18'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_18'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_19'])
df_junho_19_reg['PERGUNTA_TRIAGEM_19'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_19'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_20'])
df_junho_19_reg['PERGUNTA_TRIAGEM_20'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_20'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_21'])
df_junho_19_reg['PERGUNTA_TRIAGEM_21'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_21'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_22'])
df_junho_19_reg['PERGUNTA_TRIAGEM_22'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_22'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_23'])
df_junho_19_reg['PERGUNTA_TRIAGEM_23'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_23'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_24'])
df_junho_19_reg['PERGUNTA_TRIAGEM_24'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_24'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_25'])
df_junho_19_reg['PERGUNTA_TRIAGEM_25'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_25'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_26'])
df_junho_19_reg['PERGUNTA_TRIAGEM_26'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_26'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_27'])
df_junho_19_reg['PERGUNTA_TRIAGEM_27'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_27'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_28'])
df_junho_19_reg['PERGUNTA_TRIAGEM_28'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_28'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_29'])
df_junho_19_reg['PERGUNTA_TRIAGEM_29'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_29'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_30'])
df_junho_19_reg['PERGUNTA_TRIAGEM_30'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_30'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_31'])
df_junho_19_reg['PERGUNTA_TRIAGEM_31'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_31'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_32'])
df_junho_19_reg['PERGUNTA_TRIAGEM_32'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_32'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_TRIAGEM_33'])
df_junho_19_reg['PERGUNTA_TRIAGEM_33'].count()
df_junho_19_reg['PERGUNTA_TRIAGEM_33'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_02'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_02'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_02'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_03'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_03'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_03'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_04'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_04'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_04'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_05'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_05'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_05'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_06'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_06'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_06'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_07'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_07'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_07'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_08'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_08'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_08'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_09'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_09'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_09'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_10'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_10'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_10'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_11'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_11'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_11'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_12'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_12'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_12'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_13'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_13'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_13'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_14'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_14'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_14'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_15'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_15'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_15'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_16'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_16'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_16'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_17'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_17'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_17'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_18'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_18'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_18'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_19'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_19'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_19'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_20'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_20'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_20'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_21'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_21'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_21'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_22'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_22'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_22'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_23'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_23'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_23'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_24'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_24'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_24'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_25'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_25'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_25'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['PERGUNTA_FICHA_EME_PA_26'])
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_26'].count()
df_junho_19_reg['PERGUNTA_FICHA_EME_PA_26'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['COD_CID_ADM'])
df_junho_19_reg['COD_CID_ADM'].count()
df_junho_19_reg['COD_CID_ADM'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['COD_CID_ADM_PRINCIPAL'])
df_junho_19_reg['COD_CID_ADM_PRINCIPAL'].count()
df_junho_19_reg['COD_CID_ADM_PRINCIPAL'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['INDICADOR_IAM_CSST'])
df_junho_19_reg['INDICADOR_IAM_CSST'].count()
df_junho_19_reg['INDICADOR_IAM_CSST'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MIN_TEMPERATURA'])
df_junho_19_reg['CSV_MIN_TEMPERATURA'].count()
df_junho_19_reg['CSV_MIN_TEMPERATURA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MAX_TEMPERATURA'])
df_junho_19_reg['CSV_MAX_TEMPERATURA'].count()
df_junho_19_reg['CSV_MAX_TEMPERATURA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MEDIA_TEMPERATURA'])
df_junho_19_reg['CSV_MEDIA_TEMPERATURA'].count()
df_junho_19_reg['CSV_MEDIA_TEMPERATURA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MIN_TA_SISTOLICA'])
df_junho_19_reg['CSV_MIN_TA_SISTOLICA'].count()
df_junho_19_reg['CSV_MIN_TA_SISTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MAX_TA_SISTOLICA'])
df_junho_19_reg['CSV_MAX_TA_SISTOLICA'].count()
df_junho_19_reg['CSV_MAX_TA_SISTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MEDIA_TA_SISTOLICA'])
df_junho_19_reg['CSV_MEDIA_TA_SISTOLICA'].count()
df_junho_19_reg['CSV_MEDIA_TA_SISTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MIN_TA_DIASTOLICA'])
df_junho_19_reg['CSV_MIN_TA_DIASTOLICA'].count()
df_junho_19_reg['CSV_MIN_TA_DIASTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MAX_TA_DIASTOLICA'])
df_junho_19_reg['CSV_MAX_TA_DIASTOLICA'].count()
df_junho_19_reg['CSV_MAX_TA_DIASTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MEDIA_TA_DIASTOLICA'])
df_junho_19_reg['CSV_MEDIA_TA_DIASTOLICA'].count()
df_junho_19_reg['CSV_MEDIA_TA_DIASTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MIN_FREQ_RESPIRATORIA'])
df_junho_19_reg['CSV_MIN_FREQ_RESPIRATORIA'].count()
df_junho_19_reg['CSV_MIN_FREQ_RESPIRATORIA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MAX_FREQ_RESPIRATORIA'])
df_junho_19_reg['CSV_MAX_FREQ_RESPIRATORIA'].count()
df_junho_19_reg['CSV_MAX_FREQ_RESPIRATORIA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MEDIA_FREQ_RESPIRATORIA'])
df_junho_19_reg['CSV_MEDIA_FREQ_RESPIRATORIA'].count()
df_junho_19_reg['CSV_MEDIA_FREQ_RESPIRATORIA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MIN_GLICEMIA_DIGITAL'])
df_junho_19_reg['CSV_MIN_GLICEMIA_DIGITAL'].count()
df_junho_19_reg['CSV_MIN_GLICEMIA_DIGITAL'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MAX_GLICEMIA_DIGITAL'])
df_junho_19_reg['CSV_MAX_GLICEMIA_DIGITAL'].count()
df_junho_19_reg['CSV_MAX_GLICEMIA_DIGITAL'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MEDIA_GLICEMIA_DIGITAL'])
df_junho_19_reg['CSV_MEDIA_GLICEMIA_DIGITAL'].count()
df_junho_19_reg['CSV_MEDIA_GLICEMIA_DIGITAL'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MIN_SATURACAO_O2'])
df_junho_19_reg['CSV_MIN_SATURACAO_O2'].count()
df_junho_19_reg['CSV_MIN_SATURACAO_O2'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MAX_SATURACAO_O2'])
df_junho_19_reg['CSV_MAX_SATURACAO_O2'].count()
df_junho_19_reg['CSV_MAX_SATURACAO_O2'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MEDIA_SATURACAO_O2'])
df_junho_19_reg['CSV_MEDIA_SATURACAO_O2'].count()
df_junho_19_reg['CSV_MEDIA_SATURACAO_O2'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MIN_PRESSAO_ARTERIAL'])
df_junho_19_reg['CSV_MIN_PRESSAO_ARTERIAL'].count()
df_junho_19_reg['CSV_MIN_PRESSAO_ARTERIAL'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MAX_PRESSAO_ARTERIAL'])
df_junho_19_reg['CSV_MAX_PRESSAO_ARTERIAL'].count()
df_junho_19_reg['CSV_MAX_PRESSAO_ARTERIAL'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MEDIA_PRESSAO_ARTERIAL'])
df_junho_19_reg['CSV_MEDIA_PRESSAO_ARTERIAL'].count()
df_junho_19_reg['CSV_MEDIA_PRESSAO_ARTERIAL'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MIN_FREQ_CARDIACA'])
df_junho_19_reg['CSV_MIN_FREQ_CARDIACA'].count()
df_junho_19_reg['CSV_MIN_FREQ_CARDIACA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MAX_FREQ_CARDIACA'])
df_junho_19_reg['CSV_MAX_FREQ_CARDIACA'].count()
df_junho_19_reg['CSV_MAX_FREQ_CARDIACA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['CSV_MEDIA_FREQ_CARDIACA'])
df_junho_19_reg['CSV_MEDIA_FREQ_CARDIACA'].count()
df_junho_19_reg['CSV_MEDIA_FREQ_CARDIACA'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['TEMPO_ATENDIMENTO'])
df_junho_19_reg['TEMPO_ATENDIMENTO'].count()
df_junho_19_reg['TEMPO_ATENDIMENTO'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['TEMPO_ATD_ULT_ATENDIMENTO'])
df_junho_19_reg['TEMPO_ATD_ULT_ATENDIMENTO'].count()
df_junho_19_reg['TEMPO_ATD_ULT_ATENDIMENTO'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['MEDIA_TEMPO_ATD_ATD_ANTERIOR'])
df_junho_19_reg['MEDIA_TEMPO_ATD_ATD_ANTERIOR'].count()
df_junho_19_reg['MEDIA_TEMPO_ATD_ATD_ANTERIOR'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['TEMPO_PERM_ULT_INTERNAMENTO'])
df_junho_19_reg['TEMPO_PERM_ULT_INTERNAMENTO'].count()
df_junho_19_reg['TEMPO_PERM_ULT_INTERNAMENTO'].fillna(-1, inplace = True)

pd.value_counts(df_junho_19_reg['MEDIA_TEMPO_PERM_INT_ANT'])
df_junho_19_reg['MEDIA_TEMPO_PERM_INT_ANT'].count()
df_junho_19_reg['MEDIA_TEMPO_PERM_INT_ANT'].fillna(-1, inplace = True)

#Verificação dos Valores Nulos Ainda Existentes

for col in df_junho_19_reg.columns:
    if df_junho_19_reg[col].isnull().sum() > 0:
        print(col)

#Exclusão das Colunas Que Não Serão Utilizadas
        
df_junho_19_reg_trat = df_junho_19_reg.drop('DATA_ADM', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('DESC_DIA_DA_SEMANA', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('REGISTRO', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('NOME_PACIENTE', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('DESC_SEXO', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('DESC_FAIXA_ETARIA', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('NUM_TRATAMENTO', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('DATA_ADMISSAO', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('DESC_DATA_ADMISSAO', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('DESC_CONVENIO', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('COD_EMPRESA', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('DESC_EMPRESA', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('COD_PLANTONISTA', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('NOME_PLANTONISTA', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('PERGUNTA_TRIAGEM_01', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('PERGUNTA_FICHA_EME_PA_01', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('PERGUNTA_CONSULTA_P_07', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('PERGUNTA_CONSULTA_P_13', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('DESC_DATA_ALTA', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('CONT_STATUS_INT', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('DESC_STATUS_INT', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('NUM_INT_POSTERIOR', axis = 1)

df_junho_19_reg_trat = df_junho_19_reg_trat.drop('CONTADOR', axis = 1)

#Verificação dos Valores Nulos Ainda Existentes

for col in df_junho_19_reg_trat.columns:
    if df_junho_19_reg_trat[col].isnull().sum() > 0:
        print(col)
        
#Estatísticas Descritivas da Base de Dados - Após Tratamento dos Dados
        
df_junho_19_reg_trat.info([''])

#Criação das Bases X_junho_19_reg e Y_junho_19_reg 

X_junho_19_reg = df_junho_19_reg_trat

Y_junho_19_reg = df_junho_19_reg_trat['TEMPO_PERM_INT_POSTERIOR']

X_junho_19_reg = X_junho_19_reg.drop('TEMPO_PERM_INT_POSTERIOR', axis = 1)

del df_junho_19_reg

del df_junho_19_reg_trat

#Estatísticas Descritivas da Base de Dados - X_junho_19_reg - Após Remoção da Coluna Target

X_junho_19_reg.info([''])

#Aplicação do Label Encoder na Base X_junho_19_reg

from sklearn.preprocessing import LabelEncoder

lbl_enc_X_junho_19_reg = LabelEncoder()

X_junho_19_reg['DESC_ESTACAO_ANO'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['DESC_ESTACAO_ANO'])

X_junho_19_reg['DESC_FINAL_SEMANA'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['DESC_FINAL_SEMANA'])

X_junho_19_reg['DESC_BAIRRO'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['DESC_BAIRRO'])

X_junho_19_reg['COD_CID'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['COD_CID'])

X_junho_19_reg['DESC_GRUPO_SANGUINEO'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['DESC_GRUPO_SANGUINEO'])

X_junho_19_reg['APEL_PLANTONISTA'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['APEL_PLANTONISTA'])

X_junho_19_reg['CID_PREVALENTE'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['CID_PREVALENTE'])

X_junho_19_reg['COD_CLAC_RISCO'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['COD_CLAC_RISCO'])

X_junho_19_reg['PERGUNTA_TRIAGEM_02'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_02'])

X_junho_19_reg['PERGUNTA_TRIAGEM_03'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_03'])

X_junho_19_reg['PERGUNTA_TRIAGEM_04'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_04'])

X_junho_19_reg['PERGUNTA_TRIAGEM_05'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_05'])

X_junho_19_reg['PERGUNTA_TRIAGEM_06'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_06'])

X_junho_19_reg['PERGUNTA_TRIAGEM_07'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_07'])

X_junho_19_reg['PERGUNTA_TRIAGEM_08'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_08'])

X_junho_19_reg['PERGUNTA_TRIAGEM_09'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_09'])

X_junho_19_reg['PERGUNTA_TRIAGEM_10'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_10'])

X_junho_19_reg['PERGUNTA_TRIAGEM_11'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_11'])

X_junho_19_reg['PERGUNTA_TRIAGEM_12'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_12'])

X_junho_19_reg['PERGUNTA_TRIAGEM_13'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_13'])

X_junho_19_reg['PERGUNTA_TRIAGEM_14'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_14'])

X_junho_19_reg['PERGUNTA_TRIAGEM_15'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_15'])

X_junho_19_reg['PERGUNTA_TRIAGEM_16'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_16'])

X_junho_19_reg['PERGUNTA_TRIAGEM_24'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_24'])

X_junho_19_reg['PERGUNTA_TRIAGEM_25'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_25'])

X_junho_19_reg['PERGUNTA_TRIAGEM_26'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_26'])

X_junho_19_reg['PERGUNTA_TRIAGEM_27'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_27'])

X_junho_19_reg['PERGUNTA_TRIAGEM_28'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_28'])

X_junho_19_reg['PERGUNTA_TRIAGEM_29'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_29'])

X_junho_19_reg['PERGUNTA_TRIAGEM_30'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_30'])

X_junho_19_reg['PERGUNTA_TRIAGEM_31'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_31'])

X_junho_19_reg['PERGUNTA_TRIAGEM_32'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_32'])

X_junho_19_reg['PERGUNTA_TRIAGEM_33'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_TRIAGEM_33'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_02'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_02'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_03'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_03'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_04'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_04'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_05'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_05'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_06'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_06'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_07'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_07'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_08'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_08'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_09'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_09'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_10'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_10'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_11'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_11'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_12'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_12'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_13'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_13'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_14'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_14'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_15'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_15'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_16'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_16'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_17'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_17'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_18'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_18'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_19'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_19'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_20'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_20'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_21'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_21'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_22'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_22'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_23'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_23'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_24'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_24'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_25'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_25'])

X_junho_19_reg['PERGUNTA_FICHA_EME_PA_26'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_FICHA_EME_PA_26'])

X_junho_19_reg['COD_CID_ADM'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['COD_CID_ADM'])

X_junho_19_reg['COD_CID_ADM_PRINCIPAL'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['COD_CID_ADM_PRINCIPAL'])

X_junho_19_reg['PERGUNTA_CONSULTA_P_09'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_CONSULTA_P_09'])

X_junho_19_reg['PERGUNTA_CONSULTA_P_10'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_CONSULTA_P_10'])

X_junho_19_reg['PERGUNTA_CONSULTA_P_12'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_CONSULTA_P_12'])

X_junho_19_reg['PERGUNTA_CONSULTA_P_14'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_CONSULTA_P_14'])

X_junho_19_reg['PERGUNTA_CONSULTA_P_15'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_CONSULTA_P_15'])

X_junho_19_reg['PERGUNTA_CONSULTA_P_16'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['PERGUNTA_CONSULTA_P_16'])

X_junho_19_reg['INDICADOR_IAM_CSST'] = lbl_enc_X_junho_19_reg.fit_transform(X_junho_19_reg['INDICADOR_IAM_CSST'])

#Estatísticas Descritivas da Base de Dados - X_junho_19_reg - Após Aplicação do Label Encoder

X_junho_19_reg.info([''])

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

feature_list_X_junho_19_reg = list(X_junho_19_reg.columns)

feature_list_X_junho_19_reg_integer, feature_list_X_junho_19_reg_float, features_list_X_junho_19_reg_object = groupFeatures(feature_list_X_junho_19_reg, X_junho_19_reg)

print("# of integer feature : ", len(feature_list_X_junho_19_reg_integer))
print("# of float feature : ", len(feature_list_X_junho_19_reg_float))
print("# of object feature : ", len(features_list_X_junho_19_reg_object))

#Transformação da Base Y_junho_19_reg em Array

import numpy as np

Y_junho_19_reg = np.array(Y_junho_19_reg)

#Transformação da Base X_junho_19_reg em Array

X_list_reg = list(X_junho_19_reg.columns)

X_junho_19_reg = np.array(X_junho_19_reg)

#Escalonamento da Base X_junho_19_reg - Opção 01 - StandardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_junho_19_reg = scaler.fit_transform(X_junho_19_reg)

#ETAPA 03: PREDIÇÃO

Y_junho_19_pred_train_reg_xgb = regressor_xgb.predict(X_junho_19_reg)

#Análise de Métricas

from sklearn.metrics import explained_variance_score 
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import median_absolute_error
from sklearn.metrics import mean_squared_error
from sklearn.metrics import r2_score
from sklearn.metrics import max_error

mtrc_evs_xgb = explained_variance_score(Y_junho_19_reg, Y_junho_19_pred_train_reg_xgb)
mtrc_mae_xgb = mean_absolute_error(Y_junho_19_reg, Y_junho_19_pred_train_reg_xgb)
mtrc_mdae_xgb = median_absolute_error(Y_junho_19_reg, Y_junho_19_pred_train_reg_xgb)
mtrc_mse_xgb = mean_squared_error(Y_junho_19_reg, Y_junho_19_pred_train_reg_xgb)
mtrc_rmse_xgb = np.sqrt(mean_squared_error(Y_junho_19_reg, Y_junho_19_pred_train_reg_xgb))
mtrc_r2_xgb = r2_score(Y_junho_19_reg, Y_junho_19_pred_train_reg_xgb)
mtrc_max_error_xgb = max_error(Y_junho_19_reg, Y_junho_19_pred_train_reg_xgb)

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
Explained Variance Score: 0.1319757975802817
Mean Absolute Error: 1.8523284813453411
Median Absolute Error: 1.6556223630905151
Mean Squared Error: 5.0954648181863433
Root Mean Squared Error: 2.2573136286715552
R² Score: 0.0959100492544963
Max Error: 6.3273682594299316