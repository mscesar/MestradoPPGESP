#Setup da Pasta de Trabalho
import os
os.chdir('C:/Users/Mauricio/Dropbox/MauricioPessoal/Pós-Graduação/PPGESP/Projeto Final/Estudo Python/Bases Cárdio Pulmonar')

import warnings
warnings.filterwarnings("ignore")

#ETAPA 01: TRATAMENTO DOS DADOS

#Importação do Pandas

import pandas as pd

#Leitura da Base de Dados

#Criação da Base df com Todas as Informações Disponíveis

import time

start = time.time()
df1 = pd.read_excel('TB_IND_ASSIST_EME_0115_0819_V6.xlsx')
end = time.time()

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 19.24 min

df1.shape
#(92563, 652)

df = df1.loc[(df1['TEMPO_PERM_INT_POSTERIOR'] != 0)].copy()

#Verificação da Quantidade de Linhas e Colunas da Base de Dados

df.shape
#(91843, 652)

#Descrição do Index

df.index

#Contagem de Dados Não Nulos

df.count()

#Verificação do Nome Das Colunas da Base de Dados

df.columns

df.columns.values

#Visualização das n Primeiras Linhas

df.head(10)

#Visualização das n Últimas Linhas

df.tail(10)

#Estatísticas Descritivas da Base de Dados

df.info([''])

df.describe()

#Verificação da Distribuição dos Tipos de Dados das Variáveis

feature_list_df = list(df.columns)

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

feature_list_df_integer, feature_list_df_float, features_list_df_object = groupFeatures(feature_list_df, df)
print("# of integer feature : ", len(feature_list_df_integer))
print("# of float feature : ", len(feature_list_df_float))
print("# of object feature : ", len(features_list_df_object))

#Verificação do % de Dados da Variável Target

all_data = df.shape[0]
pac_internado = df[df['CONT_STATUS_INT'] == 1]
pac_não_internado = df[df['CONT_STATUS_INT'] == 0]

x = len(pac_internado)/all_data
y = len(pac_não_internado)/all_data

print('Pacientes Internados :', round(x*100,2),'%')
print('Pacientes Não Internados :', round(y*100,2),'%')

#Pacientes Internados : 10.95 %
#Pacientes Não Internados : 89.05 %

#Verificação do % de Dados da Variável Target - Gráfico - Op 1

import matplotlib.pyplot as plt

labels = ['Pacientes Não Internados','Pacientes Internados']
classes = pd.value_counts(df['CONT_STATUS_INT'], sort = True)
plt.figure(figsize = (14, 7))
classes.plot(kind = 'bar', rot = 0)
plt.title("Target Class Distribution")
plt.xticks(range(2), labels)
plt.xlabel("Class")
plt.ylabel("Frequency")

#Verificação do % de Dados da Variável Target - Gráfico - Op 2

import seaborn as sns

import matplotlib.pyplot as plt

targets = df['DESC_STATUS_INT'].values
sns.set(style = "darkgrid")
plt.figure(figsize = (14, 7))
ax = sns.countplot(x = targets)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x() + 0.3, p.get_height() + 10000))
plt.title('Target Class Distribution', fontsize = 20)
plt.xlabel('Class', fontsize = 20)
plt.ylabel('Frequency [%]', fontsize = 20)
ax.set_ylim(top = 100000)

#Verificação do % de Dados da Variável Target - Gráfico - Op 3

import matplotlib.pyplot as plt

import seaborn as sns

pac_internado = 'Pacientes Internados'
pac_nao_internado = 'Pacientes Não Internados'
fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize = (14, 7))
women = df[df['COD_SEXO'] == 2]
men = df[df['COD_SEXO'] == 1]
ax = sns.distplot(women[women['CONT_STATUS_INT'] == 1].NUM_IDADE.dropna(), bins = 18, label = pac_internado, ax = axes[0], kde = False)
ax = sns.distplot(women[women['CONT_STATUS_INT'] == 0].NUM_IDADE.dropna(), bins = 40, label = pac_nao_internado, ax = axes[0], kde = False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['CONT_STATUS_INT'] == 1].NUM_IDADE.dropna(), bins = 18, label = pac_internado, ax = axes[1], kde = False)
ax = sns.distplot(men[men['CONT_STATUS_INT'] == 0].NUM_IDADE.dropna(), bins = 40, label = pac_nao_internado, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')

pac_internado = 'Pacientes Internados'
pac_nao_internado = 'Pacientes Não Internados'
fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize = (14, 7))
women = df[df['COD_SEXO'] == 2]
men = df[df['COD_SEXO'] == 1]
ax = sns.distplot(women[women['CONT_STATUS_INT'] == 1].NUM_ANO.dropna(), bins = 18, label = pac_internado, ax = axes[0], kde = False)
ax = sns.distplot(women[women['CONT_STATUS_INT'] == 0].NUM_ANO.dropna(), bins = 40, label = pac_nao_internado, ax = axes[0], kde = False)
ax.legend()
ax.set_title('Female')
ax = sns.distplot(men[men['CONT_STATUS_INT'] == 1].NUM_ANO.dropna(), bins = 18, label = pac_internado, ax = axes[1], kde = False)
ax = sns.distplot(men[men['CONT_STATUS_INT'] == 0].NUM_ANO.dropna(), bins = 40, label = pac_nao_internado, ax = axes[1], kde = False)
ax.legend()
_ = ax.set_title('Male')

#Gráficos de Distribuição de Variáveis

import matplotlib.pyplot as plt

import seaborn as sns

#Ano

var_ano = df['NUM_ANO'].values
plt.figure(figsize = (14, 7))
sns.distplot(var_ano)

#Mês

var_mes = df['NUM_MES'].values
plt.figure(figsize = (14, 7))
sns.distplot(var_mes)

#Dia da Semana

var_dia_da_semana = df['NUM_DIA_DA_SEMANA'].values
plt.figure(figsize = (14, 7))
sns.distplot(var_dia_da_semana)

#Sexo

var_sexo = df['COD_SEXO'].values
plt.figure(figsize = (14, 7))
sns.distplot(var_sexo)

#Idade

var_idade = df['NUM_IDADE'].values
plt.figure(figsize = (14, 7))
sns.distplot(var_idade)

#Faixa Etária

var_faixa_etaria = df['COD_FAIXA_ETARIA'].values
plt.figure(figsize = (14, 7))
sns.distplot(var_faixa_etaria)

#Cálculo de Simetria dos Dados

import matplotlib.pyplot as plt

skew_df = df.skew()

print(skew_df)

plt.figure(figsize = (14, 7))
skew_df.plot()

#Análise de Correlação das Variáveis - Pearson

import numpy as np

import time

import seaborn as sns

import matplotlib.pyplot as plt

start = time.time()

corr_matrix_pearson = df.corr('pearson')

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 4.27 min

corr_matrix_pearson['CONT_STATUS_INT'].sort_values(ascending = False)

k = 20

corr_matrix_pearson.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT']

cols = corr_matrix_pearson.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1.2)
plt.figure(figsize = (10, 10))
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()

corr_matrix_pearson_new = df[corr_matrix_pearson['CONT_STATUS_INT'].nlargest(20).index].corr('pearson')

#Análise de Correlação das Variáveis - Spearman

import numpy as np

import time

import seaborn as sns

import matplotlib.pyplot as plt

start = time.time()

corr_matrix_spearman = df.corr('spearman')

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 132.87 min

corr_matrix_spearman["CONT_STATUS_INT"].sort_values(ascending = False)

k = 20

corr_matrix_spearman.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT']

#cols = corr_matrix_spearman.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
#cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale = 1.2)
#plt.figure(figsize = (28, 28))
#hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
#plt.show()

corr_matrix_spearman_new = df[corr_matrix_spearman['CONT_STATUS_INT'].nlargest(20).index].corr('spearman')

cols = corr_matrix_spearman_new.index
sns.set(font_scale = 1.2)
plt.figure(figsize = (10, 10))
hm = sns.heatmap(corr_matrix_spearman_new, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()

#Análise de Correlação das Variáveis - Kendall

import numpy as np

import time

import seaborn as sns

import matplotlib.pyplot as plt

start = time.time()

corr_matrix_kendall = df.corr('kendall')

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 62.92 min

corr_matrix_kendall["CONT_STATUS_INT"].sort_values(ascending = False)

k = 20

corr_matrix_kendall.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT']

#cols = corr_matrix_kendall.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
#cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale = 1.2)
#plt.figure(figsize = (28, 28))
#hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
#plt.show()

corr_matrix_kendall_new = df[corr_matrix_kendall['CONT_STATUS_INT'].nlargest(20).index].corr('kendall')

cols = corr_matrix_kendall_new.index
sns.set(font_scale = 1.2)
plt.figure(figsize = (10, 10))
hm = sns.heatmap(corr_matrix_kendall_new, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()

#Análise de Correlação das Variáveis - Gráficos

#Heat Map - Pearson Correlation - Op 1

import matplotlib.pyplot as plt

import seaborn as sns

print("PEARSON CORRELATION")
print(corr_matrix_pearson_new)
plt.figure(figsize = (56, 28))
sns.heatmap(corr_matrix_pearson_new)
plt.savefig("heatmap_pearson.png", dpi = 480, orientation = 'landscape')
plt.clf()
plt.close()

#Heat Map - Pearson Correlation - Op 2

import matplotlib.pyplot as plt

import seaborn as sns

fig = plt.figure(figsize = (12,9))
sns.heatmap(corr_matrix_pearson_new, vmax = 0.8, square = True)
plt.show()

#Heat Map - Pearson Correlation - Op 3

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix_pearson_new, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix_pearson_new, mask = mask, cmap = cmap, vmax = .3, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})

#Heat Map - Pearson Correlation - Op 4

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (28,28))
sns.heatmap(corr_matrix_pearson_new, vmax = 1, square = True, annot = True, cmap = 'viridis')
plt.title('Correlation Between Different Fearures')

#Heat Map - Pearson Correlation - Op 5

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (28,28))
sns.heatmap(corr_matrix_pearson_new, vmax = 0.8, square = True)
plt.xticks(rotation = 45, fontsize = 7); plt.yticks(rotation = 45, fontsize = 7);

#Heat Map - Spearman Correlation - Op 1

import matplotlib.pyplot as plt

import seaborn as sns

print("SPEARMAN CORRELATION")
print(corr_matrix_spearman_new)
plt.figure(figsize = (56, 28))
sns.heatmap(corr_matrix_spearman_new)
plt.savefig("heatmap_spearman.png", dpi = 480, orientation = 'landscape')
plt.clf()
plt.close()

#Heat Map - Spearman Correlation - Op 2

import matplotlib.pyplot as plt

import seaborn as sns

fig = plt.figure(figsize = (12,9))
sns.heatmap(corr_matrix_spearman_new, vmax = 0.8, square = True)
plt.show()

#Heat Map - Spearman Correlation - Op 3

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix_spearman_new, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix_spearman_new, mask = mask, cmap = cmap, vmax = .3, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})

#Heat Map - Spearman Correlation - Op 4

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (28,28))
sns.heatmap(corr_matrix_spearman_new, vmax = 1, square = True, annot = True, cmap = 'viridis')
plt.title('Correlation Between Different Fearures')

#Heat Map - Spearman Correlation - Op 5

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (28,28))
sns.heatmap(corr_matrix_spearman_new, vmax = 0.8, square = True)
plt.xticks(rotation = 45, fontsize = 7); plt.yticks(rotation = 45, fontsize = 7);

#Heat Map - Kendall Correlation - Op 1

import matplotlib.pyplot as plt

import seaborn as sns

print("KENDALL CORRELATION")
print(corr_matrix_kendall_new)
plt.figure(figsize = (56, 28))
sns.heatmap(corr_matrix_kendall_new)
plt.savefig("heatmap_kendall.png", dpi = 480, orientation = 'landscape')
plt.clf()
plt.close()

#Heat Map - Kendall Correlation - Op 2

import matplotlib.pyplot as plt

import seaborn as sns

fig = plt.figure(figsize = (12,9))
sns.heatmap(corr_matrix_kendall_new, vmax = 0.8, square = True)
plt.show()

#Heat Map - Kendall Correlation - Op 3

import numpy as np

import matplotlib.pyplot as plt

import seaborn as sns

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix_kendall_new, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix_kendall_new, mask = mask, cmap = cmap, vmax = .3, center = 0,
            square = True, linewidths = .5, cbar_kws = {"shrink": .5})

#Heat Map - Kendall Correlation - Op 4

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (28,28))
sns.heatmap(corr_matrix_kendall_new, vmax = 1, square = True, annot = True, cmap = 'viridis')
plt.title('Correlation Between Different Fearures')

#Heat Map - Kendall Correlation - Op 5

import matplotlib.pyplot as plt

import seaborn as sns

plt.figure(figsize = (28,28))
sns.heatmap(corr_matrix_kendall_new, vmax = 0.8, square = True)
plt.xticks(rotation = 45, fontsize = 7); plt.yticks(rotation = 45, fontsize = 7);

#Análise de Valores Nulos ou Não Existentes

df.isnull().sum()

df.isna().sum()

for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(col)
        
#Análise de Valores Nulos ou Não Existentes - Data Frame        
        
total = df.isnull().sum().sort_values(ascending = False)

percent_1 = df.isnull().sum()/df.isnull().count()*100

percent_2 = (round(percent_1, 2)).sort_values(ascending = False)

missing_data = pd.concat([total, percent_2], axis = 1, keys = ['Total', '%'])

missing_data[missing_data.Total > 0].sort_values(ascending = False, by = 'Total').head(50)

#Preenchimento dos Valores Nulos

pd.value_counts(df['PERGUNTA_TRIAGEM_02'])
df['PERGUNTA_TRIAGEM_02'].count()
df['PERGUNTA_TRIAGEM_02'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_03'])
df['PERGUNTA_TRIAGEM_03'].count()
df['PERGUNTA_TRIAGEM_03'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_04'])
df['PERGUNTA_TRIAGEM_04'].count()
df['PERGUNTA_TRIAGEM_04'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_05'])
df['PERGUNTA_TRIAGEM_05'].count()
df['PERGUNTA_TRIAGEM_05'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_06'])
df['PERGUNTA_TRIAGEM_06'].count()
df['PERGUNTA_TRIAGEM_06'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_07'])
df['PERGUNTA_TRIAGEM_07'].count()
df['PERGUNTA_TRIAGEM_07'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_08'])
df['PERGUNTA_TRIAGEM_08'].count()
df['PERGUNTA_TRIAGEM_08'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_09'])
df['PERGUNTA_TRIAGEM_09'].count()
df['PERGUNTA_TRIAGEM_09'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_10'])
df['PERGUNTA_TRIAGEM_10'].count()
df['PERGUNTA_TRIAGEM_10'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_11'])
df['PERGUNTA_TRIAGEM_11'].count()
df['PERGUNTA_TRIAGEM_11'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_12'])
df['PERGUNTA_TRIAGEM_12'].count()
df['PERGUNTA_TRIAGEM_12'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_13'])
df['PERGUNTA_TRIAGEM_13'].count()
df['PERGUNTA_TRIAGEM_13'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_14'])
df['PERGUNTA_TRIAGEM_14'].count()
df['PERGUNTA_TRIAGEM_14'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_15'])
df['PERGUNTA_TRIAGEM_15'].count()
df['PERGUNTA_TRIAGEM_15'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_16'])
df['PERGUNTA_TRIAGEM_16'].count()
df['PERGUNTA_TRIAGEM_16'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_17'])
df['PERGUNTA_TRIAGEM_17'].count()
df['PERGUNTA_TRIAGEM_17'].fillna(-1, inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_18'])
df['PERGUNTA_TRIAGEM_18'].count()
df['PERGUNTA_TRIAGEM_18'].fillna(-1, inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_19'])
df['PERGUNTA_TRIAGEM_19'].count()
df['PERGUNTA_TRIAGEM_19'].fillna(-1, inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_20'])
df['PERGUNTA_TRIAGEM_20'].count()
df['PERGUNTA_TRIAGEM_20'].fillna(-1, inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_21'])
df['PERGUNTA_TRIAGEM_21'].count()
df['PERGUNTA_TRIAGEM_21'].fillna(-1, inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_22'])
df['PERGUNTA_TRIAGEM_22'].count()
df['PERGUNTA_TRIAGEM_22'].fillna(-1, inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_23'])
df['PERGUNTA_TRIAGEM_23'].count()
df['PERGUNTA_TRIAGEM_23'].fillna(-1, inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_24'])
df['PERGUNTA_TRIAGEM_24'].count()
df['PERGUNTA_TRIAGEM_24'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_25'])
df['PERGUNTA_TRIAGEM_25'].count()
df['PERGUNTA_TRIAGEM_25'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_26'])
df['PERGUNTA_TRIAGEM_26'].count()
df['PERGUNTA_TRIAGEM_26'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_27'])
df['PERGUNTA_TRIAGEM_27'].count()
df['PERGUNTA_TRIAGEM_27'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_28'])
df['PERGUNTA_TRIAGEM_28'].count()
df['PERGUNTA_TRIAGEM_28'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_29'])
df['PERGUNTA_TRIAGEM_29'].count()
df['PERGUNTA_TRIAGEM_29'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_30'])
df['PERGUNTA_TRIAGEM_30'].count()
df['PERGUNTA_TRIAGEM_30'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_31'])
df['PERGUNTA_TRIAGEM_31'].count()
df['PERGUNTA_TRIAGEM_31'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_32'])
df['PERGUNTA_TRIAGEM_32'].count()
df['PERGUNTA_TRIAGEM_32'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_TRIAGEM_33'])
df['PERGUNTA_TRIAGEM_33'].count()
df['PERGUNTA_TRIAGEM_33'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_02'])
df['PERGUNTA_FICHA_EME_PA_02'].count()
df['PERGUNTA_FICHA_EME_PA_02'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_03'])
df['PERGUNTA_FICHA_EME_PA_03'].count()
df['PERGUNTA_FICHA_EME_PA_03'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_04'])
df['PERGUNTA_FICHA_EME_PA_04'].count()
df['PERGUNTA_FICHA_EME_PA_04'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_05'])
df['PERGUNTA_FICHA_EME_PA_05'].count()
df['PERGUNTA_FICHA_EME_PA_05'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_06'])
df['PERGUNTA_FICHA_EME_PA_06'].count()
df['PERGUNTA_FICHA_EME_PA_06'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_07'])
df['PERGUNTA_FICHA_EME_PA_07'].count()
df['PERGUNTA_FICHA_EME_PA_07'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_08'])
df['PERGUNTA_FICHA_EME_PA_08'].count()
df['PERGUNTA_FICHA_EME_PA_08'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_09'])
df['PERGUNTA_FICHA_EME_PA_09'].count()
df['PERGUNTA_FICHA_EME_PA_09'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_10'])
df['PERGUNTA_FICHA_EME_PA_10'].count()
df['PERGUNTA_FICHA_EME_PA_10'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_11'])
df['PERGUNTA_FICHA_EME_PA_11'].count()
df['PERGUNTA_FICHA_EME_PA_11'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_12'])
df['PERGUNTA_FICHA_EME_PA_12'].count()
df['PERGUNTA_FICHA_EME_PA_12'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_13'])
df['PERGUNTA_FICHA_EME_PA_13'].count()
df['PERGUNTA_FICHA_EME_PA_13'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_14'])
df['PERGUNTA_FICHA_EME_PA_14'].count()
df['PERGUNTA_FICHA_EME_PA_14'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_15'])
df['PERGUNTA_FICHA_EME_PA_15'].count()
df['PERGUNTA_FICHA_EME_PA_15'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_16'])
df['PERGUNTA_FICHA_EME_PA_16'].count()
df['PERGUNTA_FICHA_EME_PA_16'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_17'])
df['PERGUNTA_FICHA_EME_PA_17'].count()
df['PERGUNTA_FICHA_EME_PA_17'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_18'])
df['PERGUNTA_FICHA_EME_PA_18'].count()
df['PERGUNTA_FICHA_EME_PA_18'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_19'])
df['PERGUNTA_FICHA_EME_PA_19'].count()
df['PERGUNTA_FICHA_EME_PA_19'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_20'])
df['PERGUNTA_FICHA_EME_PA_20'].count()
df['PERGUNTA_FICHA_EME_PA_20'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_21'])
df['PERGUNTA_FICHA_EME_PA_21'].count()
df['PERGUNTA_FICHA_EME_PA_21'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_22'])
df['PERGUNTA_FICHA_EME_PA_22'].count()
df['PERGUNTA_FICHA_EME_PA_22'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_23'])
df['PERGUNTA_FICHA_EME_PA_23'].count()
df['PERGUNTA_FICHA_EME_PA_23'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_24'])
df['PERGUNTA_FICHA_EME_PA_24'].count()
df['PERGUNTA_FICHA_EME_PA_24'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_25'])
df['PERGUNTA_FICHA_EME_PA_25'].count()
df['PERGUNTA_FICHA_EME_PA_25'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['PERGUNTA_FICHA_EME_PA_26'])
df['PERGUNTA_FICHA_EME_PA_26'].count()
df['PERGUNTA_FICHA_EME_PA_26'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['COD_CID_ADM'])
df['COD_CID_ADM'].count()
df['COD_CID_ADM'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['COD_CID_ADM_PRINCIPAL'])
df['COD_CID_ADM_PRINCIPAL'].count()
df['COD_CID_ADM_PRINCIPAL'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['INDICADOR_IAM_CSST'])
df['INDICADOR_IAM_CSST'].count()
df['INDICADOR_IAM_CSST'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df['CSV_MIN_TEMPERATURA'])
df['CSV_MIN_TEMPERATURA'].count()
df['CSV_MIN_TEMPERATURA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MAX_TEMPERATURA'])
df['CSV_MAX_TEMPERATURA'].count()
df['CSV_MAX_TEMPERATURA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MEDIA_TEMPERATURA'])
df['CSV_MEDIA_TEMPERATURA'].count()
df['CSV_MEDIA_TEMPERATURA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MIN_TA_SISTOLICA'])
df['CSV_MIN_TA_SISTOLICA'].count()
df['CSV_MIN_TA_SISTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MAX_TA_SISTOLICA'])
df['CSV_MAX_TA_SISTOLICA'].count()
df['CSV_MAX_TA_SISTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MEDIA_TA_SISTOLICA'])
df['CSV_MEDIA_TA_SISTOLICA'].count()
df['CSV_MEDIA_TA_SISTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MIN_TA_DIASTOLICA'])
df['CSV_MIN_TA_DIASTOLICA'].count()
df['CSV_MIN_TA_DIASTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MAX_TA_DIASTOLICA'])
df['CSV_MAX_TA_DIASTOLICA'].count()
df['CSV_MAX_TA_DIASTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MEDIA_TA_DIASTOLICA'])
df['CSV_MEDIA_TA_DIASTOLICA'].count()
df['CSV_MEDIA_TA_DIASTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MIN_FREQ_RESPIRATORIA'])
df['CSV_MIN_FREQ_RESPIRATORIA'].count()
df['CSV_MIN_FREQ_RESPIRATORIA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MAX_FREQ_RESPIRATORIA'])
df['CSV_MAX_FREQ_RESPIRATORIA'].count()
df['CSV_MAX_FREQ_RESPIRATORIA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MEDIA_FREQ_RESPIRATORIA'])
df['CSV_MEDIA_FREQ_RESPIRATORIA'].count()
df['CSV_MEDIA_FREQ_RESPIRATORIA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MIN_GLICEMIA_DIGITAL'])
df['CSV_MIN_GLICEMIA_DIGITAL'].count()
df['CSV_MIN_GLICEMIA_DIGITAL'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MAX_GLICEMIA_DIGITAL'])
df['CSV_MAX_GLICEMIA_DIGITAL'].count()
df['CSV_MAX_GLICEMIA_DIGITAL'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MEDIA_GLICEMIA_DIGITAL'])
df['CSV_MEDIA_GLICEMIA_DIGITAL'].count()
df['CSV_MEDIA_GLICEMIA_DIGITAL'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MIN_SATURACAO_O2'])
df['CSV_MIN_SATURACAO_O2'].count()
df['CSV_MIN_SATURACAO_O2'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MAX_SATURACAO_O2'])
df['CSV_MAX_SATURACAO_O2'].count()
df['CSV_MAX_SATURACAO_O2'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MEDIA_SATURACAO_O2'])
df['CSV_MEDIA_SATURACAO_O2'].count()
df['CSV_MEDIA_SATURACAO_O2'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MIN_PRESSAO_ARTERIAL'])
df['CSV_MIN_PRESSAO_ARTERIAL'].count()
df['CSV_MIN_PRESSAO_ARTERIAL'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MAX_PRESSAO_ARTERIAL'])
df['CSV_MAX_PRESSAO_ARTERIAL'].count()
df['CSV_MAX_PRESSAO_ARTERIAL'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MEDIA_PRESSAO_ARTERIAL'])
df['CSV_MEDIA_PRESSAO_ARTERIAL'].count()
df['CSV_MEDIA_PRESSAO_ARTERIAL'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MIN_FREQ_CARDIACA'])
df['CSV_MIN_FREQ_CARDIACA'].count()
df['CSV_MIN_FREQ_CARDIACA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MAX_FREQ_CARDIACA'])
df['CSV_MAX_FREQ_CARDIACA'].count()
df['CSV_MAX_FREQ_CARDIACA'].fillna(-1, inplace = True)

pd.value_counts(df['CSV_MEDIA_FREQ_CARDIACA'])
df['CSV_MEDIA_FREQ_CARDIACA'].count()
df['CSV_MEDIA_FREQ_CARDIACA'].fillna(-1, inplace = True)

pd.value_counts(df['TEMPO_ATD_ULT_ATENDIMENTO'])
df['TEMPO_ATD_ULT_ATENDIMENTO'].count()
df['TEMPO_ATD_ULT_ATENDIMENTO'].fillna(-1, inplace = True)

pd.value_counts(df['MEDIA_TEMPO_ATD_ATD_ANTERIOR'])
df['MEDIA_TEMPO_ATD_ATD_ANTERIOR'].count()
df['MEDIA_TEMPO_ATD_ATD_ANTERIOR'].fillna(-1, inplace = True)

pd.value_counts(df['TEMPO_PERM_ULT_INTERNAMENTO'])
df['TEMPO_PERM_ULT_INTERNAMENTO'].count()
df['TEMPO_PERM_ULT_INTERNAMENTO'].fillna(-1, inplace = True)

pd.value_counts(df['MEDIA_TEMPO_PERM_INT_ANT'])
df['MEDIA_TEMPO_PERM_INT_ANT'].count()
df['MEDIA_TEMPO_PERM_INT_ANT'].fillna(-1, inplace = True)

#Verificação dos Valores Nulos Ainda Existentes

for col in df.columns:
    if df[col].isnull().sum() > 0:
        print(col)

#Exclusão das Colunas Que Não Serão Utilizadas
        
df_trat = df.drop('DATA_ADM', axis = 1)

df_trat = df_trat.drop('DESC_DIA_DA_SEMANA', axis = 1)

df_trat = df_trat.drop('REGISTRO', axis = 1)

df_trat = df_trat.drop('NOME_PACIENTE', axis = 1)

df_trat = df_trat.drop('DESC_SEXO', axis = 1)

df_trat = df_trat.drop('DESC_FAIXA_ETARIA', axis = 1)

df_trat = df_trat.drop('NUM_TRATAMENTO', axis = 1)

df_trat = df_trat.drop('DATA_ADMISSAO', axis = 1)

df_trat = df_trat.drop('DESC_DATA_ADMISSAO', axis = 1)

df_trat = df_trat.drop('DESC_CONVENIO', axis = 1)

df_trat = df_trat.drop('COD_EMPRESA', axis = 1)

df_trat = df_trat.drop('DESC_EMPRESA', axis = 1)

df_trat = df_trat.drop('COD_PLANTONISTA', axis = 1)

df_trat = df_trat.drop('NOME_PLANTONISTA', axis = 1)

df_trat = df_trat.drop('PERGUNTA_TRIAGEM_01', axis = 1)

df_trat = df_trat.drop('PERGUNTA_FICHA_EME_PA_01', axis = 1)

df_trat = df_trat.drop('PERGUNTA_CONSULTA_P_07', axis = 1)

df_trat = df_trat.drop('PERGUNTA_CONSULTA_P_13', axis = 1)

df_trat = df_trat.drop('DESC_DATA_ALTA', axis = 1)

df_trat = df_trat.drop('TEMPO_ATENDIMENTO', axis = 1)

df_trat = df_trat.drop('DESC_STATUS_INT', axis = 1)

df_trat = df_trat.drop('NUM_INT_POSTERIOR', axis = 1)

df_trat = df_trat.drop('TEMPO_PERM_INT_POSTERIOR', axis = 1)

df_trat = df_trat.drop('CONTADOR', axis = 1)

#Verificação dos Valores Nulos Ainda Existentes

for col in df_trat.columns:
    if df_trat[col].isnull().sum() > 0:
        print(col)
        
#Estatísticas Descritivas da Base de Dados - Após Tratamento dos Dados
        
df_trat.info([''])

#Criação das Bases X e Y

X = df_trat

Y = df_trat['CONT_STATUS_INT']

X = X.drop('CONT_STATUS_INT', axis = 1)

#Estatísticas Descritivas da Base de Dados - X - Após Remoção da Coluna Target

X.info([''])

del df

del df_trat

#Aplicação do Label Encoder na Base X

from sklearn.preprocessing import LabelEncoder

lbl_enc_X = LabelEncoder()

X['DESC_ESTACAO_ANO'] = lbl_enc_X.fit_transform(X['DESC_ESTACAO_ANO'])

X['DESC_FINAL_SEMANA'] = lbl_enc_X.fit_transform(X['DESC_FINAL_SEMANA'])

X['DESC_BAIRRO'] = lbl_enc_X.fit_transform(X['DESC_BAIRRO'])

X['COD_CID'] = lbl_enc_X.fit_transform(X['COD_CID'])

X['DESC_GRUPO_SANGUINEO'] = lbl_enc_X.fit_transform(X['DESC_GRUPO_SANGUINEO'])

X['APEL_PLANTONISTA'] = lbl_enc_X.fit_transform(X['APEL_PLANTONISTA'])

X['CID_PREVALENTE'] = lbl_enc_X.fit_transform(X['CID_PREVALENTE'])

X['COD_CLAC_RISCO'] = lbl_enc_X.fit_transform(X['COD_CLAC_RISCO'])

X['PERGUNTA_TRIAGEM_02'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_02'])

X['PERGUNTA_TRIAGEM_03'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_03'])

X['PERGUNTA_TRIAGEM_04'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_04'])

X['PERGUNTA_TRIAGEM_05'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_05'])

X['PERGUNTA_TRIAGEM_06'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_06'])

X['PERGUNTA_TRIAGEM_07'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_07'])

X['PERGUNTA_TRIAGEM_08'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_08'])

X['PERGUNTA_TRIAGEM_09'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_09'])

X['PERGUNTA_TRIAGEM_10'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_10'])

X['PERGUNTA_TRIAGEM_11'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_11'])

X['PERGUNTA_TRIAGEM_12'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_12'])

X['PERGUNTA_TRIAGEM_13'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_13'])

X['PERGUNTA_TRIAGEM_14'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_14'])

X['PERGUNTA_TRIAGEM_15'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_15'])

X['PERGUNTA_TRIAGEM_16'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_16'])

X['PERGUNTA_TRIAGEM_24'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_24'])

X['PERGUNTA_TRIAGEM_25'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_25'])

X['PERGUNTA_TRIAGEM_26'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_26'])

X['PERGUNTA_TRIAGEM_27'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_27'])

X['PERGUNTA_TRIAGEM_28'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_28'])

X['PERGUNTA_TRIAGEM_29'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_29'])

X['PERGUNTA_TRIAGEM_30'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_30'])

X['PERGUNTA_TRIAGEM_31'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_31'])

X['PERGUNTA_TRIAGEM_32'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_32'])

X['PERGUNTA_TRIAGEM_33'] = lbl_enc_X.fit_transform(X['PERGUNTA_TRIAGEM_33'])

X['PERGUNTA_FICHA_EME_PA_02'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_02'])

X['PERGUNTA_FICHA_EME_PA_03'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_03'])

X['PERGUNTA_FICHA_EME_PA_04'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_04'])

X['PERGUNTA_FICHA_EME_PA_05'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_05'])

X['PERGUNTA_FICHA_EME_PA_06'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_06'])

X['PERGUNTA_FICHA_EME_PA_07'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_07'])

X['PERGUNTA_FICHA_EME_PA_08'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_08'])

X['PERGUNTA_FICHA_EME_PA_09'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_09'])

X['PERGUNTA_FICHA_EME_PA_10'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_10'])

X['PERGUNTA_FICHA_EME_PA_11'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_11'])

X['PERGUNTA_FICHA_EME_PA_12'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_12'])

X['PERGUNTA_FICHA_EME_PA_13'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_13'])

X['PERGUNTA_FICHA_EME_PA_14'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_14'])

X['PERGUNTA_FICHA_EME_PA_15'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_15'])

X['PERGUNTA_FICHA_EME_PA_16'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_16'])

X['PERGUNTA_FICHA_EME_PA_17'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_17'])

X['PERGUNTA_FICHA_EME_PA_18'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_18'])

X['PERGUNTA_FICHA_EME_PA_19'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_19'])

X['PERGUNTA_FICHA_EME_PA_20'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_20'])

X['PERGUNTA_FICHA_EME_PA_21'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_21'])

X['PERGUNTA_FICHA_EME_PA_22'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_22'])

X['PERGUNTA_FICHA_EME_PA_23'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_23'])

X['PERGUNTA_FICHA_EME_PA_24'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_24'])

X['PERGUNTA_FICHA_EME_PA_25'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_25'])

X['PERGUNTA_FICHA_EME_PA_26'] = lbl_enc_X.fit_transform(X['PERGUNTA_FICHA_EME_PA_26'])

X['COD_CID_ADM'] = lbl_enc_X.fit_transform(X['COD_CID_ADM'])

X['COD_CID_ADM_PRINCIPAL'] = lbl_enc_X.fit_transform(X['COD_CID_ADM_PRINCIPAL'])

X['PERGUNTA_CONSULTA_P_09'] = lbl_enc_X.fit_transform(X['PERGUNTA_CONSULTA_P_09'])

X['PERGUNTA_CONSULTA_P_10'] = lbl_enc_X.fit_transform(X['PERGUNTA_CONSULTA_P_10'])

X['PERGUNTA_CONSULTA_P_12'] = lbl_enc_X.fit_transform(X['PERGUNTA_CONSULTA_P_12'])

X['PERGUNTA_CONSULTA_P_14'] = lbl_enc_X.fit_transform(X['PERGUNTA_CONSULTA_P_14'])

X['PERGUNTA_CONSULTA_P_15'] = lbl_enc_X.fit_transform(X['PERGUNTA_CONSULTA_P_15'])

X['PERGUNTA_CONSULTA_P_16'] = lbl_enc_X.fit_transform(X['PERGUNTA_CONSULTA_P_16'])

X['INDICADOR_IAM_CSST'] = lbl_enc_X.fit_transform(X['INDICADOR_IAM_CSST'])

#Estatísticas Descritivas da Base de Dados - X - Após Aplicação do Label Encoder

X.info([''])

#Verificação da Distribuição dos Tipos de Dados das Variáveis - Base X

feature_list_X = list(X.columns)

feature_list_X_integer, feature_list_X_float, features_list_X_object = groupFeatures(feature_list_X, X)

print("# of integer feature : ", len(feature_list_X_integer))
print("# of float feature : ", len(feature_list_X_float))
print("# of object feature : ", len(features_list_X_object))

#Transformação da Base Y em Array

import numpy as np

Y = np.array(Y)

#Transformação da Base X em Array

X_list = list(X.columns)

X_list_array = np.array(X_list)

X = np.array(X)

#Escalonamento da Base X - Opção 01 - StandardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

#ETAPA 02: TESTES DO MODELO LIGHT GBM

#Parametrização do Modelo

from lightgbm import LGBMClassifier

#28° Teste - Random Search

classifier_lgbm = LGBMClassifier( max_depth = 1880, 
                                  learning_rate = 0.1,
                                  num_leaves = 100,
                                  n_estimators = 4500,
                                  min_data_in_leaf = 140,
                                  n_jobs = 4 )

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(classifier_lgbm.get_params())

# Fit e Predição

import time
start = time.time()

classifier_lgbm.fit(X, Y)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 10.65 min

#Learning Curve - X, Y

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import learning_curve

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time

start = time.time()
train_sizes_lc_lgbm, train_scores_lc_lgbm, test_scores_lc_lgbm = learning_curve( estimator = classifier_lgbm,
                                                                                 X = X,
                                                                                 y = Y,
                                                                                 train_sizes = np.linspace(0.1, 1.0, 20),
                                                                                 cv = 10,
                                                                                 scoring = make_scorer(matthews_corrcoef),
                                                                                 shuffle = True,
                                                                                 random_state = 42 )
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 1048.20 min

train_mean_lc_lgbm = np.mean(train_scores_lc_lgbm, axis = 1)
train_std_lc_lgbm = np.std(train_scores_lc_lgbm, axis = 1)
test_mean_lc_lgbm = np.mean(test_scores_lc_lgbm, axis = 1)
test_std_lc_lgbm = np.std(test_scores_lc_lgbm, axis = 1)

plt.figure(figsize = (14, 7))
plt.plot( train_sizes_lc_lgbm, 
          train_mean_lc_lgbm,
          color = 'blue', 
          marker = 'o',
          markersize = 5, 
          label = 'Training MCC' )

plt.fill_between( train_sizes_lc_lgbm,
                  train_mean_lc_lgbm + train_std_lc_lgbm,
                  train_mean_lc_lgbm - train_std_lc_lgbm,
                  alpha = 0.15, 
                  color ='blue' )

plt.plot( train_sizes_lc_lgbm, 
          test_mean_lc_lgbm,
          color = 'green', 
          linestyle = '--',
          marker = 's', 
          markersize = 5,
          label = 'Validation MCC')

plt.fill_between(train_sizes_lc_lgbm,
                 test_mean_lc_lgbm + test_std_lc_lgbm,
                 test_mean_lc_lgbm - test_std_lc_lgbm,
                 alpha = 0.15, 
                 color = 'green')

plt.minorticks_on()
plt.grid(b = True, which = 'both', color = '#666666', linestyle = '-', alpha = 0.2, axis = 'both', linewidth = 1)

#plt.grid()
plt.xlabel('Number of Training Samples')
plt.ylabel('Matthews Correlation Coefficient (MCC)')
plt.legend(loc = 'lower right')
plt.ylim([0.5, 1.03])
plt.tight_layout()
#plt.savefig('images/06_05.png', dpi = 300)
plt.show()

#Validation Curve - X, Y

from sklearn.model_selection import validation_curve

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time

param_range_vc = [3500, 4000, 4500, 5000, 5500, 6000]

start = time.time()
train_scores_vc_lgbm, test_scores_vc_lgbm = validation_curve( estimator = classifier_lgbm, 
                                                              X = X, 
                                                              y = Y, 
                                                              param_name = 'n_estimators', 
                                                              param_range = param_range_vc,
                                                              scoring = make_scorer(matthews_corrcoef),
                                                              cv = 10 )
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 667.86 min

train_mean_vc_lgbm = np.mean(train_scores_vc_lgbm, axis = 1)
train_std_vc_lgbm = np.std(train_scores_vc_lgbm, axis = 1)
test_mean_vc_lgbm = np.mean(test_scores_vc_lgbm, axis = 1)
test_std_vc_lgbm = np.std(test_scores_vc_lgbm, axis = 1)

plt.figure(figsize = (14, 7))
plt.plot( param_range_vc, 
          train_mean_vc_lgbm, 
          color = 'blue', 
          marker = 'o', 
          markersize = 5, 
          label = 'Training MCC' )

plt.fill_between( param_range_vc, 
                  train_mean_vc_lgbm + train_std_vc_lgbm,
                  train_mean_vc_lgbm - train_std_vc_lgbm, 
                  alpha = 0.15,
                  color = 'blue' )

plt.plot( param_range_vc, 
          test_mean_vc_lgbm, 
          color = 'green', 
          linestyle = '--', 
          marker = 's', 
          markersize = 5, 
          label = 'Validation MCC' )

plt.fill_between( param_range_vc, 
                  test_mean_vc_lgbm + test_std_vc_lgbm,
                  test_mean_vc_lgbm - test_std_vc_lgbm, 
                  alpha = 0.15, 
                  color = 'green')

plt.minorticks_on()
plt.grid(b = True, which = 'both', color = '#666666', linestyle = '-', alpha = 0.2, axis = 'both', linewidth = 1)

#plt.grid()
#plt.xscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('Parameter n Estimator')
plt.ylabel('Matthews Correlation Coefficient (MCC)')
plt.ylim([0.5, 1.03])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi = 300)
plt.show()

#CrossValidate - 10 KFolds - X_train, Y_train - Múltiplas Métricas

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import make_scorer
from sklearn.metrics import matthews_corrcoef
from sklearn.metrics import cohen_kappa_score
from sklearn.model_selection import StratifiedKFold

metricas_cross_validate = { 'accuracy': 'accuracy',
                            'balanced_accuracy': 'balanced_accuracy',
                            'average_precision': 'average_precision',
                            'f1': 'f1',
                            'precision': 'precision',
                            'recall': 'recall',
                            'roc_auc': 'roc_auc',
                            'ms_matthews_corrcoef': make_scorer(matthews_corrcoef),
                            'ms_cohen_kappa_score': make_scorer(cohen_kappa_score) }

skf_lgbm = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

start = time.time()
result_lgbm_cross_validate = cross_validate( estimator = classifier_lgbm, 
                                             X = X, 
                                             y = Y, 
                                             cv = skf_lgbm, 
                                             scoring = metricas_cross_validate, 
                                             return_train_score = False )
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 97.83 min

sorted(result_lgbm_cross_validate.keys())

pd.set_option('display.float_format', lambda x: '%.3f' % x)

cv_fit_time_lgbm = result_lgbm_cross_validate['fit_time'].mean() 
print('Fit Time LGBM: ' + str(cv_fit_time_lgbm))

cv_score_time_lgbm = result_lgbm_cross_validate['score_time'].mean()
print('Score Time LGBM: ' + str(cv_score_time_lgbm))

cv_test_accuracy_lgbm = result_lgbm_cross_validate['test_accuracy'].mean()
print('Test Accuracy LGBM: ' + str(cv_test_accuracy_lgbm))

cv_test_accuracy_std_lgbm = result_lgbm_cross_validate['test_accuracy'].std()
print('STD Test Accuracy LGBM: ' + str(cv_test_accuracy_std_lgbm))

cv_test_balanced_accuracy_lgbm = result_lgbm_cross_validate['test_balanced_accuracy'].mean()
print('Test Balanced Accuracy LGBM: ' + str(cv_test_balanced_accuracy_lgbm))

cv_test_balanced_accuracy_std_lgbm = result_lgbm_cross_validate['test_balanced_accuracy'].std()
print('STD Test Balanced Accuracy LGBM: ' + str(cv_test_balanced_accuracy_std_lgbm))

cv_test_average_precision_lgbm = result_lgbm_cross_validate['test_average_precision'].mean()
print('Test Area Under the Precision-Recall Curve LGBM: ' + str(cv_test_average_precision_lgbm))

cv_test_average_precision_std_lgbm = result_lgbm_cross_validate['test_average_precision'].std()
print('STD Test Area Under the Precision-Recall Curve LGBM: ' + str(cv_test_average_precision_std_lgbm))

cv_test_f1_lgbm = result_lgbm_cross_validate['test_f1'].mean()
print('Test F1 LGBM: ' + str(cv_test_f1_lgbm))

cv_test_f1_std_lgbm = result_lgbm_cross_validate['test_f1'].std()
print('STD Test F1 LGBM: ' + str(cv_test_f1_std_lgbm))

cv_test_precision_lgbm = result_lgbm_cross_validate['test_precision'].mean()
print('Test Precision LGBM: ' + str(cv_test_precision_lgbm))

cv_test_precision_std_lgbm = result_lgbm_cross_validate['test_precision'].std()
print('STD Test Precision LGBM: ' + str(cv_test_precision_std_lgbm))

cv_test_recall_lgbm = result_lgbm_cross_validate['test_recall'].mean()
print('Test Recall LGBM: ' + str(cv_test_recall_lgbm))

cv_test_recall_std_lgbm = result_lgbm_cross_validate['test_recall'].std()
print('STD Test Recall LGBM: ' + str(cv_test_recall_std_lgbm))

cv_test_roc_auc_lgbm = result_lgbm_cross_validate['test_roc_auc'].mean()
print('Test ROC AUC LGBM: ' + str(cv_test_roc_auc_lgbm))

cv_test_roc_auc_std_lgbm = result_lgbm_cross_validate['test_roc_auc'].std()
print('STD Test ROC AUC LGBM: ' + str(cv_test_roc_auc_std_lgbm))

cv_test_ms_matthews_corrcoef_lgbm = result_lgbm_cross_validate['test_ms_matthews_corrcoef'].mean()
print('Test MCC Score LGBM: ' + str(cv_test_ms_matthews_corrcoef_lgbm))

cv_test_ms_matthews_corrcoef_std_lgbm = result_lgbm_cross_validate['test_ms_matthews_corrcoef'].std()
print('STD Test MCC Score LGBM: ' + str(cv_test_ms_matthews_corrcoef_std_lgbm))

cv_test_ms_cohen_kappa_score_lgbm = result_lgbm_cross_validate['test_ms_cohen_kappa_score'].mean()
print('Test Cohen Kappa Score LGBM: ' + str(cv_test_ms_cohen_kappa_score_lgbm))

cv_test_ms_cohen_kappa_score_std_lgbm = result_lgbm_cross_validate['test_ms_cohen_kappa_score'].std()
print('STD Test Cohen Kappa Score LGBM: ' + str(cv_test_ms_cohen_kappa_score_std_lgbm))

X_train, Y_train
Fit Time LGBM: 202.0697096824646
Score Time LGBM: 4.976563763618469
Test Accuracy LGBM: 0.9530433902981278
STD Test Accuracy LGBM: 0.001662769576048885
Test Balanced Accuracy LGBM: 0.8513732523800291
STD Test Balanced Accuracy LGBM: 0.006998467757850687
Test Average Precision LGBM: 0.8728624149015947
STD Test Average Precision LGBM: 0.008354670382306537
Test F1 LGBM: 0.7792703747396165
STD Test F1 LGBM: 0.008894416038862877
Test Precision LGBM: 0.8505524279520816
STD Test Precision LGBM: 0.011100482287342337
Test Recall LGBM: 0.7192379286516368
STD Test Recall LGBM: 0.014548789159096751
Test ROC AUC LGBM: 0.9770871136991633
STD Test ROC AUC LGBM: 0.0018364065511548052
Test MCC Score LGBM: 0.7566008536644395
STD Test MCC Score LGBM: 0.009331447110967587
Test Cohen Kappa Score LGBM: 0.7532080577673416
STD Test Cohen Kappa Score LGBM: 0.009730196206723458

X, Y
Tempo de Execução: 137.80 min
Fit Time LGBM: 777.0959687709808
Score Time LGBM: 49.66205992698669
Test Accuracy LGBM: 0.9537287857295776
STD Test Accuracy LGBM: 0.0015271055729099091
Test Balanced Accuracy LGBM: 0.8549317580982132
STD Test Balanced Accuracy LGBM: 0.006140423688479952
Test Area Under the Precision-Recall Curve LGBM: 0.8782052672646031
STD Test Area Under the Precision-Recall Curve LGBM: 0.004721530154045778
Test F1 LGBM: 0.785085877638865
STD Test F1 LGBM: 0.008039643945025085
Test Precision LGBM: 0.8545990127773262
STD Test Precision LGBM: 0.007588798295477309
Test Recall LGBM: 0.7261497356602808
STD Test Recall LGBM: 0.012470995813284006
Test ROC AUC LGBM: 0.9774092821641254
STD Test ROC AUC LGBM: 0.0011977145076197508
Test MCC Score LGBM: 0.762557379191694
STD Test MCC Score LGBM: 0.008479202316110213
Test Cohen Kappa Score LGBM: 0.7593539429571322
STD Test Cohen Kappa Score LGBM: 0.008833813096903865

X, Y - Sem Zeros
Fit Time LGBM: 560.3736342906952
Score Time LGBM: 26.57514214515686
Test Accuracy LGBM: 0.9578410913500196
STD Test Accuracy LGBM: 0.0017292408357070176
Test Balanced Accuracy LGBM: 0.8581943144358396
STD Test Balanced Accuracy LGBM: 0.00705231593060965
Test Area Under the Precision-Recall Curve LGBM: 0.8801283490529863
STD Test Area Under the Precision-Recall Curve LGBM: 0.007599429206194305
Test F1 LGBM: 0.7913908953040439
STD Test F1 LGBM: 0.009665516790146362
Test Precision LGBM: 0.8634056799307857
STD Test Precision LGBM: 0.008938813856634103
Test Recall LGBM: 0.7306084883732431
STD Test Recall LGBM: 0.014187551790655124
Test ROC AUC LGBM: 0.9787701114052172
STD Test ROC AUC LGBM: 0.0016631311674953086
Test MCC Score LGBM: 0.771489455009567
STD Test MCC Score LGBM: 0.01011212196428209
Test Cohen Kappa Score LGBM: 0.7681264475605192
STD Test Cohen Kappa Score LGBM: 0.010559069218380145