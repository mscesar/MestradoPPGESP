#Setup da Pasta de Trabalho
import os
os.chdir('C:/Users/Mauricio/Dropbox/MauricioPessoal/Pós-Graduação/PPGESP/Projeto Final/Estudo Python/Bases Cárdio Pulmonar')

import warnings
warnings.filterwarnings("ignore")

#ETAPA 01: TRATAMENTO DOS DADOS

#Importação do Pandas

import pandas as pd

#Leitura da Base de Dados

#Criação da Base df_julho_19 com Todas as Informações Disponíveis

import time

start = time.time()
df_aux = pd.read_excel('TB_IND_ASSIST_EME_0419_0819_V6.xlsx')
df_full = pd.read_excel('TB_IND_ASSIST_EME_0115_0819_V6.xlsx')
end = time.time()

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 11.32 min
#Tempo de Execução: 20.25 min
#Tempo de Execução: 12.79 min
#Tempo de Execução: 46.79 min

df_julho_19 = df_aux.loc[(df_aux['NUM_MES'] == 7)].copy()

#del df_aux

#Criação da Base df_julho_19_triagem A Fim de Mapear Apenas as Informações Disponíveis Após a Realização da Triagem

df_julho_19_triagem = df_julho_19.copy()

#Criação da Base df_julho_19_plantonista A Fim de Mapear Apenas as Informações Disponíveis Após a Realização da Consulta com o Plantonista

df_julho_19_plantonista = df_julho_19.copy()

#Verificação da Quantidade de Linhas e Colunas da Base de Dados

df_julho_19.shape
#(1674, 652)

#Descrição do Index

df_julho_19.index

#Contagem de Dados Não Nulos

df_julho_19.count()

#Verificação do Nome Das Colunas da Base de Dados

df_julho_19.columns

df_julho_19.columns.values

#Visualização das n Primeiras Linhas

df_julho_19.head(10)

#Visualização das n Últimas Linhas

df_julho_19.tail(10)

#Estatísticas Descritivas da Base de Dados

df_julho_19.info([''])

df_julho_19.describe()

#Verificação da Distribuição dos Tipos de Dados das Variáveis

feature_list_df_julho_19 = list(df_julho_19.columns)

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

feature_list_df_julho_19_integer, feature_list_df_julho_19_float, features_list_df_julho_19_object = groupFeatures(feature_list_df_julho_19, df_julho_19)
print("# of integer feature : ", len(feature_list_df_julho_19_integer))
print("# of float feature : ", len(feature_list_df_julho_19_float))
print("# of object feature : ", len(features_list_df_julho_19_object))

#Verificação do % de Dados da Variável Target

all_data = df_julho_19.shape[0]
pac_internado = df_julho_19[df_julho_19['CONT_STATUS_INT'] == 1]
pac_não_internado = df_julho_19[df_julho_19['CONT_STATUS_INT'] == 0]

x = len(pac_internado)/all_data
y = len(pac_não_internado)/all_data

print('Pacientes Internados :', round(x*100,2),'%')
print('Pacientes Não Internados :', round(y*100,2),'%')

#Verificação do % de Dados da Variável Target - Gráfico - Op 1

import matplotlib.pyplot as plt

labels = ['Pacientes Não Internados','Pacientes Internados']
classes = pd.value_counts(df_julho_19['CONT_STATUS_INT'], sort = True)
plt.figure(figsize = (14, 7))
classes.plot(kind = 'bar', rot = 0)
plt.title("Target Class Distribution")
plt.xticks(range(2), labels)
plt.xlabel("Class")
plt.ylabel("Frequency")

#Verificação do % de Dados da Variável Target - Gráfico - Op 2

import seaborn as sns

import matplotlib.pyplot as plt

targets = df_julho_19['DESC_STATUS_INT'].values
sns.set(style = "darkgrid")
plt.figure(figsize = (14, 7))
ax = sns.countplot(x = targets)
for p in ax.patches:
    ax.annotate('{:.2f}%'.format(100*p.get_height()/len(targets)), (p.get_x() + 0.3, p.get_height() + 100))
plt.title('Target Class Distribution', fontsize = 20)
plt.xlabel('Class', fontsize = 20)
plt.ylabel('Frequency [%]', fontsize = 20)
ax.set_ylim(top = 2500)

#Verificação do % de Dados da Variável Target - Gráfico - Op 3

import matplotlib.pyplot as plt

import seaborn as sns

pac_internado = 'Pacientes Internados'
pac_nao_internado = 'Pacientes Não Internados'
fig, axes = plt.subplots(nrows = 1, ncols = 2,figsize = (14, 7))
women = df_julho_19[df_julho_19['COD_SEXO'] == 2]
men = df_julho_19[df_julho_19['COD_SEXO'] == 1]
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
women = df_julho_19[df_julho_19['COD_SEXO'] == 2]
men = df_julho_19[df_julho_19['COD_SEXO'] == 1]
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

var_ano = df_julho_19['NUM_ANO'].values
plt.figure(figsize = (14, 7))
sns.distplot(var_ano)

#Mês

var_mes = df_julho_19['NUM_MES'].values
plt.figure(figsize = (14, 7))
sns.distplot(var_mes)

#Dia da Semana

var_dia_da_semana = df_julho_19['NUM_DIA_DA_SEMANA'].values
plt.figure(figsize = (14, 7))
sns.distplot(var_dia_da_semana)

#Sexo

var_sexo = df_julho_19['COD_SEXO'].values
plt.figure(figsize = (14, 7))
sns.distplot(var_sexo)

#Idade

var_idade = df_julho_19['NUM_IDADE'].values
plt.figure(figsize = (14, 7))
sns.distplot(var_idade)

#Faixa Etária

var_faixa_etaria = df_julho_19['COD_FAIXA_ETARIA'].values
plt.figure(figsize = (14, 7))
sns.distplot(var_faixa_etaria)

#Cálculo de Simetria dos Dados

import matplotlib.pyplot as plt

skew_df_julho_19 = df_julho_19.skew()

print(skew_df_julho_19)

plt.figure(figsize = (14, 7))
skew_df_julho_19.plot()

#Análise de Correlação das Variáveis - Pearson

import numpy as np

import time

import seaborn as sns

import matplotlib.pyplot as plt

start = time.time()

corr_matrix_pearson = df_julho_19.corr('pearson')

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 4.27 min

corr_matrix_pearson['CONT_STATUS_INT'].sort_values(ascending = False)

k = 20

corr_matrix_pearson.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT']

cols = corr_matrix_pearson.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
cm = np.corrcoef(df_julho_19[cols].values.T)
sns.set(font_scale = 1.2)
plt.figure(figsize = (10, 10))
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()

corr_matrix_pearson_new = df_julho_19[corr_matrix_pearson['CONT_STATUS_INT'].nlargest(20).index].corr('pearson')

#Análise de Correlação das Variáveis - Spearman

import numpy as np

import time

import seaborn as sns

import matplotlib.pyplot as plt

start = time.time()

corr_matrix_spearman = df_julho_19.corr('spearman')

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 132.87 min

corr_matrix_spearman["CONT_STATUS_INT"].sort_values(ascending = False)

k = 20

corr_matrix_spearman.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT']

#cols = corr_matrix_spearman.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
#cm = np.corrcoef(df_julho_19[cols].values.T)
#sns.set(font_scale = 1.2)
#plt.figure(figsize = (28, 28))
#hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
#plt.show()

corr_matrix_spearman_new = df_julho_19[corr_matrix_spearman['CONT_STATUS_INT'].nlargest(20).index].corr('spearman')

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

corr_matrix_kendall = df_julho_19.corr('kendall')

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 62.92 min

corr_matrix_kendall["CONT_STATUS_INT"].sort_values(ascending = False)

k = 20

corr_matrix_kendall.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT']

#cols = corr_matrix_kendall.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
#cm = np.corrcoef(df_julho_19[cols].values.T)
#sns.set(font_scale = 1.2)
#plt.figure(figsize = (28, 28))
#hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
#plt.show()

corr_matrix_kendall_new = df_julho_19[corr_matrix_kendall['CONT_STATUS_INT'].nlargest(20).index].corr('kendall')

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

df_julho_19.isnull().sum()

df_julho_19.isna().sum()

for col in df_julho_19.columns:
    if df_julho_19[col].isnull().sum() > 0:
        print(col)
        
#Análise de Valores Nulos ou Não Existentes - Data Frame        
        
total = df_julho_19.isnull().sum().sort_values(ascending = False)

percent_1 = df_julho_19.isnull().sum()/df_julho_19.isnull().count()*100

percent_2 = (round(percent_1, 2)).sort_values(ascending = False)

missing_data = pd.concat([total, percent_2], axis = 1, keys = ['Total', '%'])

missing_data[missing_data.Total > 0].sort_values(ascending = False, by = 'Total').head(50)

#Preenchimento dos Valores Nulos

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_02'])
df_julho_19['PERGUNTA_TRIAGEM_02'].count()
df_julho_19['PERGUNTA_TRIAGEM_02'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_03'])
df_julho_19['PERGUNTA_TRIAGEM_03'].count()
df_julho_19['PERGUNTA_TRIAGEM_03'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_04'])
df_julho_19['PERGUNTA_TRIAGEM_04'].count()
df_julho_19['PERGUNTA_TRIAGEM_04'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_05'])
df_julho_19['PERGUNTA_TRIAGEM_05'].count()
df_julho_19['PERGUNTA_TRIAGEM_05'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_06'])
df_julho_19['PERGUNTA_TRIAGEM_06'].count()
df_julho_19['PERGUNTA_TRIAGEM_06'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_07'])
df_julho_19['PERGUNTA_TRIAGEM_07'].count()
df_julho_19['PERGUNTA_TRIAGEM_07'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_08'])
df_julho_19['PERGUNTA_TRIAGEM_08'].count()
df_julho_19['PERGUNTA_TRIAGEM_08'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_09'])
df_julho_19['PERGUNTA_TRIAGEM_09'].count()
df_julho_19['PERGUNTA_TRIAGEM_09'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_10'])
df_julho_19['PERGUNTA_TRIAGEM_10'].count()
df_julho_19['PERGUNTA_TRIAGEM_10'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_11'])
df_julho_19['PERGUNTA_TRIAGEM_11'].count()
df_julho_19['PERGUNTA_TRIAGEM_11'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_12'])
df_julho_19['PERGUNTA_TRIAGEM_12'].count()
df_julho_19['PERGUNTA_TRIAGEM_12'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_13'])
df_julho_19['PERGUNTA_TRIAGEM_13'].count()
df_julho_19['PERGUNTA_TRIAGEM_13'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_14'])
df_julho_19['PERGUNTA_TRIAGEM_14'].count()
df_julho_19['PERGUNTA_TRIAGEM_14'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_15'])
df_julho_19['PERGUNTA_TRIAGEM_15'].count()
df_julho_19['PERGUNTA_TRIAGEM_15'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_16'])
df_julho_19['PERGUNTA_TRIAGEM_16'].count()
df_julho_19['PERGUNTA_TRIAGEM_16'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_17'])
df_julho_19['PERGUNTA_TRIAGEM_17'].count()
df_julho_19['PERGUNTA_TRIAGEM_17'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_18'])
df_julho_19['PERGUNTA_TRIAGEM_18'].count()
df_julho_19['PERGUNTA_TRIAGEM_18'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_19'])
df_julho_19['PERGUNTA_TRIAGEM_19'].count()
df_julho_19['PERGUNTA_TRIAGEM_19'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_20'])
df_julho_19['PERGUNTA_TRIAGEM_20'].count()
df_julho_19['PERGUNTA_TRIAGEM_20'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_21'])
df_julho_19['PERGUNTA_TRIAGEM_21'].count()
df_julho_19['PERGUNTA_TRIAGEM_21'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_22'])
df_julho_19['PERGUNTA_TRIAGEM_22'].count()
df_julho_19['PERGUNTA_TRIAGEM_22'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_23'])
df_julho_19['PERGUNTA_TRIAGEM_23'].count()
df_julho_19['PERGUNTA_TRIAGEM_23'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_24'])
df_julho_19['PERGUNTA_TRIAGEM_24'].count()
df_julho_19['PERGUNTA_TRIAGEM_24'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_25'])
df_julho_19['PERGUNTA_TRIAGEM_25'].count()
df_julho_19['PERGUNTA_TRIAGEM_25'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_26'])
df_julho_19['PERGUNTA_TRIAGEM_26'].count()
df_julho_19['PERGUNTA_TRIAGEM_26'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_27'])
df_julho_19['PERGUNTA_TRIAGEM_27'].count()
df_julho_19['PERGUNTA_TRIAGEM_27'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_28'])
df_julho_19['PERGUNTA_TRIAGEM_28'].count()
df_julho_19['PERGUNTA_TRIAGEM_28'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_29'])
df_julho_19['PERGUNTA_TRIAGEM_29'].count()
df_julho_19['PERGUNTA_TRIAGEM_29'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_30'])
df_julho_19['PERGUNTA_TRIAGEM_30'].count()
df_julho_19['PERGUNTA_TRIAGEM_30'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_31'])
df_julho_19['PERGUNTA_TRIAGEM_31'].count()
df_julho_19['PERGUNTA_TRIAGEM_31'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_32'])
df_julho_19['PERGUNTA_TRIAGEM_32'].count()
df_julho_19['PERGUNTA_TRIAGEM_32'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_TRIAGEM_33'])
df_julho_19['PERGUNTA_TRIAGEM_33'].count()
df_julho_19['PERGUNTA_TRIAGEM_33'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_02'])
df_julho_19['PERGUNTA_FICHA_EME_PA_02'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_02'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_03'])
df_julho_19['PERGUNTA_FICHA_EME_PA_03'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_03'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_04'])
df_julho_19['PERGUNTA_FICHA_EME_PA_04'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_04'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_05'])
df_julho_19['PERGUNTA_FICHA_EME_PA_05'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_05'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_06'])
df_julho_19['PERGUNTA_FICHA_EME_PA_06'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_06'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_07'])
df_julho_19['PERGUNTA_FICHA_EME_PA_07'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_07'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_08'])
df_julho_19['PERGUNTA_FICHA_EME_PA_08'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_08'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_09'])
df_julho_19['PERGUNTA_FICHA_EME_PA_09'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_09'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_10'])
df_julho_19['PERGUNTA_FICHA_EME_PA_10'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_10'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_11'])
df_julho_19['PERGUNTA_FICHA_EME_PA_11'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_11'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_12'])
df_julho_19['PERGUNTA_FICHA_EME_PA_12'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_12'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_13'])
df_julho_19['PERGUNTA_FICHA_EME_PA_13'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_13'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_14'])
df_julho_19['PERGUNTA_FICHA_EME_PA_14'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_14'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_15'])
df_julho_19['PERGUNTA_FICHA_EME_PA_15'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_15'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_16'])
df_julho_19['PERGUNTA_FICHA_EME_PA_16'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_16'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_17'])
df_julho_19['PERGUNTA_FICHA_EME_PA_17'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_17'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_18'])
df_julho_19['PERGUNTA_FICHA_EME_PA_18'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_18'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_19'])
df_julho_19['PERGUNTA_FICHA_EME_PA_19'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_19'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_20'])
df_julho_19['PERGUNTA_FICHA_EME_PA_20'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_20'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_21'])
df_julho_19['PERGUNTA_FICHA_EME_PA_21'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_21'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_22'])
df_julho_19['PERGUNTA_FICHA_EME_PA_22'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_22'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_23'])
df_julho_19['PERGUNTA_FICHA_EME_PA_23'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_23'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_24'])
df_julho_19['PERGUNTA_FICHA_EME_PA_24'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_24'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_25'])
df_julho_19['PERGUNTA_FICHA_EME_PA_25'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_25'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['PERGUNTA_FICHA_EME_PA_26'])
df_julho_19['PERGUNTA_FICHA_EME_PA_26'].count()
df_julho_19['PERGUNTA_FICHA_EME_PA_26'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['COD_CID_ADM'])
df_julho_19['COD_CID_ADM'].count()
df_julho_19['COD_CID_ADM'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['COD_CID_ADM_PRINCIPAL'])
df_julho_19['COD_CID_ADM_PRINCIPAL'].count()
df_julho_19['COD_CID_ADM_PRINCIPAL'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['INDICADOR_IAM_CSST'])
df_julho_19['INDICADOR_IAM_CSST'].count()
df_julho_19['INDICADOR_IAM_CSST'].fillna('SEM REGISTRO', inplace = True)

pd.value_counts(df_julho_19['CSV_MIN_TEMPERATURA'])
df_julho_19['CSV_MIN_TEMPERATURA'].count()
df_julho_19['CSV_MIN_TEMPERATURA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MAX_TEMPERATURA'])
df_julho_19['CSV_MAX_TEMPERATURA'].count()
df_julho_19['CSV_MAX_TEMPERATURA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MEDIA_TEMPERATURA'])
df_julho_19['CSV_MEDIA_TEMPERATURA'].count()
df_julho_19['CSV_MEDIA_TEMPERATURA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MIN_TA_SISTOLICA'])
df_julho_19['CSV_MIN_TA_SISTOLICA'].count()
df_julho_19['CSV_MIN_TA_SISTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MAX_TA_SISTOLICA'])
df_julho_19['CSV_MAX_TA_SISTOLICA'].count()
df_julho_19['CSV_MAX_TA_SISTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MEDIA_TA_SISTOLICA'])
df_julho_19['CSV_MEDIA_TA_SISTOLICA'].count()
df_julho_19['CSV_MEDIA_TA_SISTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MIN_TA_DIASTOLICA'])
df_julho_19['CSV_MIN_TA_DIASTOLICA'].count()
df_julho_19['CSV_MIN_TA_DIASTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MAX_TA_DIASTOLICA'])
df_julho_19['CSV_MAX_TA_DIASTOLICA'].count()
df_julho_19['CSV_MAX_TA_DIASTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MEDIA_TA_DIASTOLICA'])
df_julho_19['CSV_MEDIA_TA_DIASTOLICA'].count()
df_julho_19['CSV_MEDIA_TA_DIASTOLICA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MIN_FREQ_RESPIRATORIA'])
df_julho_19['CSV_MIN_FREQ_RESPIRATORIA'].count()
df_julho_19['CSV_MIN_FREQ_RESPIRATORIA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MAX_FREQ_RESPIRATORIA'])
df_julho_19['CSV_MAX_FREQ_RESPIRATORIA'].count()
df_julho_19['CSV_MAX_FREQ_RESPIRATORIA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MEDIA_FREQ_RESPIRATORIA'])
df_julho_19['CSV_MEDIA_FREQ_RESPIRATORIA'].count()
df_julho_19['CSV_MEDIA_FREQ_RESPIRATORIA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MIN_GLICEMIA_DIGITAL'])
df_julho_19['CSV_MIN_GLICEMIA_DIGITAL'].count()
df_julho_19['CSV_MIN_GLICEMIA_DIGITAL'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MAX_GLICEMIA_DIGITAL'])
df_julho_19['CSV_MAX_GLICEMIA_DIGITAL'].count()
df_julho_19['CSV_MAX_GLICEMIA_DIGITAL'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MEDIA_GLICEMIA_DIGITAL'])
df_julho_19['CSV_MEDIA_GLICEMIA_DIGITAL'].count()
df_julho_19['CSV_MEDIA_GLICEMIA_DIGITAL'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MIN_SATURACAO_O2'])
df_julho_19['CSV_MIN_SATURACAO_O2'].count()
df_julho_19['CSV_MIN_SATURACAO_O2'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MAX_SATURACAO_O2'])
df_julho_19['CSV_MAX_SATURACAO_O2'].count()
df_julho_19['CSV_MAX_SATURACAO_O2'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MEDIA_SATURACAO_O2'])
df_julho_19['CSV_MEDIA_SATURACAO_O2'].count()
df_julho_19['CSV_MEDIA_SATURACAO_O2'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MIN_PRESSAO_ARTERIAL'])
df_julho_19['CSV_MIN_PRESSAO_ARTERIAL'].count()
df_julho_19['CSV_MIN_PRESSAO_ARTERIAL'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MAX_PRESSAO_ARTERIAL'])
df_julho_19['CSV_MAX_PRESSAO_ARTERIAL'].count()
df_julho_19['CSV_MAX_PRESSAO_ARTERIAL'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MEDIA_PRESSAO_ARTERIAL'])
df_julho_19['CSV_MEDIA_PRESSAO_ARTERIAL'].count()
df_julho_19['CSV_MEDIA_PRESSAO_ARTERIAL'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MIN_FREQ_CARDIACA'])
df_julho_19['CSV_MIN_FREQ_CARDIACA'].count()
df_julho_19['CSV_MIN_FREQ_CARDIACA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MAX_FREQ_CARDIACA'])
df_julho_19['CSV_MAX_FREQ_CARDIACA'].count()
df_julho_19['CSV_MAX_FREQ_CARDIACA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['CSV_MEDIA_FREQ_CARDIACA'])
df_julho_19['CSV_MEDIA_FREQ_CARDIACA'].count()
df_julho_19['CSV_MEDIA_FREQ_CARDIACA'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['TEMPO_ATD_ULT_ATENDIMENTO'])
df_julho_19['TEMPO_ATD_ULT_ATENDIMENTO'].count()
df_julho_19['TEMPO_ATD_ULT_ATENDIMENTO'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['MEDIA_TEMPO_ATD_ATD_ANTERIOR'])
df_julho_19['MEDIA_TEMPO_ATD_ATD_ANTERIOR'].count()
df_julho_19['MEDIA_TEMPO_ATD_ATD_ANTERIOR'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['TEMPO_PERM_ULT_INTERNAMENTO'])
df_julho_19['TEMPO_PERM_ULT_INTERNAMENTO'].count()
df_julho_19['TEMPO_PERM_ULT_INTERNAMENTO'].fillna(-1, inplace = True)

pd.value_counts(df_julho_19['MEDIA_TEMPO_PERM_INT_ANT'])
df_julho_19['MEDIA_TEMPO_PERM_INT_ANT'].count()
df_julho_19['MEDIA_TEMPO_PERM_INT_ANT'].fillna(-1, inplace = True)

#Verificação dos Valores Nulos Ainda Existentes

for col in df_julho_19.columns:
    if df_julho_19[col].isnull().sum() > 0:
        print(col)

#Exclusão das Colunas Que Não Serão Utilizadas
        
df_julho_19_trat = df_julho_19.drop('DATA_ADM', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('DESC_DIA_DA_SEMANA', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('REGISTRO', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('NOME_PACIENTE', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('DESC_SEXO', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('DESC_FAIXA_ETARIA', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('NUM_TRATAMENTO', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('DATA_ADMISSAO', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('DESC_DATA_ADMISSAO', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('DESC_CONVENIO', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('COD_EMPRESA', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('DESC_EMPRESA', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('COD_PLANTONISTA', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('NOME_PLANTONISTA', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('PERGUNTA_TRIAGEM_01', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('PERGUNTA_FICHA_EME_PA_01', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('PERGUNTA_CONSULTA_P_07', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('PERGUNTA_CONSULTA_P_13', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('DESC_DATA_ALTA', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('TEMPO_ATENDIMENTO', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('DESC_STATUS_INT', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('NUM_INT_POSTERIOR', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('TEMPO_PERM_INT_POSTERIOR', axis = 1)

df_julho_19_trat = df_julho_19_trat.drop('CONTADOR', axis = 1)

#Verificação dos Valores Nulos Ainda Existentes

for col in df_julho_19_trat.columns:
    if df_julho_19_trat[col].isnull().sum() > 0:
        print(col)
        
#Estatísticas Descritivas da Base de Dados - Após Tratamento dos Dados
        
df_julho_19_trat.info([''])

#Criação das Bases X_julho_19 e Y_julho_19

X_julho_19 = df_julho_19_trat

Y_julho_19 = df_julho_19_trat['CONT_STATUS_INT']

X_julho_19 = X_julho_19.drop('CONT_STATUS_INT', axis = 1)

#Estatísticas Descritivas da Base de Dados - X_julho_19 - Após Remoção da Coluna Target

X_julho_19.info([''])

#del df_julho_19

#del df_julho_19_trat

#Aplicação do Label Encoder na Base X_julho_19

from sklearn.preprocessing import LabelEncoder

lbl_enc_X_julho_19 = LabelEncoder()

X_julho_19['DESC_ESTACAO_ANO'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['DESC_ESTACAO_ANO'])

X_julho_19['DESC_FINAL_SEMANA'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['DESC_FINAL_SEMANA'])

X_julho_19['DESC_BAIRRO'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['DESC_BAIRRO'])

X_julho_19['COD_CID'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['COD_CID'])

X_julho_19['DESC_GRUPO_SANGUINEO'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['DESC_GRUPO_SANGUINEO'])

X_julho_19['APEL_PLANTONISTA'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['APEL_PLANTONISTA'])

X_julho_19['CID_PREVALENTE'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['CID_PREVALENTE'])

X_julho_19['COD_CLAC_RISCO'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['COD_CLAC_RISCO'])

X_julho_19['PERGUNTA_TRIAGEM_02'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_02'])

X_julho_19['PERGUNTA_TRIAGEM_03'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_03'])

X_julho_19['PERGUNTA_TRIAGEM_04'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_04'])

X_julho_19['PERGUNTA_TRIAGEM_05'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_05'])

X_julho_19['PERGUNTA_TRIAGEM_06'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_06'])

X_julho_19['PERGUNTA_TRIAGEM_07'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_07'])

X_julho_19['PERGUNTA_TRIAGEM_08'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_08'])

X_julho_19['PERGUNTA_TRIAGEM_09'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_09'])

X_julho_19['PERGUNTA_TRIAGEM_10'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_10'])

X_julho_19['PERGUNTA_TRIAGEM_11'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_11'])

X_julho_19['PERGUNTA_TRIAGEM_12'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_12'])

X_julho_19['PERGUNTA_TRIAGEM_13'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_13'])

X_julho_19['PERGUNTA_TRIAGEM_14'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_14'])

X_julho_19['PERGUNTA_TRIAGEM_15'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_15'])

X_julho_19['PERGUNTA_TRIAGEM_16'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_16'])

X_julho_19['PERGUNTA_TRIAGEM_24'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_24'])

X_julho_19['PERGUNTA_TRIAGEM_25'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_25'])

X_julho_19['PERGUNTA_TRIAGEM_26'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_26'])

X_julho_19['PERGUNTA_TRIAGEM_27'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_27'])

X_julho_19['PERGUNTA_TRIAGEM_28'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_28'])

X_julho_19['PERGUNTA_TRIAGEM_29'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_29'])

X_julho_19['PERGUNTA_TRIAGEM_30'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_30'])

X_julho_19['PERGUNTA_TRIAGEM_31'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_31'])

X_julho_19['PERGUNTA_TRIAGEM_32'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_32'])

X_julho_19['PERGUNTA_TRIAGEM_33'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_TRIAGEM_33'])

X_julho_19['PERGUNTA_FICHA_EME_PA_02'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_02'])

X_julho_19['PERGUNTA_FICHA_EME_PA_03'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_03'])

X_julho_19['PERGUNTA_FICHA_EME_PA_04'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_04'])

X_julho_19['PERGUNTA_FICHA_EME_PA_05'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_05'])

X_julho_19['PERGUNTA_FICHA_EME_PA_06'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_06'])

X_julho_19['PERGUNTA_FICHA_EME_PA_07'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_07'])

X_julho_19['PERGUNTA_FICHA_EME_PA_08'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_08'])

X_julho_19['PERGUNTA_FICHA_EME_PA_09'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_09'])

X_julho_19['PERGUNTA_FICHA_EME_PA_10'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_10'])

X_julho_19['PERGUNTA_FICHA_EME_PA_11'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_11'])

X_julho_19['PERGUNTA_FICHA_EME_PA_12'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_12'])

X_julho_19['PERGUNTA_FICHA_EME_PA_13'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_13'])

X_julho_19['PERGUNTA_FICHA_EME_PA_14'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_14'])

X_julho_19['PERGUNTA_FICHA_EME_PA_15'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_15'])

X_julho_19['PERGUNTA_FICHA_EME_PA_16'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_16'])

X_julho_19['PERGUNTA_FICHA_EME_PA_17'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_17'])

X_julho_19['PERGUNTA_FICHA_EME_PA_18'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_18'])

X_julho_19['PERGUNTA_FICHA_EME_PA_19'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_19'])

X_julho_19['PERGUNTA_FICHA_EME_PA_20'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_20'])

X_julho_19['PERGUNTA_FICHA_EME_PA_21'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_21'])

X_julho_19['PERGUNTA_FICHA_EME_PA_22'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_22'])

X_julho_19['PERGUNTA_FICHA_EME_PA_23'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_23'])

X_julho_19['PERGUNTA_FICHA_EME_PA_24'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_24'])

X_julho_19['PERGUNTA_FICHA_EME_PA_25'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_25'])

X_julho_19['PERGUNTA_FICHA_EME_PA_26'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_FICHA_EME_PA_26'])

X_julho_19['COD_CID_ADM'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['COD_CID_ADM'])

X_julho_19['COD_CID_ADM_PRINCIPAL'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['COD_CID_ADM_PRINCIPAL'])

X_julho_19['PERGUNTA_CONSULTA_P_09'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_CONSULTA_P_09'])

X_julho_19['PERGUNTA_CONSULTA_P_10'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_CONSULTA_P_10'])

X_julho_19['PERGUNTA_CONSULTA_P_12'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_CONSULTA_P_12'])

X_julho_19['PERGUNTA_CONSULTA_P_14'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_CONSULTA_P_14'])

X_julho_19['PERGUNTA_CONSULTA_P_15'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_CONSULTA_P_15'])

X_julho_19['PERGUNTA_CONSULTA_P_16'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['PERGUNTA_CONSULTA_P_16'])

X_julho_19['INDICADOR_IAM_CSST'] = lbl_enc_X_julho_19.fit_transform(X_julho_19['INDICADOR_IAM_CSST'])

#Estatísticas Descritivas da Base de Dados - X_julho_19 - Após Aplicação do Label Encoder

X_julho_19.info([''])

#Verificação da Distribuição dos Tipos de Dados das Variáveis - Base X_julho_19

feature_list_X_julho_19 = list(X_julho_19.columns)

feature_list_X_julho_19_integer, feature_list_X_julho_19_float, features_list_X_julho_19_object = groupFeatures(feature_list_X_julho_19, X_julho_19)

print("# of integer feature : ", len(feature_list_X_julho_19_integer))
print("# of float feature : ", len(feature_list_X_julho_19_float))
print("# of object feature : ", len(features_list_X_julho_19_object))

#Transformação da Base Y_julho_19 em Array

import numpy as np

Y_julho_19 = np.array(Y_julho_19)

#Transformação da Base X_julho_19 em Array

X_julho_19_list = list(X_julho_19.columns)

X_julho_19_list_array = np.array(X_julho_19_list)

X_julho_19 = np.array(X_julho_19)

#Escalonamento da Base X_julho_19 - Opção 01 - StandardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_julho_19 = scaler.fit_transform(X_julho_19)

#Escalonamento da Base X_julho_19 - Opção 02 - MinMaX_julho_19Scaler

from sklearn.preprocessing import MinMaX_julho_19Scaler

scaler = MinMaX_julho_19Scaler()

X_julho_19 = scaler.fit_transform(X_julho_19)

#Escalonamento da Base X_julho_19 - Opção 03 - Normalizer

from sklearn.preprocessing import Normalizer

scaler = Normalizer()

X_julho_19 = scaler.fit_transform(X_julho_19)

#Escalonamento da Base X_julho_19 - Opção 04 - RobustScaler

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X_julho_19 = scaler.fit_transform(X_julho_19)

#ETAPA 02: FIT DO MODELO COM X E Y

import time
start = time.time()

classifier_lgbm.fit(X, Y)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 6.61 min

#ETAPA 03: PREDIÇÃO

Y_julho_19_pred_lgbm = classifier_lgbm.predict(X_julho_19)

#Análise de Métricas

from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm = accuracy_score(Y_julho_19, Y_julho_19_pred_lgbm)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm))

#Balanced Accuracy

mtrc_balanced_accuracy_score_lgbm = balanced_accuracy_score(Y_julho_19, Y_julho_19_pred_lgbm)
print('Balanced Accuracy Score : ' + str(mtrc_balanced_accuracy_score_lgbm))

#Precision Score
#Número de vezes que uma classe foi predita corretamente dividida pelo número de vezes que a classe foi predita.

mtrc_precision_score_lgbm = precision_score(Y_julho_19, Y_julho_19_pred_lgbm)
print('Precision Score : ' + str(mtrc_precision_score_lgbm))

#Recall Score
#Número de vezes que uma classe foi predita corretamente (TP) dividido pelo número de vezes que a classe aparece no dado de teste (TP + FN).

mtrc_recall_score_lgbm = recall_score(Y_julho_19, Y_julho_19_pred_lgbm)
print('Recall Score : ' + str(mtrc_recall_score_lgbm))

#F1 Score
#Média harmônica entre precisão e revocação. Com essa informação podemos dizer a performance do classificador com um indicador apenas.

mtrc_f1_score_lgbm = f1_score(Y_julho_19, Y_julho_19_pred_lgbm)
print('F1 Score : ' + str(mtrc_f1_score_lgbm))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm = matthews_corrcoef(Y_julho_19, Y_julho_19_pred_lgbm)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm = cohen_kappa_score(Y_julho_19, Y_julho_19_pred_lgbm)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm))

#Area Under the Receiver Operating Characteristic Curve (ROC AUC)

from sklearn.metrics import roc_auc_score

mtrc_roc_auc_score_lgbm = roc_auc_score(Y_julho_19, Y_julho_19_pred_lgbm)
print('Area Under the Receiver Operating Characteristic Curve (ROC AUC) : ' + str(mtrc_roc_auc_score_lgbm))

#Area Under the Precision-Recall Curve

from sklearn.metrics import average_precision_score

mtrc_average_precision_score_lgbm = average_precision_score(Y_julho_19, Y_julho_19_pred_lgbm)
print('Area Under the Precision-Recall Curve : ' + str(mtrc_average_precision_score_lgbm))

#Confusion Matrix

from sklearn.metrics import confusion_matrix

mtrc_confusion_matrix_lgbm = confusion_matrix(Y_julho_19, Y_julho_19_pred_lgbm)

print('Confusion Matrix : \n' + str(mtrc_confusion_matrix_lgbm))

print(pd.crosstab(Y_julho_19, Y_julho_19_pred_lgbm, rownames = ['Real'], colnames = ['Predito'], margins = True))

#Confusion Matrix - True Positives
mtrc_true_positive_lgbm = mtrc_confusion_matrix_lgbm[1, 1]

#Confusion Matrix - True Negatives
mtrc_true_negative_lgbm = mtrc_confusion_matrix_lgbm[0, 0]

#Confusion Matrix - False Positives
mtrc_false_positive_lgbm = mtrc_confusion_matrix_lgbm[0, 1]

#Confusion Matrix - False Negatives
mtrc_false_negative_lgbm = mtrc_confusion_matrix_lgbm[1, 0]

#cm_true_negative_lgbm, cm_false_positive_lgbm, cm_false_negative_lgbm, cm_true_positive_lgbm = confusion_matrix(Y_julho_19, Y_julho_19_pred_lgbm).ravel()

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
Accuracy Score : 0.9307048984468339
Balanced Accuracy Score : 0.773747073293619
Precision Score : 0.8181818181818182
Recall Score : 0.5652173913043478
F1 Score : 0.6685714285714286
Matthews Correlation Coefficient : 0.644798614985018
Cohen’s Kappa Score : 0.6313181367690783
Area Under the Receiver Operating Characteristic Curve (ROC AUC) : 0.7737470732936189
Area Under the Precision-Recall Curve : 0.5162140337455905