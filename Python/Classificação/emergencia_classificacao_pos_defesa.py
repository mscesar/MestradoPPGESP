#Setup da Pasta de Trabalho
import os
os.chdir('D:/Dropbox/MauricioPessoal/Pós-Graduação/PPGESP/Projeto Final/Estudo Python/Bases Cárdio Pulmonar')

import warnings
warnings.filterwarnings("ignore")

#ETAPA 01: TRATAMENTO DOS DADOS

#Importação do Pandas

import pandas as pd

#Leitura da Base de Dados

#Criação da Base df com Todas as Informações Disponíveis

import time

start = time.time()
df = pd.read_excel('TB_IND_ASSIST_EME_0115_0319_V3.xlsx')
end = time.time()

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 11.32 min
#Tempo de Execução: 20.25 min
#Tempo de Execução: 12.79 min
#Tempo de Execução: 46.79 min
#Tempo de Execução: 11.98 min
#Tempo de Execução: 14.53 min
Tempo de Execução: 9.73 min

#Criação da Base df_chegada A Fim de Mapear Apenas as Informações Disponíveis na Chegada do Paciente

df_chegada = df.copy()

#Criação da Base df_triagem A Fim de Mapear Apenas as Informações Disponíveis Após a Realização da Triagem

df_triagem = df.copy()

#Criação da Base df_plantonista A Fim de Mapear Apenas as Informações Disponíveis Após a Realização da Consulta com o Plantonista

df_plantonista = df.copy()

#Verificação da Quantidade de Linhas e Colunas da Base de Dados

df.shape
#(84303, 652)

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
#Tempo de Execução: 1.41 min

corr_matrix_pearson_order = corr_matrix_pearson['CONT_STATUS_INT'].sort_values(ascending = False)

k = 20

corr_matrix_pearson.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT']

corr_matrix_pearson.nsmallest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT']

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
corr_matrix_pearson_excel = pd.ExcelWriter('V4/corr_matrix_pearson_excel.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
corr_matrix_pearson_order.to_excel(corr_matrix_pearson_excel, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
corr_matrix_pearson_excel.save()

#Gráfico da Matriz de Pearson com 20+
cols = corr_matrix_pearson.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1.2)
plt.figure(figsize = (10, 10))
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()

#Gráfico da Matriz de Pearson com 20-
cols = corr_matrix_pearson.nlargest(1, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
cols_1 = corr_matrix_pearson.nsmallest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index 
cols = cols.append(cols_1)
cm = np.corrcoef(df[cols].values.T)
sns.set(font_scale = 1.2)
plt.figure(figsize = (10, 10))
hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()

#DataFrame com 15+
k = 15

cols = corr_matrix_pearson.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index

corr_matrix_pearson_new = df[corr_matrix_pearson['CONT_STATUS_INT'].nlargest(k).index].corr('pearson')
#cols = corr_matrix_pearson_new.index

sns.set(font_scale = 1.2)
plt.figure(figsize = (12, 12))
hm = sns.heatmap(corr_matrix_pearson_new, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 14}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()

#DataFrame com 15-
k = 14

corr_matrix_pearson_order_2 = pd.DataFrame(corr_matrix_pearson_order)
cols = corr_matrix_pearson.nlargest(1, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
cols_1 = corr_matrix_pearson_order_2.nsmallest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].sort_values(ascending = False).index 
cols = cols.append(cols_1)

corr_matrix_pearson_new_menos = df[cols].corr('pearson')

sns.set(font_scale = 1.2)
plt.figure(figsize = (12, 12))
hm = sns.heatmap(corr_matrix_pearson_new_menos, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 14}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()

#Análise de Correlação das Variáveis - Spearman

import numpy as np

import time

import seaborn as sns

import matplotlib.pyplot as plt

start = time.time()

corr_matrix_spearman = df.corr('spearman')

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 36.52 min

corr_matrix_spearman_order = corr_matrix_spearman['CONT_STATUS_INT'].sort_values(ascending = False)

k = 20

corr_matrix_spearman.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT']

corr_matrix_spearman.nsmallest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT']

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
corr_matrix_spearman_excel = pd.ExcelWriter('V4/corr_matrix_spearman_excel.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
corr_matrix_spearman_order.to_excel(corr_matrix_spearman_excel, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
corr_matrix_spearman_excel.save()

#Gráfico da Matriz de Spearman com 20+
#cols = corr_matrix_spearman.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
#cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale = 1.2)
#plt.figure(figsize = (10, 10))
#hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
#plt.show()

#Gráfico da Matriz de Spearman com 20-
#cols = corr_matrix_spearman.nlargest(1, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
#cols_1 = corr_matrix_spearman.nsmallest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index 
#cols = cols.append(cols_1)
#cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale = 1.2)
#plt.figure(figsize = (10, 10))
#hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
#plt.show()

#DataFrame com 15+
k = 15

cols = corr_matrix_spearman.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index

corr_matrix_spearman_new = df[corr_matrix_spearman['CONT_STATUS_INT'].nlargest(k).index].corr('spearman')
#cols = corr_matrix_spearman_new.index

sns.set(font_scale = 1.2)
plt.figure(figsize = (12, 12))
hm = sns.heatmap(corr_matrix_spearman_new, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 14}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()

#DataFrame com 15-
k = 14

corr_matrix_spearman_order_2 = pd.DataFrame(corr_matrix_spearman_order)
cols = corr_matrix_spearman.nlargest(1, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
cols_1 = corr_matrix_spearman_order_2.nsmallest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].sort_values(ascending = False).index 
cols = cols.append(cols_1)

corr_matrix_spearman_new_menos = df[cols].corr('spearman')

sns.set(font_scale = 1.2)
plt.figure(figsize = (12, 12))
hm = sns.heatmap(corr_matrix_spearman_new_menos, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 14}, yticklabels = cols.values, xticklabels = cols.values)
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
#Tempo de Execução: 29.72 min

corr_matrix_kendall_order = corr_matrix_kendall['CONT_STATUS_INT'].sort_values(ascending = False)

k = 20

corr_matrix_kendall.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT']

corr_matrix_kendall.nsmallest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT']

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
corr_matrix_kendall_excel = pd.ExcelWriter('V4/corr_matrix_kendall_excel.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
corr_matrix_kendall_order.to_excel(corr_matrix_kendall_excel, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
corr_matrix_kendall_excel.save()

#Gráfico da Matriz de Kendall com 20+
#cols = corr_matrix_kendall.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
#cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale = 1.2)
#plt.figure(figsize = (10, 10))
#hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
#plt.show()

#Gráfico da Matriz de Kendall com 20-
#cols = corr_matrix_kendall.nlargest(1, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
#cols_1 = corr_matrix_kendall.nsmallest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index 
#cols = cols.append(cols_1)
#cm = np.corrcoef(df[cols].values.T)
#sns.set(font_scale = 1.2)
#plt.figure(figsize = (10, 10))
#hm = sns.heatmap(cm, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 10}, yticklabels = cols.values, xticklabels = cols.values)
#plt.show()

#DataFrame com 15+
k = 15

cols = corr_matrix_kendall.nlargest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index

corr_matrix_kendall_new = df[corr_matrix_kendall['CONT_STATUS_INT'].nlargest(k).index].corr('kendall')
#cols = corr_matrix_kendall_new.index

sns.set(font_scale = 1.2)
plt.figure(figsize = (12, 12))
hm = sns.heatmap(corr_matrix_kendall_new, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 14}, yticklabels = cols.values, xticklabels = cols.values)
plt.show()

#DataFrame com 15-
k = 14

corr_matrix_kendall_order_2 = pd.DataFrame(corr_matrix_kendall_order)
cols = corr_matrix_kendall.nlargest(1, 'CONT_STATUS_INT')['CONT_STATUS_INT'].index
cols_1 = corr_matrix_kendall_order_2.nsmallest(k, 'CONT_STATUS_INT')['CONT_STATUS_INT'].sort_values(ascending = False).index 
cols = cols.append(cols_1)

corr_matrix_kendall_new_menos = df[cols].corr('kendall')

sns.set(font_scale = 1.2)
plt.figure(figsize = (12, 12))
hm = sns.heatmap(corr_matrix_kendall_new_menos, cbar = True, annot = True, square = True, fmt = '.2f', annot_kws = {'size': 14}, yticklabels = cols.values, xticklabels = cols.values)
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

corr_matrix_pearson_new_menos

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix_pearson_new_menos, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix_pearson_new_menos, mask = mask, cmap = cmap, vmax = .3, center = 0,
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

corr_matrix_spearman_new_menos

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix_spearman_new_menos, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix_spearman_new_menos, mask = mask, cmap = cmap, vmax = .3, center = 0,
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

corr_matrix_kendall_new_menos

# Generate a mask for the upper triangle
mask = np.zeros_like(corr_matrix_kendall_new_menos, dtype = np.bool)
mask[np.triu_indices_from(mask)] = True
# Set up the matplotlib figure
f, ax = plt.subplots(figsize = (11, 9))
# Generate a custom diverging colormap
cmap = sns.diverging_palette(220, 10, as_cmap = True)
# Draw the heatmap with the mask and correct aspect ratio
sns.heatmap(corr_matrix_kendall_new_menos, mask = mask, cmap = cmap, vmax = .3, center = 0,
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

#Escalonamento da Base X - Opção 02 - MinMaxScaler

from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

X = scaler.fit_transform(X)

#Escalonamento da Base X - Opção 03 - Normalizer

from sklearn.preprocessing import Normalizer

scaler = Normalizer()

X = scaler.fit_transform(X)

#Escalonamento da Base X - Opção 04 - RobustScaler

from sklearn.preprocessing import RobustScaler

scaler = RobustScaler()

X = scaler.fit_transform(X)

#ETAPA 02: DIVISÃO DA BASE DE DADOS EM TESTE E TREINAMENTO

#Criação da Base de Teste e Treinamento - Sem Balanceamento em Y

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

print('Training X Data Dimensions:', X_train.shape)
print('Training Y Data Dimensions:', Y_train.shape)
print('Testing X Data Dimensions:', X_test.shape)
print('Testing Y Data Dimensions:', Y_test.shape)

print('Labels counts in Y:', np.bincount(Y))
#
print('Labels counts in Y_train:', np.bincount(Y_train))
#
print('Labels counts in Y_test:', np.bincount(Y_test))
#

#Criação da Base de Teste e Treinamento - Com Balanceamento em Y

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42, stratify = Y)

print('Training X Data Dimensions:', X_train.shape)
print('Training Y Data Dimensions:', Y_train.shape)
print('Testing X Data Dimensions:', X_test.shape)
print('Testing Y Data Dimensions:', Y_test.shape)

#Training X Data Dimensions: (59012, 627)
#Training Y Data Dimensions: (59012,)
#Testing X Data Dimensions: (25291, 627)
#Testing Y Data Dimensions: (25291,)

print('Labels counts in Y:', np.bincount(Y))
#Labels counts in Y: [74584  9719]
print('Labels counts in Y_train:', np.bincount(Y_train))
#Labels counts in Y_train: [52209  6803]
print('Labels counts in Y_test:', np.bincount(Y_test))
#Labels counts in Y_test: [22375  2916]

#Cópia da Base X_train

X_train_pca = X_train.copy()

X_train_spca = X_train.copy()

X_train_rpca = X_train.copy()

X_train_ipca = X_train.copy()

X_train_lda = X_train.copy()

X_train_kpca = X_train.copy()

#Cópia da Base X_test

X_test_pca = X_test.copy()

X_test_spca = X_test.copy()

X_test_rpca = X_test.copy()

X_test_ipca = X_test.copy()

X_test_lda = X_test.copy()

X_test_kpca = X_test.copy()  

#ETAPA 03: TESTE DE VÁRIOS MODELOS

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.linear_model import Perceptron
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.naive_bayes import BernoulliNB
from sklearn.ensemble import ExtraTreesClassifier
from rgf.sklearn import RGFClassifier
from rgf.sklearn import FastRGFClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('PRT', Perceptron()))
models.append(('MLP', MLPClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('BNB', BernoulliNB()))
models.append(('ETC', ExtraTreesClassifier()))
models.append(('RGF', RGFClassifier()))
models.append(('FastRGF', FastRGFClassifier()))
models.append(('RF', RandomForestClassifier()))
models.append(('AB', AdaBoostClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('XGB', XGBClassifier()))
models.append(('LGBM', LGBMClassifier()))
models.append(('CB', CatBoostClassifier()))

#models = []
#models.append(('LR', LogisticRegression()))
#models.append(('PRT', Perceptron()))
#models.append(('MLP', MLPClassifier()))
#models.append(('GNB', GaussianNB()))
#models.append(('BNB', BernoulliNB()))
#models.append(('ETC', ExtraTreesClassifier()))
#models.append(('RGF', RGFClassifier()))
#models.append(('FastRGF', FastRGFClassifier()))
#models.append(('RF', RandomForestClassifier()))
#models.append(('AB', AdaBoostClassifier()))
#models.append(('GB', GradientBoostingClassifier()))
#models.append(('XGB', XGBClassifier()))
#models.append(('LGBM', LGBMClassifier()))
#models.append(('CB', CatBoostClassifier()))

#Treino e Teste dos Modelos

from sklearn.metrics import matthews_corrcoef

import time

names = []
scores = []

start = time.time()
for name, model in models:
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    scores.append(matthews_corrcoef(Y_test, Y_pred))
    names.append(name)
    
end = time.time()    
tr_split = pd.DataFrame({'Name': names, 'Score': scores})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(tr_split)

#Plotagem das Acurácias

import seaborn as sns
import matplotlib.pyplot as plt

axis = sns.barplot(x = 'Name', y = 'Score', data = tr_split)
axis.set(xlabel = 'Classifier', ylabel = 'Matthews Correlation Coefficient (MCC)')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 

plt.figure(figsize = (14, 7))    
plt.show()

#Treino e Teste dos Modelos com CrossValScore + 10 KFold(shuffle = False) - X_train, Y_train

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time

names = []
scores = []
std = []

start = time.time()
for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 42, shuffle = False) 
    result = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = make_scorer(matthews_corrcoef)) 
    names.append(name)
    scores.append(result.mean())
    std.append(result.std())
end = time.time()
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores, 'Standard Deviation': std })

print("Tempo de Execução: {:.2f} min".format((end - start)/60)) 
print(kf_cross_val)

#Tempo de Execução: 465.35 min
#Tempo de Execução: 884.21 min

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_classifier = pd.ExcelWriter('V3/cross_val_score_classifier_X_train_Y_train_kfold.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val.to_excel(cross_val_score_classifier, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_classifier.save()

#Treino e Teste dos Modelos com CrossValScore + 10 KFold(shuffle = True) - X_train, Y_train

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time

names = []
scores = []
std = []

start = time.time()
for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 42, shuffle = True,) 
    result = cross_val_score(model, X_train, Y_train, cv = kfold, scoring = make_scorer(matthews_corrcoef)) 
    names.append(name)
    scores.append(result.mean())
    std.append(result.std())
end = time.time()
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores, 'Standard Deviation': std })

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_val)

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_classifier = pd.ExcelWriter('cross_val_score_classifier_X_train_Y_train_kfold_shuffle.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val.to_excel(cross_val_score_classifier, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_classifier.save()

#Treino e Teste dos Modelos com CrossValScore + 10 KFold(shuffle = False) - X, Y

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time

names = []
scores = []
std = []

start = time.time()
for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 42, shuffle = False) 
    result = cross_val_score(model, X, Y, cv = kfold, scoring = make_scorer(matthews_corrcoef)) 
    names.append(name)
    scores.append(result.mean())
    std.append(result.std())
    
end = time.time()
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores, 'Standard Deviation': std })

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_val)

#Tempo de Execução: 1165.48 min

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_classifier = pd.ExcelWriter('V3/cross_val_score_classifier_X_Y_kfold.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val.to_excel(cross_val_score_classifier, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_classifier.save()

#Treino e Teste dos Modelos com CrossValScore + 10 KFold(shuffle = True) - X, Y

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time

names = []
scores = []
std = []

start = time.time()
for name, model in models:
    kfold = KFold(n_splits = 10, random_state = 42, shuffle = True) 
    result = cross_val_score(model, X, Y, cv = kfold, scoring = make_scorer(matthews_corrcoef)) 
    names.append(name)
    scores.append(result.mean())
    std.append(result.std())
end = time.time() 
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores, 'Standard Deviation': std })

print("Tempo de Execução: {:.2f} min".format((end - start)/60)) 
print(kf_cross_val)

#Tempo de Execução: 593.71 min

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_classifier = pd.ExcelWriter('V3/cross_val_score_classifier_X_Y_kfold_shuffle.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val.to_excel(cross_val_score_classifier, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_classifier.save()

#Plotagem das Acurácias com CrossValScore + 10 KFold

import seaborn as sns
import matplotlib.pyplot as plt

axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel = 'Classifier', ylabel = 'Matthews Correlation Coefficient (MCC)')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha = "center") 
    
plt.figure(figsize = (14, 7))    
plt.show()

#Treino e Teste dos Modelos com CrossValScore + ShuffleSplit(n_splits = 10) - X_train, Y_train

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time

names = []
scores = []
std = []

start = time.time()
for name, model in models:
    shuffle = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state = 42)
    result = cross_val_score(model, X_train, Y_train, cv = shuffle, scoring = make_scorer(matthews_corrcoef)) 
    names.append(name)
    scores.append(result.mean())
    std.append(result.std())
end = time.time()
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores, 'Standard Deviation': std })

print("Tempo de Execução: {:.2f} min".format((end - start)/60)) 
print(kf_cross_val)

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_classifier = pd.ExcelWriter('cross_val_score_classifier_shuffle_X_train_Y_train.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val.to_excel(cross_val_score_classifier, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_classifier.save()

#Treino e Teste dos Modelos com CrossValScore + ShuffleSplit(n_splits = 10) - X, Y

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time

names = []
scores = []
std = []

start = time.time()
for name, model in models:
    shuffle = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state = 42)
    result = cross_val_score(model, X, Y, cv = shuffle, scoring = make_scorer(matthews_corrcoef)) 
    names.append(name)
    scores.append(result.mean())
    std.append(result.std())
end = time.time()     
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores, 'Standard Deviation': std })

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_val)

#Tempo de Execução: 606.47 min

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_classifier = pd.ExcelWriter('V3/cross_val_score_classifier_shuffle_X_Y.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val.to_excel(cross_val_score_classifier, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_classifier.save()

#Plotagem das Acurácias com CrossValScore + ShuffleSplit(n_splits = 10)

import seaborn as sns
import matplotlib.pyplot as plt

axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel = 'Classifier', ylabel = 'Matthews Correlation Coefficient (MCC)')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha = "center") 
    
plt.figure(figsize = (14, 7))    
plt.show()

#Treino e Teste dos Modelos com CrossValScore + 10 StratifiedKFold(shuffle = False) - X, Y

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time

names = []
scores = []
std = []

start = time.time()
for name, model in models:
    skfold = StratifiedKFold(n_splits = 10, shuffle = False, random_state = 42)
    result = cross_val_score(model, X, Y, cv = skfold, scoring = make_scorer(matthews_corrcoef)) 
    names.append(name)
    scores.append(result.mean())
    std.append(result.std())
    
end = time.time()
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores, 'Standard Deviation': std })

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_val)

#Tempo de Execução: 604.83 min

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_classifier = pd.ExcelWriter('V3/cross_val_score_classifier_X_Y_stratifiedkfold.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val.to_excel(cross_val_score_classifier, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_classifier.save()

#Treino e Teste dos Modelos com CrossValScore + 10 StratifiedKFold(shuffle = True) - X, Y

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time

names = []
scores = []
std = []

start = time.time()
for name, model in models:
    skfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
    result = cross_val_score(model, X, Y, cv = skfold, scoring = make_scorer(matthews_corrcoef)) 
    names.append(name)
    scores.append(result.mean())
    std.append(result.std())
end = time.time() 
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores, 'Standard Deviation': std })

print("Tempo de Execução: {:.2f} min".format((end - start)/60)) 
print(kf_cross_val)

#Tempo de Execução: 607.72 min

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_classifier = pd.ExcelWriter('V3/cross_val_score_classifier_X_Y_stratifiedkfold_shuffle.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val.to_excel(cross_val_score_classifier, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_classifier.save()

#Treino e Teste dos Modelos com CrossValidate + 10 KFold(shuffle = False) + Várias Métricas - X_train, Y_train

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import cohen_kappa_score

import time

metricas_cross_validate = { 'accuracy': 'accuracy',
                            'balanced_accuracy': 'balanced_accuracy',
                            'average_precision': 'average_precision',
                            'f1': 'f1',
                            'precision': 'precision',
                            'recall': 'recall',
                            'roc_auc': 'roc_auc',
                            'ms_matthews_corrcoef': make_scorer(matthews_corrcoef),
                            'ms_cohen_kappa_score': make_scorer(cohen_kappa_score) }

lst_names = []
lst_fit_time = []
lst_score_time = []
lst_test_accuracy = []
lst_test_balanced_accuracy = []
lst_test_average_precision = []
lst_test_f1 = []
lst_test_precision = []
lst_test_recall = []
lst_test_roc_auc = []
lst_test_matthews_corrcoef = []
lst_test_cohen_kappa_score = []

start = time.time()
for var_name, model in models:
    kfold = KFold(n_splits = 10, random_state = 42) 
    result = cross_validate( estimator = model, 
                             X = X_train, 
                             y = Y_train, 
                             cv = kfold, 
                             scoring = metricas_cross_validate, 
                             return_train_score = False )    
    var_fit_time   = result['fit_time'].mean()
    var_score_time = result['score_time'].mean()
    var_test_accuracy = result['test_accuracy'].mean()
    var_test_balanced_accuracy = result['test_balanced_accuracy'].mean()
    var_test_average_precision = result['test_average_precision'].mean()
    var_test_f1 = result['test_f1'].mean()
    var_test_precision = result['test_precision'].mean()
    var_test_recall = result['test_recall'].mean()
    var_test_roc_auc = result['test_roc_auc'].mean()
    var_test_matthews_corrcoef = result['test_ms_matthews_corrcoef'].mean()
    var_test_cohen_kappa_score = result['test_ms_cohen_kappa_score'].mean()
    lst_names.append(var_name)
    lst_fit_time.append(var_fit_time)
    lst_score_time.append(var_score_time)
    lst_test_accuracy.append(var_test_accuracy)
    lst_test_balanced_accuracy.append(var_test_balanced_accuracy)
    lst_test_average_precision.append(var_test_average_precision)
    lst_test_f1.append(var_test_f1)
    lst_test_precision.append(var_test_precision)
    lst_test_recall.append(var_test_recall)
    lst_test_roc_auc.append(var_test_roc_auc)
    lst_test_matthews_corrcoef.append(var_test_matthews_corrcoef)
    lst_test_cohen_kappa_score.append(var_test_cohen_kappa_score)
end = time.time()    
kf_cross_validate = pd.DataFrame({'Name': lst_names, 
                                  'Fit Time': lst_fit_time, 
                                  'Score Time': lst_score_time, 
                                  'Test Accuracy': lst_test_accuracy, 
                                  'Test Balanced Accuracy': lst_test_balanced_accuracy, 
                                  'Test Average Precision': lst_test_average_precision, 
                                  'Test F1': lst_test_f1, 
                                  'Test Precision': lst_test_precision, 
                                  'Test Recall': lst_test_recall,
                                  'Test ROC AUC': lst_test_roc_auc, 
                                  'Test Matthews Correlation Coefficient': lst_test_matthews_corrcoef,
                                  'Test Cohen’s Kappa Score': lst_test_cohen_kappa_score})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_validate)

# Create a Pandas Excel writer using XlsxWriter as the engine.
cross_validate_classifier = pd.ExcelWriter('cross_validate_classifier_X_train_Y_train_kfold.xlsx', engine = 'xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
kf_cross_validate.to_excel(cross_validate_classifier, sheet_name = 'Dados')
# Close the Pandas Excel writer and output the Excel file.
cross_validate_classifier.save()

#Treino e Teste dos Modelos com CrossValidate + 10 KFold(shuffle = False) + Várias Métricas - X, Y 

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import cohen_kappa_score

import time

metricas_cross_validate = { 'accuracy': 'accuracy',
                            'balanced_accuracy': 'balanced_accuracy',
                            'average_precision': 'average_precision',
                            'f1': 'f1',
                            'precision': 'precision',
                            'recall': 'recall',
                            'roc_auc': 'roc_auc',
                            'ms_matthews_corrcoef': make_scorer(matthews_corrcoef),
                            'ms_cohen_kappa_score': make_scorer(cohen_kappa_score) }

lst_names = []
lst_fit_time = []
lst_score_time = []
lst_test_accuracy = []
lst_test_balanced_accuracy = []
lst_test_average_precision = []
lst_test_f1 = []
lst_test_precision = []
lst_test_recall = []
lst_test_roc_auc = []
lst_test_matthews_corrcoef = []
lst_test_cohen_kappa_score = []

start = time.time()
for var_name, model in models:
    kfold = KFold(n_splits = 10, random_state = 42) 
    result = cross_validate( estimator = model, 
                             X = X, 
                             y = Y, 
                             cv = kfold, 
                             scoring = metricas_cross_validate, 
                             return_train_score = False )    
    var_fit_time   = result['fit_time'].mean()
    var_score_time = result['score_time'].mean()
    var_test_accuracy = result['test_accuracy'].mean()
    var_test_balanced_accuracy = result['test_balanced_accuracy'].mean()
    var_test_average_precision = result['test_average_precision'].mean()
    var_test_f1 = result['test_f1'].mean()
    var_test_precision = result['test_precision'].mean()
    var_test_recall = result['test_recall'].mean()
    var_test_roc_auc = result['test_roc_auc'].mean()
    var_test_matthews_corrcoef = result['test_ms_matthews_corrcoef'].mean()
    var_test_cohen_kappa_score = result['test_ms_cohen_kappa_score'].mean()
    lst_names.append(var_name)
    lst_fit_time.append(var_fit_time)
    lst_score_time.append(var_score_time)
    lst_test_accuracy.append(var_test_accuracy)
    lst_test_balanced_accuracy.append(var_test_balanced_accuracy)
    lst_test_average_precision.append(var_test_average_precision)
    lst_test_f1.append(var_test_f1)
    lst_test_precision.append(var_test_precision)
    lst_test_recall.append(var_test_recall)
    lst_test_roc_auc.append(var_test_roc_auc)
    lst_test_matthews_corrcoef.append(var_test_matthews_corrcoef)
    lst_test_cohen_kappa_score.append(var_test_cohen_kappa_score)
end = time.time()    
kf_cross_validate = pd.DataFrame({'Name': lst_names, 
                                  'Fit Time': lst_fit_time, 
                                  'Score Time': lst_score_time, 
                                  'Test Accuracy': lst_test_accuracy, 
                                  'Test Balanced Accuracy': lst_test_balanced_accuracy, 
                                  'Test Average Precision': lst_test_average_precision, 
                                  'Test F1': lst_test_f1, 
                                  'Test Precision': lst_test_precision, 
                                  'Test Recall': lst_test_recall,
                                  'Test ROC AUC': lst_test_roc_auc, 
                                  'Test Matthews Correlation Coefficient': lst_test_matthews_corrcoef,
                                  'Test Cohen’s Kappa Score': lst_test_cohen_kappa_score})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_validate)
#Tempo de Execução: 2870.81 min

# Create a Pandas Excel writer using XlsxWriter as the engine.
cross_validate_classifier = pd.ExcelWriter('V3/cross_validate_classifier_X_Y_kfold.xlsx', engine = 'xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
kf_cross_validate.to_excel(cross_validate_classifier, sheet_name = 'Dados')
# Close the Pandas Excel writer and output the Excel file.
cross_validate_classifier.save()

#Treino e Teste dos Modelos com CrossValidate + 10 KFold(shuffle = True) + Várias Métricas - X, Y 

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import cohen_kappa_score

import time

metricas_cross_validate = { 'accuracy': 'accuracy',
                            'balanced_accuracy': 'balanced_accuracy',
                            'average_precision': 'average_precision',
                            'f1': 'f1',
                            'precision': 'precision',
                            'recall': 'recall',
                            'roc_auc': 'roc_auc',
                            'ms_matthews_corrcoef': make_scorer(matthews_corrcoef),
                            'ms_cohen_kappa_score': make_scorer(cohen_kappa_score) }

lst_names = []
lst_fit_time = []
lst_score_time = []
lst_test_accuracy = []
lst_test_balanced_accuracy = []
lst_test_average_precision = []
lst_test_f1 = []
lst_test_precision = []
lst_test_recall = []
lst_test_roc_auc = []
lst_test_matthews_corrcoef = []
lst_test_cohen_kappa_score = []

start = time.time()
for var_name, model in models:
    kfold = KFold(n_splits = 10, random_state = 42, shuffle = True) 
    result = cross_validate( estimator = model, 
                             X = X, 
                             y = Y, 
                             cv = kfold, 
                             scoring = metricas_cross_validate, 
                             return_train_score = False )    
    var_fit_time   = result['fit_time'].mean()
    var_score_time = result['score_time'].mean()
    var_test_accuracy = result['test_accuracy'].mean()
    var_test_balanced_accuracy = result['test_balanced_accuracy'].mean()
    var_test_average_precision = result['test_average_precision'].mean()
    var_test_f1 = result['test_f1'].mean()
    var_test_precision = result['test_precision'].mean()
    var_test_recall = result['test_recall'].mean()
    var_test_roc_auc = result['test_roc_auc'].mean()
    var_test_matthews_corrcoef = result['test_ms_matthews_corrcoef'].mean()
    var_test_cohen_kappa_score = result['test_ms_cohen_kappa_score'].mean()
    lst_names.append(var_name)
    lst_fit_time.append(var_fit_time)
    lst_score_time.append(var_score_time)
    lst_test_accuracy.append(var_test_accuracy)
    lst_test_balanced_accuracy.append(var_test_balanced_accuracy)
    lst_test_average_precision.append(var_test_average_precision)
    lst_test_f1.append(var_test_f1)
    lst_test_precision.append(var_test_precision)
    lst_test_recall.append(var_test_recall)
    lst_test_roc_auc.append(var_test_roc_auc)
    lst_test_matthews_corrcoef.append(var_test_matthews_corrcoef)
    lst_test_cohen_kappa_score.append(var_test_cohen_kappa_score)
end = time.time()    
kf_cross_validate = pd.DataFrame({'Name': lst_names, 
                                  'Fit Time': lst_fit_time, 
                                  'Score Time': lst_score_time, 
                                  'Test Accuracy': lst_test_accuracy, 
                                  'Test Balanced Accuracy': lst_test_balanced_accuracy, 
                                  'Test Average Precision': lst_test_average_precision, 
                                  'Test F1': lst_test_f1, 
                                  'Test Precision': lst_test_precision, 
                                  'Test Recall': lst_test_recall,
                                  'Test ROC AUC': lst_test_roc_auc, 
                                  'Test Matthews Correlation Coefficient': lst_test_matthews_corrcoef,
                                  'Test Cohen’s Kappa Score': lst_test_cohen_kappa_score})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_validate)
#Tempo de Execução: 3084.77 min

# Create a Pandas Excel writer using XlsxWriter as the engine.
cross_validate_classifier = pd.ExcelWriter('V3/cross_validate_classifier_X_Y_kfold_shuffle.xlsx', engine = 'xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
kf_cross_validate.to_excel(cross_validate_classifier, sheet_name = 'Dados')
# Close the Pandas Excel writer and output the Excel file.
cross_validate_classifier.save()

#Treino e Teste dos Modelos com CrossValidate + ShuffleSplit(n_splits = 10) + Várias Métricas - X, Y 

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import cohen_kappa_score

import time

metricas_cross_validate = { 'accuracy': 'accuracy',
                            'balanced_accuracy': 'balanced_accuracy',
                            'average_precision': 'average_precision',
                            'f1': 'f1',
                            'precision': 'precision',
                            'recall': 'recall',
                            'roc_auc': 'roc_auc',
                            'ms_matthews_corrcoef': make_scorer(matthews_corrcoef),
                            'ms_cohen_kappa_score': make_scorer(cohen_kappa_score) }

lst_names = []
lst_fit_time = []
lst_score_time = []
lst_test_accuracy = []
lst_test_balanced_accuracy = []
lst_test_average_precision = []
lst_test_f1 = []
lst_test_precision = []
lst_test_recall = []
lst_test_roc_auc = []
lst_test_matthews_corrcoef = []
lst_test_cohen_kappa_score = []

start = time.time()
for var_name, model in models:
    shuffle = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state = 42) 
    result = cross_validate( estimator = model, 
                             X = X, 
                             y = Y, 
                             cv = shuffle, 
                             scoring = metricas_cross_validate, 
                             return_train_score = False )    
    var_fit_time   = result['fit_time'].mean()
    var_score_time = result['score_time'].mean()
    var_test_accuracy = result['test_accuracy'].mean()
    var_test_balanced_accuracy = result['test_balanced_accuracy'].mean()
    var_test_average_precision = result['test_average_precision'].mean()
    var_test_f1 = result['test_f1'].mean()
    var_test_precision = result['test_precision'].mean()
    var_test_recall = result['test_recall'].mean()
    var_test_roc_auc = result['test_roc_auc'].mean()
    var_test_matthews_corrcoef = result['test_ms_matthews_corrcoef'].mean()
    var_test_cohen_kappa_score = result['test_ms_cohen_kappa_score'].mean()
    lst_names.append(var_name)
    lst_fit_time.append(var_fit_time)
    lst_score_time.append(var_score_time)
    lst_test_accuracy.append(var_test_accuracy)
    lst_test_balanced_accuracy.append(var_test_balanced_accuracy)
    lst_test_average_precision.append(var_test_average_precision)
    lst_test_f1.append(var_test_f1)
    lst_test_precision.append(var_test_precision)
    lst_test_recall.append(var_test_recall)
    lst_test_roc_auc.append(var_test_roc_auc)
    lst_test_matthews_corrcoef.append(var_test_matthews_corrcoef)
    lst_test_cohen_kappa_score.append(var_test_cohen_kappa_score)
end = time.time()    
kf_cross_validate = pd.DataFrame({'Name': lst_names, 
                                  'Fit Time': lst_fit_time, 
                                  'Score Time': lst_score_time, 
                                  'Test Accuracy': lst_test_accuracy, 
                                  'Test Balanced Accuracy': lst_test_balanced_accuracy, 
                                  'Test Average Precision': lst_test_average_precision, 
                                  'Test F1': lst_test_f1, 
                                  'Test Precision': lst_test_precision, 
                                  'Test Recall': lst_test_recall,
                                  'Test ROC AUC': lst_test_roc_auc, 
                                  'Test Matthews Correlation Coefficient': lst_test_matthews_corrcoef,
                                  'Test Cohen’s Kappa Score': lst_test_cohen_kappa_score})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_validate)

# Create a Pandas Excel writer using XlsxWriter as the engine.
cross_validate_classifier = pd.ExcelWriter('V3/cross_validate_classifier_shuffle_X_Y.xlsx', engine = 'xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
kf_cross_validate.to_excel(cross_validate_classifier, sheet_name = 'Dados')
# Close the Pandas Excel writer and output the Excel file.
cross_validate_classifier.save()

#Treino e Teste dos Modelos com CrossValidate + 10 StratifiedKFold(shuffle = False) + Várias Métricas - X, Y 

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import cohen_kappa_score

import time

metricas_cross_validate = { 'accuracy': 'accuracy',
                            'balanced_accuracy': 'balanced_accuracy',
                            'average_precision': 'average_precision',
                            'f1': 'f1',
                            'precision': 'precision',
                            'recall': 'recall',
                            'roc_auc': 'roc_auc',
                            'ms_matthews_corrcoef': make_scorer(matthews_corrcoef),
                            'ms_cohen_kappa_score': make_scorer(cohen_kappa_score) }

lst_names = []
lst_fit_time = []
lst_score_time = []
lst_test_accuracy = []
lst_test_balanced_accuracy = []
lst_test_average_precision = []
lst_test_f1 = []
lst_test_precision = []
lst_test_recall = []
lst_test_roc_auc = []
lst_test_matthews_corrcoef = []
lst_test_cohen_kappa_score = []

start = time.time()
for var_name, model in models:
    skfold = StratifiedKFold(n_splits = 10, shuffle = False, random_state = 42)
    result = cross_validate( estimator = model, 
                             X = X, 
                             y = Y, 
                             cv = skfold, 
                             scoring = metricas_cross_validate, 
                             return_train_score = False )    
    var_fit_time   = result['fit_time'].mean()
    var_score_time = result['score_time'].mean()
    var_test_accuracy = result['test_accuracy'].mean()
    var_test_balanced_accuracy = result['test_balanced_accuracy'].mean()
    var_test_average_precision = result['test_average_precision'].mean()
    var_test_f1 = result['test_f1'].mean()
    var_test_precision = result['test_precision'].mean()
    var_test_recall = result['test_recall'].mean()
    var_test_roc_auc = result['test_roc_auc'].mean()
    var_test_matthews_corrcoef = result['test_ms_matthews_corrcoef'].mean()
    var_test_cohen_kappa_score = result['test_ms_cohen_kappa_score'].mean()
    lst_names.append(var_name)
    lst_fit_time.append(var_fit_time)
    lst_score_time.append(var_score_time)
    lst_test_accuracy.append(var_test_accuracy)
    lst_test_balanced_accuracy.append(var_test_balanced_accuracy)
    lst_test_average_precision.append(var_test_average_precision)
    lst_test_f1.append(var_test_f1)
    lst_test_precision.append(var_test_precision)
    lst_test_recall.append(var_test_recall)
    lst_test_roc_auc.append(var_test_roc_auc)
    lst_test_matthews_corrcoef.append(var_test_matthews_corrcoef)
    lst_test_cohen_kappa_score.append(var_test_cohen_kappa_score)
end = time.time()    
kf_cross_validate = pd.DataFrame({'Name': lst_names, 
                                  'Fit Time': lst_fit_time, 
                                  'Score Time': lst_score_time, 
                                  'Test Accuracy': lst_test_accuracy, 
                                  'Test Balanced Accuracy': lst_test_balanced_accuracy, 
                                  'Test Average Precision': lst_test_average_precision, 
                                  'Test F1': lst_test_f1, 
                                  'Test Precision': lst_test_precision, 
                                  'Test Recall': lst_test_recall,
                                  'Test ROC AUC': lst_test_roc_auc, 
                                  'Test Matthews Correlation Coefficient': lst_test_matthews_corrcoef,
                                  'Test Cohen’s Kappa Score': lst_test_cohen_kappa_score})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_validate)
#Tempo de Execução: 1425.00 min

# Create a Pandas Excel writer using XlsxWriter as the engine.
cross_validate_classifier = pd.ExcelWriter('V3/cross_validate_classifier_X_Y_stratifiedkfold.xlsx', engine = 'xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
kf_cross_validate.to_excel(cross_validate_classifier, sheet_name = 'Dados')
# Close the Pandas Excel writer and output the Excel file.
cross_validate_classifier.save()

#Treino e Teste dos Modelos com CrossValidate + 10 StratifiedKFold(shuffle = True) + Várias Métricas - X, Y 

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import cohen_kappa_score

import time

metricas_cross_validate = { 'accuracy': 'accuracy',
                            'balanced_accuracy': 'balanced_accuracy',
                            'average_precision': 'average_precision',
                            'f1': 'f1',
                            'precision': 'precision',
                            'recall': 'recall',
                            'roc_auc': 'roc_auc',
                            'ms_matthews_corrcoef': make_scorer(matthews_corrcoef),
                            'ms_cohen_kappa_score': make_scorer(cohen_kappa_score) }

lst_names = []
lst_fit_time = []
lst_score_time = []
lst_test_accuracy = []
lst_test_balanced_accuracy = []
lst_test_average_precision = []
lst_test_f1 = []
lst_test_precision = []
lst_test_recall = []
lst_test_roc_auc = []
lst_test_matthews_corrcoef = []
lst_test_cohen_kappa_score = []

start = time.time()
for var_name, model in models:
    skfold = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)
    result = cross_validate( estimator = model, 
                             X = X, 
                             y = Y, 
                             cv = skfold, 
                             scoring = metricas_cross_validate, 
                             return_train_score = False )    
    var_fit_time   = result['fit_time'].mean()
    var_score_time = result['score_time'].mean()
    var_test_accuracy = result['test_accuracy'].mean()
    var_test_balanced_accuracy = result['test_balanced_accuracy'].mean()
    var_test_average_precision = result['test_average_precision'].mean()
    var_test_f1 = result['test_f1'].mean()
    var_test_precision = result['test_precision'].mean()
    var_test_recall = result['test_recall'].mean()
    var_test_roc_auc = result['test_roc_auc'].mean()
    var_test_matthews_corrcoef = result['test_ms_matthews_corrcoef'].mean()
    var_test_cohen_kappa_score = result['test_ms_cohen_kappa_score'].mean()
    lst_names.append(var_name)
    lst_fit_time.append(var_fit_time)
    lst_score_time.append(var_score_time)
    lst_test_accuracy.append(var_test_accuracy)
    lst_test_balanced_accuracy.append(var_test_balanced_accuracy)
    lst_test_average_precision.append(var_test_average_precision)
    lst_test_f1.append(var_test_f1)
    lst_test_precision.append(var_test_precision)
    lst_test_recall.append(var_test_recall)
    lst_test_roc_auc.append(var_test_roc_auc)
    lst_test_matthews_corrcoef.append(var_test_matthews_corrcoef)
    lst_test_cohen_kappa_score.append(var_test_cohen_kappa_score)
end = time.time()    
kf_cross_validate = pd.DataFrame({'Name': lst_names, 
                                  'Fit Time': lst_fit_time, 
                                  'Score Time': lst_score_time, 
                                  'Test Accuracy': lst_test_accuracy, 
                                  'Test Balanced Accuracy': lst_test_balanced_accuracy, 
                                  'Test Average Precision': lst_test_average_precision, 
                                  'Test F1': lst_test_f1, 
                                  'Test Precision': lst_test_precision, 
                                  'Test Recall': lst_test_recall,
                                  'Test ROC AUC': lst_test_roc_auc, 
                                  'Test Matthews Correlation Coefficient': lst_test_matthews_corrcoef,
                                  'Test Cohen’s Kappa Score': lst_test_cohen_kappa_score})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_validate)
#Tempo de Execução: 1411.73 min

# Create a Pandas Excel writer using XlsxWriter as the engine.
cross_validate_classifier = pd.ExcelWriter('V3/cross_validate_classifier_X_Y_stratifiedkfold_shuffle.xlsx', engine = 'xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
kf_cross_validate.to_excel(cross_validate_classifier, sheet_name = 'Dados')
# Close the Pandas Excel writer and output the Excel file.
cross_validate_classifier.save()

#ETAPA 04: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO KNN

#Criação do Modelo

from sklearn.neighbors import KNeighborsClassifier

classifier_knc = KNeighborsClassifier()

classifier_knc = KNeighborsClassifier( n_neighbors = 5, 
                                       metric = 'minkowski', 
                                       p = 2 )

classifier_knc.fit(X_train, Y_train)

Y_pred_knc = classifier_knc.predict(X_test)

#Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_knc)))

print('Precision Score : ' + str(precision_score(Y_test, Y_pred_knc)))

print('Recall Score : ' + str(recall_score(Y_test, Y_pred_knc)))

print('F1 Score : ' + str(f1_score(Y_test, Y_pred_knc)))

#Classification Report

from sklearn.metrics import classification_report  

print(classification_report(Y_test,Y_pred_knc))  

#Matriz de Confusão

from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_knc)))

#ETAPA 05: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO SVC

#Criação do Modelo

from sklearn.svm import SVC

classifier_svc = SVC( kernel = 'linear', 
                      random_state = 42 )

classifier_svc.fit(X_train, Y_train)

Y_pred_svc = classifier_svc.predict(X_test)

#Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_svc)))

print('Precision Score : ' + str(precision_score(Y_test, Y_pred_svc)))

print('Recall Score : ' + str(recall_score(Y_test, Y_pred_svc)))

print('F1 Score : ' + str(f1_score(Y_test, Y_pred_svc)))

#Classification Report

from sklearn.metrics import classification_report  

print(classification_report(Y_test,Y_pred_svc))  

#Matriz de Confusão

from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_svc)))

#ETAPA 06: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO REGRESSÃO LOGÍSTICA

#Criação do Modelo

from sklearn.linear_model import LogisticRegression

classifier_lr = LogisticRegression( random_state = 42 )

classifier_lr.fit(X_train, Y_train)

Y_pred_lr = classifier_lr.predict(X_test)

#Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_lr)))

print('Precision Score : ' + str(precision_score(Y_test, Y_pred_lr)))

print('Recall Score : ' + str(recall_score(Y_test, Y_pred_lr)))

print('F1 Score : ' + str(f1_score(Y_test, Y_pred_lr)))

#Classification Report

from sklearn.metrics import classification_report

print(classification_report(Y_test,Y_pred_lr))  

#Matriz de Confusão

from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_lr)))

#ETAPA 07: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO PERCEPTRON


#ETAPA 08: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO MLP

#Criação do Modelo

from sklearn.neural_network import MLPClassifier

classifier_mlp = MLPClassifier( verbose  = True,                       
                                max_iter = 1000,                            
                                tol = 0.000010 )

classifier_mlp.fit(X_train, Y_train)

Y_pred_mlp = classifier_mlp.predict(X_test)

#Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_mlp)))

print('Precision Score : ' + str(precision_score(Y_test, Y_pred_mlp)))

print('Recall Score : ' + str(recall_score(Y_test, Y_pred_mlp)))

print('F1 Score : ' + str(f1_score(Y_test, Y_pred_mlp)))

#Classification Report

from sklearn.metrics import classification_report
 
print(classification_report(Y_test,Y_pred_mlp))  

#Matriz de Confusão

from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_mlp)))

#ETAPA 09: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO GAUSSIAN NAIVE BAYES

#Criação do Modelo

from sklearn.naive_bayes import GaussianNB

classifier_gnb = GaussianNB()

classifier_gnb.fit(X_train, Y_train)

Y_pred_gnb = classifier_gnb.predict(X_test)

#Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_gnb)))

print('Precision Score : ' + str(precision_score(Y_test, Y_pred_gnb)))

print('Recall Score : ' + str(recall_score(Y_test, Y_pred_gnb)))

print('F1 Score : ' + str(f1_score(Y_test, Y_pred_gnb)))

#Classification Report

from sklearn.metrics import classification_report

print(classification_report(Y_test,Y_pred_gnb))

#Matriz de Confusão

from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_gnb)))

#ETAPA 10: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO BERNOULLI NAIVE BAYES

#ETAPA 11: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO EXTRA TREES CLASSIFIER

#ETAPA 12: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO RANDOM FOREST

#Criação do Modelo

from sklearn.ensemble import RandomForestClassifier

classifier_rf = RandomForestClassifier( n_estimators = 40, 
                                        criterion = 'entropy', 
                                        random_state = 42 )

classifier_rf.fit(X_train, Y_train)

Y_pred_rf = classifier_rf.predict(X_test)

#Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_rf)))

print('Precision Score : ' + str(precision_score(Y_test, Y_pred_rf)))

print('Recall Score : ' + str(recall_score(Y_test, Y_pred_rf)))

print('F1 Score : ' + str(f1_score(Y_test, Y_pred_rf)))

#Classification Report

from sklearn.metrics import classification_report
  
print(classification_report(Y_test,Y_pred_rf))  

#Matriz de Confusão

from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_rf)))

#Análise da Importância das Features (Variáveis)

importances_rf = list(classifier_rf.feature_importances_)

feature_importances_rf = [(feature, round(importance, 6)) for feature, importance in zip(X_list, importances_rf)]

feature_importances_rf = sorted(feature_importances_rf, key = lambda x: x[1], reverse = True)

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_rf];

#OTIMIZAÇÃO DE HIPERPARAMETROS - RANDOM FOREST

#Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

#Criação das Variáveis dos Parâmetros a Serem Testados

#n_estimators: It defines the number of decision trees to be created in a random forest.
#Generally, a higher number makes the predictions stronger and more stable, but a very large number can result in higher training time.
n_estimators_rf = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

#max_features : It defines the maximum number of features allowed for the split in each decision tree.
#Increasing max features usually improve performance but a very high number can decrease the diversity of each tree.
max_features_rf = ['auto', 'sqrt']

#max_depth: Random forest has multiple decision trees.
#This parameter defines the maximum depth of the trees.
#Max number of levels in each decision tree
max_depth_rf = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth_rf.append(None)

#min_samples_split: Used to define the minimum number of samples required in a leaf node before a split is attempted.
#If the number of samples is less than the required number, the node is not split.
#Min number of data points placed in a node before the node is split.
min_samples_split_rf = [2, 5, 10]

#min_samples_leaf: This defines the minimum number of samples required to be at a leaf node.
#Smaller leaf size makes the model more prone to capturing noise in train data.
#Min number of data points allowed in a leaf node.
min_samples_leaf_rf = [1, 2, 4]

#bootstrap: method for sampling data points (with or without replacement)
bootstrap_rf = [True, False]

# Criação e Fit do Random Grid

random_grid_rf = { 'n_estimators': n_estimators_rf,
                   'max_features': max_features_rf,
                   'max_depth': max_depth_rf,
                   'min_samples_split': min_samples_split_rf,
                   'min_samples_leaf': min_samples_leaf_rf,
                   'bootstrap': bootstrap_rf }

classifier_rf_rscv = RandomForestClassifier()

classifier_rf_rscv_random = RandomizedSearchCV( estimator = classifier_rf_rscv, 
                                                param_distributions = random_grid_rf, 
                                                n_iter = 100, 
                                                cv = 3, 
                                                verbose = 2, 
                                                random_state = 42,
                                                scoring = make_scorer(matthews_corrcoef),
                                                n_jobs = -1 )

classifier_rf_rscv_random.fit(X, Y)

classifier_rf_rscv_random.best_params_

classifier_rf_rscv_random.best_score_

classifier_rf_rscv_random.best_estimator_

classifier_rf_rscv_random.best_index_

#Grid Search With Cross Validation

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

classifier_rf_gscv = RandomForestClassifier()

classifier_rf_gscv.get_params().keys()

grid_param_rf = {
    'bootstrap': [True, False],
    'max_depth': [80, 90, 100, 110],
    'max_features': ['auto', 0.1, 0.2, 0.3, 2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 500, 800, 1000],
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes': [None, 2, 3, 5, 7, 9, 12, 15]
    
}

classifier_rf_gscv_gd_sr = GridSearchCV( estimator = classifier_rf_gscv,  
                                         param_grid = grid_param_rf,
                                         scoring = make_scorer(matthews_corrcoef),
                                         cv = 3,
                                         return_train_score = True,
                                         n_jobs = -1 )

classifier_rf_gscv_gd_sr.fit(X, Y) 

best_parameters_rf = classifier_rf_gscv_gd_sr.best_params_  

print(best_parameters_rf)

best_result_rf = classifier_rf_gscv_gd_sr.best_score_

print(best_result_rf)

pd.set_option('max_columns',200)

gd_sr_results_rf = pd.DataFrame(classifier_rf_gscv_gd_sr.cv_results_)

#ETAPA 13: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO ADA BOOST



#ETAPA 14: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO GRADIENT BOOSTING

#Criação do Modelo

from sklearn.ensemble import GradientBoostingClassifier

classifier_gbc = GradientBoostingClassifier()

classifier_gbc.fit(X_train, Y_train)

Y_pred_gbc = classifier_gbc.predict(X_test)

#Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_gbc)))

print('Precision Score : ' + str(precision_score(Y_test, Y_pred_gbc)))

print('Recall Score : ' + str(recall_score(Y_test, Y_pred_gbc)))

print('F1 Score : ' + str(f1_score(Y_test, Y_pred_gbc)))

#Classification Report

from sklearn.metrics import classification_report
 
print(classification_report(Y_test,Y_pred_gbc))  

#Matriz de Confusão

from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_gbc)))

#Análise da Importância das Features (Variáveis)

importances_gbc = list(classifier_gbc.feature_importances_)

feature_importances_gbc = [(feature, round(importance, 6)) for feature, importance in zip(X_list, importances_gbc)]

feature_importances_gbc = sorted(feature_importances_gbc, key = lambda x: x[1], reverse = True)

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_gbc];

#OTIMIZAÇÃO DE HIPERPARAMETROS - GRADIENT BOOST

#Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

#Criação das Variáveis dos Parâmetros a Serem Testados

#max_depth: The maximum depth of a tree.
#Used to control over-fitting as higher depth will allow model to learn relations very specific to a particular sample.
#Should be tuned using CV.
#The maximum depth limits the number of nodes in the tree. 
max_depth_gbc = [int(x) for x in np.linspace(start = 50, stop = 500, num = 10)]

#learning_rate: This determines the impact of each tree on the final outcome. 
#GBM works by starting with an initial estimate which is updated using the output of each tree. 
#The learning parameter controls the magnitude of this change in the estimates.
#Lower values are generally preferred as they make the model robust to the specific characteristics of tree and thus allowing it to generalize well.
#Lower values would require higher number of trees to model all the relations and will be computationally expensive.
#There is a trade-off between learning_rate and n_estimators.
learning_rate_gbc = [0.01, 0.03, 0.05, 0.1, 0.15]

#n_estimators: The number of sequential trees to be modeled.
#Though GBM is fairly robust at higher number of trees but it can still overfit at a point. 
#Hence, this should be tuned using CV for a particular learning rate.
n_estimators_gbc = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]

# Criação e Fit do Random Grid
random_grid_gbc = { 'max_depth': max_depth_gbc,
                    'learning_rate': learning_rate_gbc,
                    'n_estimators': n_estimators_gbc }

classifier_gbc_rscv = GradientBoostingClassifier()

classifier_gbc_rscv_random = RandomizedSearchCV( estimator = classifier_gbc_rscv, 
                                                 param_distributions = random_grid_gbc, 
                                                 n_iter = 10, 
                                                 cv = 3, 
                                                 verbose = 2, 
                                                 random_state = 42,
                                                 scoring = make_scorer(matthews_corrcoef),
                                                 n_jobs = 1 )

classifier_gbc_rscv_random.fit(X, Y)

classifier_gbc_rscv_random.best_params_

classifier_gbc_rscv_random.best_score_

classifier_gbc_rscv_random.best_estimator_

classifier_gbc_rscv_random.best_index_

#Grid Search With Cross Validation

from sklearn.model_selection import GridSearchCV

from sklearn.ensemble import GradientBoostingClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

classifier_gbc_gscv = GradientBoostingClassifier()

classifier_gbc_gscv.get_params().keys()

grid_param_gbc = {
    'max_depth': [50, 100, 150, 200, 250],
    'learning_rate': [0.03, 0.1, 0.15],
    'n_estimators': [100, 200, 300, 400, 500]
}

classifier_gbc_gscv_gd_sr = GridSearchCV( estimator = classifier_gbc_gscv,  
                                          param_grid = grid_param_gbc,
                                          scoring = make_scorer(matthews_corrcoef),
                                          cv = 3,
                                          return_train_score = True,
                                          n_jobs = -1)

classifier_gbc_gscv_gd_sr.fit(X, Y) 

best_parameters_gbc = classifier_gbc_gscv_gd_sr.best_params_

print(best_parameters_gbc)

best_result_gbc = classifier_gbc_gscv_gd_sr.best_score_

print(best_result_gbc)

pd.set_option('max_columns',200)

gd_sr_results_gbc = pd.DataFrame(classifier_gbc_gscv_gd_sr.cv_results_)

#ETAPA 15: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO XGBOOST

#Criação do Modelo

from xgboost import XGBClassifier

classifier_xgb = XGBClassifier( max_depth = 50, 
                                learning_rate = 0.16,
                                min_child_weight = 1,
                                n_estimators = 200 )

classifier_xgb.fit(X_train, Y_train)

Y_pred_xgb = classifier_xgb.predict(X_test)

#Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_xgb)))

print('Precision Score : ' + str(precision_score(Y_test, Y_pred_xgb)))

print('Recall Score : ' + str(recall_score(Y_test, Y_pred_xgb)))

print('F1 Score : ' + str(f1_score(Y_test, Y_pred_xgb)))

#Classification Report

from sklearn.metrics import classification_report
 
print(classification_report(Y_test,Y_pred_xgb))  

#Matriz de Confusão

from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_xgb)))

#Análise da Importância das Features (Variáveis)

importances_xgb = list(classifier_xgb.feature_importances_)

feature_importances_xgb = [(feature, round(importance, 6)) for feature, importance in zip(X_list, importances_xgb)]

feature_importances_xgb = sorted(feature_importances_xgb, key = lambda x: x[1], reverse = True)

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_xgb];

#OTIMIZAÇÃO DE HIPERPARAMETROS - XGBOOST

#Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

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

classifier_xgb_rscv = XGBClassifier()

classifier_xgb_rscv_random = RandomizedSearchCV( estimator = classifier_xgb_rscv, 
                                                 param_distributions = random_grid_xgb, 
                                                 n_iter = 10, 
                                                 cv = 3, 
                                                 verbose = 2, 
                                                 random_state = 42,
                                                 scoring = make_scorer(matthews_corrcoef),
                                                 n_jobs = 1 )

classifier_xgb_rscv_random.fit(X, Y)

classifier_xgb_rscv_random.best_params_

classifier_xgb_rscv_random.best_score_

classifier_xgb_rscv_random.best_estimator_

classifier_xgb_rscv_random.best_index_

#Grid Search With Cross Validation

from sklearn.model_selection import GridSearchCV

from xgboost import XGBClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

classifier_xgb_gscv = XGBClassifier()

classifier_xgb_gscv.get_params().keys()

grid_param_xgb = {
    'max_depth': [50, 100, 150, 200, 250],
    'learning_rate': [0.03, 0.1, 0.15],
    'min_child_weight': [1, 5, 9, 13, 17],
    'n_estimators': [100, 200, 300, 400, 500]    
}

classifier_xgb_gscv_gd_sr = GridSearchCV( estimator = classifier_xgb_gscv,  
                                          param_grid = grid_param_xgb,
                                          scoring = make_scorer(matthews_corrcoef),
                                          cv = 3,
                                          return_train_score = True,
                                          n_jobs = -1)

classifier_xgb_gscv_gd_sr.fit(X, Y) 

best_parameters_xgb = classifier_xgb_gscv_gd_sr.best_params_

print(best_parameters_xgb)

best_result_xgb = classifier_xgb_gscv_gd_sr.best_score_  

print(best_result_xgb)

pd.set_option('max_columns',200)

gd_sr_results_xgb = pd.DataFrame(classifier_xgb_gscv_gd_sr.cv_results_)

#ETAPA 16: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO LIGHT GBM

#Parametrização do Modelo

from lightgbm import LGBMClassifier

#1° Teste

classifier_lgbm = LGBMClassifier( max_depth = 50, 
                                  learning_rate = 0.1,
                                  num_leaves = 900,
                                  n_estimators = 300 )

Accuracy Score : 0.9516033371555098
Balanced Accuracy Score : 0.834114268416979
Precision Score : 0.8707274320771253
Recall Score : 0.6814128943758574
F1 Score : 0.764524817237399
Matthews Correlation Coefficient : 0.7448925111246457
Cohen’s Kappa Score : 0.7380013747144256

#2° Teste

classifier_lgbm = LGBMClassifier( max_depth = 100, 
                                  learning_rate = 0.15,
                                  num_leaves = 1000,
                                  n_estimators = 500 )

Accuracy Score : 0.9531058479300937
Balanced Accuracy Score : 0.8400335578698914
Precision Score : 0.8741349480968859
Recall Score : 0.6930727023319616
F1 Score : 0.7731446059678654
Matthews Correlation Coefficient : 0.7536463144514909
Cohen’s Kappa Score : 0.7473834356523674

#3° Teste

classifier_lgbm = LGBMClassifier( max_depth = 100, 
                                  learning_rate = 0.3,
                                  num_leaves = 1000,
                                  n_estimators = 1000 )

Accuracy Score : 0.9521568937566723
Balanced Accuracy Score : 0.83979548781142
Precision Score : 0.8645299145299146
Recall Score : 0.693758573388203
F1 Score : 0.769786910197869
Matthews Correlation Coefficient : 0.749055763205406
Cohen’s Kappa Score : 0.7434487016951405

#4° Teste

classifier_lgbm = LGBMClassifier( max_depth = 100, 
                                  learning_rate = 0.3,
                                  num_leaves = 500,
                                  n_estimators = 500 )

Accuracy Score : 0.9536989442884821
Balanced Accuracy Score : 0.8464827306097739
Precision Score : 0.8667507356031946
Recall Score : 0.7071330589849109
F1 Score : 0.778847969782814
Matthews Correlation Coefficient : 0.7581552549795394
Cohen’s Kappa Score : 0.753287348076416

#5° Teste

classifier_lgbm = LGBMClassifier( max_depth = 500, 
                                  learning_rate = 0.15,
                                  num_leaves = 2000,
                                  n_estimators = 2000 )

Accuracy Score : 0.9533826262306749
Balanced Accuracy Score : 0.8425759247764213
Precision Score : 0.8716302952503209
Recall Score : 0.698559670781893
F1 Score : 0.7755568246716161
Matthews Correlation Coefficient : 0.7556202208770963
Cohen’s Kappa Score : 0.7498991039900871

#6° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.15,
                                  num_leaves = 2000,
                                  n_estimators = 2000 )

Accuracy Score : 0.9533826262306749
Balanced Accuracy Score : 0.8425759247764213
Precision Score : 0.8716302952503209
Recall Score : 0.698559670781893
F1 Score : 0.7755568246716161
Matthews Correlation Coefficient : 0.7556202208770963
Cohen’s Kappa Score : 0.7498991039900871

#7° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.15,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 2000 )

Accuracy Score : 0.9538966430746115
Balanced Accuracy Score : 0.8500242545462906
Precision Score : 0.8615702479338843
Recall Score : 0.7150205761316872
F1 Score : 0.7814842578710645
Matthews Correlation Coefficient : 0.7600692398328626
Cohen’s Kappa Score : 0.7559626941346318

#8° Teste *****

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.1,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 2000 )

#Tempo de Execução: 13.17 min
Accuracy Score : 0.9538571033173856
Balanced Accuracy Score : 0.8514931221310282
Precision Score : 0.8582548136009832
Recall Score : 0.7184499314128944
F1 Score : 0.7821541907784209
Matthews Correlation Coefficient : 0.7603177280975907
Cohen’s Kappa Score : 0.7565765778768676

#9° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.01,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 2000 )

Accuracy Score : 0.9531453876873196
Balanced Accuracy Score : 0.84467866749431
Precision Score : 0.8647281921618205
Recall Score : 0.7037037037037037
F1 Score : 0.7759500850822462
Matthews Correlation Coefficient : 0.7550617254179193
Cohen’s Kappa Score : 0.7500946906186506

#10° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.01,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 5000 )

Accuracy Score : 0.9535407852595785
Balanced Accuracy Score : 0.8465424665302588
Precision Score : 0.8649895178197065
Recall Score : 0.7074759945130316
F1 Score : 0.778343708734201
Matthews Correlation Coefficient : 0.7574307277635234
Cohen’s Kappa Score : 0.7526851480744092

#11° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.01,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 500,
                                  n_estimators = 10000,
                                  colsample_bytree = 0.8 )

Accuracy Score : 0.9535012455023526
Balanced Accuracy Score : 0.8475639699289607
Precision Score : 0.8625
Recall Score : 0.7098765432098766
F1 Score : 0.7787810383747179
Matthews Correlation Coefficient : 0.7575338138410517
Cohen’s Kappa Score : 0.753074491659438

#12° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.01,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 20000,
                                  colsample_bytree = 0.8 )

Accuracy Score : 0.9541338816179669
Balanced Accuracy Score : 0.8507548183399621
Precision Score : 0.8625103220478944
Recall Score : 0.71639231824417
F1 Score : 0.782690146122143
Matthews Correlation Coefficient : 0.761374393406116
Cohen’s Kappa Score : 0.7572966267365179

#13° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.01,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 20000,
                                  colsample_bytree = 0.8 )

Accuracy Score : 0.9541338816179669
Balanced Accuracy Score : 0.8507548183399621
Precision Score : 0.8625103220478944
Recall Score : 0.71639231824417
F1 Score : 0.782690146122143
Matthews Correlation Coefficient : 0.761374393406116
Cohen’s Kappa Score : 0.7572966267365179

#14° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.01,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 5000,
                                  colsample_bytree = 0.8 )

Accuracy Score : 0.9536198647740303
Balanced Accuracy Score : 0.84718364484907
Precision Score : 0.8644918444165621
Recall Score : 0.7088477366255144
F1 Score : 0.7789711701526286
Matthews Correlation Coefficient : 0.757978708367287
Cohen’s Kappa Score : 0.7533457756504311

#15° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.01,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 5000,
                                  colsample_bytree = 0.6 )

Accuracy Score : 0.9535803250168043
Balanced Accuracy Score : 0.8449244775501759
Precision Score : 0.8687552921253175
Recall Score : 0.7037037037037037
F1 Score : 0.7775672603258811
Matthews Correlation Coefficient : 0.757175911157655
Cohen’s Kappa Score : 0.7519717909292564

#16° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.01,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 5000,
                                  colsample_bytree = 1.0 )

Accuracy Score : 0.9535407852595785
Balanced Accuracy Score : 0.8465424665302588
Precision Score : 0.8649895178197065
Recall Score : 0.7074759945130316
F1 Score : 0.778343708734201
Matthews Correlation Coefficient : 0.7574307277635234
Cohen’s Kappa Score : 0.7526851480744092

#17° Teste

classifier_lgbm = LGBMClassifier()

#LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
#               importance_type='split', learning_rate=0.1, max_depth=-1,
#               min_child_samples=20, min_child_weight=0.001, min_split_gain=0.0,
#               n_estimators=100, n_jobs=-1, num_leaves=31, objective=None,
#               random_state=None, reg_alpha=0.0, reg_lambda=0.0, silent=True,
#               subsample=1.0, subsample_for_bin=200000, subsample_freq=0)

Accuracy Score : 0.9524732118144794
Balanced Accuracy Score : 0.8477285713190947
Precision Score : 0.8518062397372742
Recall Score : 0.7115912208504801
F1 Score : 0.7754110612855007
Matthews Correlation Coefficient : 0.752863771330879
Cohen’s Kappa Score : 0.7490746069275055

#18° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.01,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 50000 )

Accuracy Score : 0.9544501996757739
Balanced Accuracy Score : 0.8530212888245167
Precision Score : 0.8611793611793612
Recall Score : 0.7211934156378601
F1 Score : 0.7849944008958568
Matthews Correlation Coefficient : 0.7634836568901676
Cohen’s Kappa Score : 0.7597439721533152

#19° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.001,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 5000,
                                  objective = 'binary',
                                  metric = 'binary_logloss' )

Accuracy Score : 0.9503380649242814
Balanced Accuracy Score : 0.8342939129901679
Precision Score : 0.8568357695614789
Recall Score : 0.6834705075445816
F1 Score : 0.7603967951163677
Matthews Correlation Coefficient : 0.7389170133805824
Cohen’s Kappa Score : 0.7330859147439391

#20° Teste

classifier_lgbm = LGBMClassifier( max_depth = 1000, 
                                  learning_rate = 0.01,
                                  num_leaves = 2000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 5000,
                                  objective = 'binary',
                                  metric = 'binary_logloss' )

Accuracy Score : 0.9535407852595785
Balanced Accuracy Score : 0.8465424665302588
Precision Score : 0.8649895178197065
Recall Score : 0.7074759945130316
F1 Score : 0.778343708734201
Matthews Correlation Coefficient : 0.7574307277635234
Cohen’s Kappa Score : 0.7526851480744092

#21° Teste

classifier_lgbm = LGBMClassifier( max_depth = 500, 
                                  learning_rate = 0.01,
                                  num_leaves = 1000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 5000,
                                  objective = 'binary',
                                  metric = 'binary_logloss' )

Accuracy Score : 0.9535407852595785
Balanced Accuracy Score : 0.8465424665302588
Precision Score : 0.8649895178197065
Recall Score : 0.7074759945130316
F1 Score : 0.778343708734201
Matthews Correlation Coefficient : 0.7574307277635234
Cohen’s Kappa Score : 0.7526851480744092

#22° Teste

classifier_lgbm = LGBMClassifier( max_depth = 500, 
                                  learning_rate = 0.01,
                                  num_leaves = 1500,
                                  min_data_in_leaf = 200,
                                  n_estimators = 5000,
                                  objective = 'binary',
                                  metric = 'binary_logloss' )

Accuracy Score : 0.9535407852595785
Balanced Accuracy Score : 0.8465424665302588
Precision Score : 0.8649895178197065
Recall Score : 0.7074759945130316
F1 Score : 0.778343708734201
Matthews Correlation Coefficient : 0.7574307277635234
Cohen’s Kappa Score : 0.7526851480744092

#23° Teste

classifier_lgbm = LGBMClassifier( max_depth = 500, 
                                  learning_rate = 0.01,
                                  num_leaves = 1000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 3000,
                                  objective = 'binary',
                                  metric = 'binary_logloss' )

Accuracy Score : 0.9532640069589973
Balanced Accuracy Score : 0.8453421921818363
Precision Score : 0.864592094196804
Recall Score : 0.7050754458161865
F1 Score : 0.7767283717415941
Matthews Correlation Coefficient : 0.75580196431452
Cohen’s Kappa Score : 0.7509294333213317

#24° Teste

classifier_lgbm = LGBMClassifier( max_depth = 500, 
                                  learning_rate = 0.01,
                                  num_leaves = 1000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 2000,
                                  objective = 'binary',
                                  metric = 'binary_logloss',
                                  random_state = 42 )

#Tempo de Execução: 28.70 min
Accuracy Score : 0.9531453876873196
Balanced Accuracy Score : 0.84467866749431
Precision Score : 0.8647281921618205
Recall Score : 0.7037037037037037
F1 Score : 0.7759500850822462
Matthews Correlation Coefficient : 0.7550617254179193
Cohen’s Kappa Score : 0.7500946906186506

#25° Teste

classifier_lgbm = LGBMClassifier( max_depth = 500, 
                                  learning_rate = 0.01,
                                  num_leaves = 1000,
                                  min_data_in_leaf = 200,
                                  n_estimators = 1000,
                                  objective = 'binary',
                                  metric = 'binary_logloss' )

Accuracy Score : 0.9529081491439643
Balanced Accuracy Score : 0.8466322888168533
Precision Score : 0.8583298712089739
Recall Score : 0.7085048010973937
F1 Score : 0.7762539921097126
Matthews Correlation Coefficient : 0.754519168751784
Cohen’s Kappa Score : 0.7502074135759076

#26° Teste

classifier_lgbm = LGBMClassifier( max_depth = 50, 
                                  learning_rate = 0.01,
                                  num_leaves = 1000,
                                  min_data_in_leaf = 50,
                                  n_estimators = 7625,
                                  objective = 'binary',
                                  metric = 'binary_logloss' )

#Tempo de Execução: 182.66 min
Accuracy Score : 0.9533826262306749
Balanced Accuracy Score : 0.844067138729874
Precision Score : 0.8684768773865083
Recall Score : 0.7019890260631001
F1 Score : 0.7764081168215436
Matthews Correlation Coefficient : 0.756012154918649
Cohen’s Kappa Score : 0.7507128443506828

#27° Teste

classifier_lgbm = LGBMClassifier( max_depth = 600, 
                                  learning_rate = 0.05,
                                  num_leaves = 8500,
                                  min_data_in_leaf = 300,
                                  n_estimators = 2750,
                                  objective = 'binary',
                                  metric = 'binary_logloss' )

Accuracy Score : 0.9536198647740303
Balanced Accuracy Score : 0.8491222229885587
Precision Score : 0.8605709557302441
Recall Score : 0.7133058984910837
F1 Score : 0.7800487530470656
Matthews Correlation Coefficient : 0.7585310937630548
Cohen’s Kappa Score : 0.7543789745865489

#28° Teste - Random Search

classifier_lgbm = LGBMClassifier( max_depth = 1880, 
                                  learning_rate = 0.1,
                                  num_leaves = 100,
                                  n_estimators = 4500,
                                  min_data_in_leaf = 140,
                                  n_jobs = 6 )

#Tempo de Execução: 3.79 min
#Tempo de Execução: 14.77 min
#Tempo de Execução: 3.53 min

Accuracy Score : 0.9546478984619035
Balanced Accuracy Score : 0.8555189629936164
Precision Score : 0.8582422033211826
Recall Score : 0.7266803840877915
F1 Score : 0.7870009285051067
Matthews Correlation Coefficient : 0.7651219344788733
Cohen’s Kappa Score : 0.7618186742639614

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(classifier_lgbm.get_params())

# Fit e Predição

import time
start = time.time()

classifier_lgbm.fit(X_train, Y_train)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

Y_pred_lgbm = classifier_lgbm.predict(X_test)

#Análise de Métricas

from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm = accuracy_score(Y_test, Y_pred_lgbm)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm))

#Balanced Accuracy

mtrc_balanced_accuracy_score_lgbm = balanced_accuracy_score(Y_test, Y_pred_lgbm)
print('Balanced Accuracy Score : ' + str(mtrc_balanced_accuracy_score_lgbm))

#Precision Score
#Número de vezes que uma classe foi predita corretamente dividida pelo número de vezes que a classe foi predita.

mtrc_precision_score_lgbm = precision_score(Y_test, Y_pred_lgbm)
print('Precision Score : ' + str(mtrc_precision_score_lgbm))

#Recall Score
#Número de vezes que uma classe foi predita corretamente (TP) dividido pelo número de vezes que a classe aparece no dado de teste (TP + FN).

mtrc_recall_score_lgbm = recall_score(Y_test, Y_pred_lgbm)
print('Recall Score : ' + str(mtrc_recall_score_lgbm))

#F1 Score
#Média harmônica entre precisão e revocação. Com essa informação podemos dizer a performance do classificador com um indicador apenas.

mtrc_f1_score_lgbm = f1_score(Y_test, Y_pred_lgbm)
print('F1 Score : ' + str(mtrc_f1_score_lgbm))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm = matthews_corrcoef(Y_test, Y_pred_lgbm)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm = cohen_kappa_score(Y_test, Y_pred_lgbm)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm))

#Classification Report

from sklearn.metrics import classification_report

mtrc_classification_report_lgbm = classification_report(Y_test,Y_pred_lgbm)
print('Classification Report : \n' + str(mtrc_classification_report_lgbm)) 

#Confusion Matrix

from sklearn.metrics import confusion_matrix

mtrc_confusion_matrix_lgbm = confusion_matrix(Y_test, Y_pred_lgbm)

print('Confusion Matrix : \n' + str(mtrc_confusion_matrix_lgbm))

print(pd.crosstab(Y_test, Y_pred_lgbm, rownames = ['Real'], colnames = ['Predito'], margins = True))

#Confusion Matrix - True Positives
mtrc_true_positive_lgbm = mtrc_confusion_matrix_lgbm[1, 1]

#Confusion Matrix - True Negatives
mtrc_true_negative_lgbm = mtrc_confusion_matrix_lgbm[0, 0]

#Confusion Matrix - False Positives
mtrc_false_positive_lgbm = mtrc_confusion_matrix_lgbm[0, 1]

#Confusion Matrix - False Negatives
mtrc_false_negative_lgbm = mtrc_confusion_matrix_lgbm[1, 0]

#cm_true_negative_lgbm, cm_false_positive_lgbm, cm_false_negative_lgbm, cm_true_positive_lgbm = confusion_matrix(Y_test, Y_pred_lgbm).ravel()

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

#Confusion Matrix - Misclassification Rate

mtrc_misclassification_rate_lgbm = ( mtrc_false_positive_lgbm + mtrc_false_negative_lgbm )/( mtrc_true_positive_lgbm + mtrc_true_negative_lgbm + mtrc_false_positive_lgbm + mtrc_false_negative_lgbm )
print('Misclassification Rate : ' + str(mtrc_misclassification_rate_lgbm))

#Confusion Matrix - Specificity

mtrc_specificity_lgbm = mtrc_true_negative_lgbm/( mtrc_true_negative_lgbm + mtrc_false_positive_lgbm )
print('Specificity/True Negative Rate : ' + str(mtrc_specificity_lgbm))

#Confusion Matrix - False Positive Rate

mtrc_false_positive_rate_lgbm = mtrc_false_positive_lgbm/( mtrc_true_negative_lgbm + mtrc_false_positive_lgbm )
print('False Positive Rate : ' + str(mtrc_false_positive_rate_lgbm))

#Confusion Matrix - Prevalence

mtrc_prevalence_lgbm = ( mtrc_false_negative_lgbm + mtrc_true_positive_lgbm )/( mtrc_true_positive_lgbm + mtrc_true_negative_lgbm + mtrc_false_positive_lgbm + mtrc_false_negative_lgbm )
print('Prevalence : ' + str(mtrc_prevalence_lgbm))

#ROC Curve

from sklearn.metrics import roc_curve

import matplotlib.pyplot as plt

Y_pred_prob_lgbm = classifier_lgbm.predict_proba(X_test)[:,1]

#Opção 1
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob_lgbm)
plt.figure(figsize = (14, 7))
plt.plot(fpr, tpr)
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.0])
plt.title('ROC Curve for LGBM Classifier')
plt.xlabel('False Positive Rate (1 — Specificity)')
plt.ylabel('True Positive Rate (Sensitivity)')
plt.grid(True)

#Opção 2
fpr, tpr, thresholds = roc_curve(Y_test, Y_pred_prob_lgbm)
plt.figure(figsize = (14, 7))
plt.plot(fpr, tpr, label = 'ROC curve')
plt.plot([0, 1], [0, 1], 'k--', label = 'Random Guess')
_ = plt.xlabel('False Positive Rate')
_ = plt.ylabel('True Positive Rate')
_ = plt.title('ROC Curve for LGBM Classifier')
_ = plt.xlim([-0.02, 1])
_ = plt.ylim([0, 1.02])
_ = plt.legend(loc = "lower right")

#Opção 3
def plot_roc_curve(false_positive_rate, true_positive_rate, label = None):
    plt.plot(false_positive_rate, true_positive_rate, linewidth = 2, label = label)
    plt.plot([0, 1], [0, 1], 'r', linewidth = 4)
    plt.axis([0, 1, 0, 1])
    plt.xlabel('False Positive Rate (FPR)', fontsize = 16)
    plt.ylabel('True Positive Rate (TPR)', fontsize = 16)

plt.figure(figsize = (14, 7))
plot_roc_curve(fpr, tpr)
plt.show()

#Area Under the Receiver Operating Characteristic Curve (ROC AUC)

from sklearn.metrics import roc_auc_score

mtrc_roc_auc_score_lgbm = roc_auc_score(Y_test, Y_pred_lgbm)
print('Area Under the Receiver Operating Characteristic Curve (ROC AUC) : ' + str(mtrc_roc_auc_score_lgbm))

#Precision-Recall Curve

from sklearn.metrics import precision_recall_curve 

#Opção 1
precision, recall, thresholds = precision_recall_curve(Y_test, Y_pred_prob_lgbm)
plt.figure(figsize = (14, 7))
plt.plot(precision, recall, label = 'Precision-Recall Curve')
_ = plt.xlabel('Precision')
_ = plt.ylabel('Recall')
_ = plt.title('Precision-Recall Curve')
_ = plt.legend(loc = "lower left")

#Opção 2
def plot_precision_and_recall(precision, recall, thresholds):
    plt.plot(thresholds, precision[:-1], "r-", label = "Precision", linewidth = 5)
    plt.plot(thresholds, recall[:-1], "b", label = "Recall", linewidth = 5)
    plt.xlabel("Thresholds", fontsize = 19)
    plt.legend(loc = "upper right", fontsize = 19)
    plt.ylim([0, 1])

plt.figure(figsize = (14, 7))
plot_precision_and_recall(precision, recall, thresholds)
plt.show()

#Opção 3
def plot_precision_vs_recall(precision, recall):
    plt.plot(recall, precision, "g--", linewidth = 2.5)
    plt.ylabel("Recall", fontsize = 19)
    plt.xlabel("Precision", fontsize = 19)
    plt.axis([0, 1.5, 0, 1.5])

plt.figure(figsize = (14, 7))
plot_precision_vs_recall(precision, recall)
plt.show()

#Area Under the Precision-Recall Curve

from sklearn.metrics import average_precision_score

mtrc_average_precision_score_lgbm = average_precision_score(Y_test, Y_pred_lgbm)
print('Area Under the Precision-Recall Curve : ' + str(mtrc_average_precision_score_lgbm))

#Calibration-Accuracy Plot

from dsutil.plotting import calibration_accuracy_plot

plt.figure(figsize = (14, 7))
calibration_accuracy_plot(Y_test, Y_pred_prob_lgbm)

#Calibration Curve

from sklearn.calibration import calibration_curve

from matplotlib import pyplot

#Gráfico Op01

plt.figure(figsize = (14, 7))
# reliability diagram
fop, mpv = calibration_curve(Y_test, Y_pred_prob_lgbm, n_bins = 20)
# plot perfectly calibrated
pyplot.plot([0, 1], [0, 1], linestyle = '--')
# plot model reliability
pyplot.plot(mpv, fop, marker = '.')
pyplot.show()

#Gráfico Op02

from sklearn.calibration import CalibratedClassifierCV

from sklearn.metrics import brier_score_loss

var_n_bins = 20

#Sigmoid Calibration

ccv_sig = CalibratedClassifierCV(classifier_lgbm, cv = 'prefit', method = 'sigmoid')
#ccv_sig = CalibratedClassifierCV(classifier_lgbm, cv = 10, method = 'sigmoid')
ccv_sig.fit(X_train, Y_train)

ccv_preds_sig = ccv_sig.predict_proba(X_test)[:,1]

#Isotonic Calibration

ccv_iso = CalibratedClassifierCV(classifier_lgbm, cv = 'prefit', method = 'isotonic')
#ccv_iso = CalibratedClassifierCV(classifier_lgbm, cv = 10, method = 'isotonic')
ccv_iso.fit(X_train,Y_train)

ccv_preds_iso = ccv_iso.predict_proba(X_test)[:,1]

#Criação do Gráfico

fig, axes = plt.subplots(1, 2, sharey = True)
ax = axes[0]
ax.set_xlim([-0.1,1.1])
ax.set_ylim([-0.1,1.1])
ax.plot([0, 1], [0, 1], "k:", label = "Perfect Calibration")

clf_score = brier_score_loss(Y_test, Y_pred_prob_lgbm, pos_label = 1)
fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, Y_pred_prob_lgbm, n_bins = var_n_bins)
ax.plot(mean_predicted_value, fraction_of_positives, "r-", label = "No Calibration (Brier loss={:.3f})".format(clf_score))

clf_score = brier_score_loss(Y_test, ccv_preds_sig, pos_label = 1)
fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, ccv_preds_sig, n_bins = var_n_bins)
ax.plot(mean_predicted_value, fraction_of_positives, "b-", label = "Sigmoid Calibration (Brier loss={:.3f})".format(clf_score))

ax.legend(loc = 'lower right')
ax.set_title('Original vs Sigmoid Calibration', size = 16)
plt.subplots_adjust(top = 0.85)

ax = axes[1]

ax.set_xlim([-0.1,1.1])
ax.set_ylim([-0.1,1.1])

ax.plot([0, 1], [0, 1], "k:", label = "Perfect Calibration")

clf_score = brier_score_loss(Y_test, Y_pred_prob_lgbm, pos_label = 1)
fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, Y_pred_prob_lgbm, n_bins = var_n_bins)
ax.plot(mean_predicted_value, fraction_of_positives, "r-", label = "No Calibration (Brier loss={:.3f})".format(clf_score))

clf_score = brier_score_loss(Y_test, ccv_preds_iso, pos_label = 1)
fraction_of_positives, mean_predicted_value = calibration_curve(Y_test, ccv_preds_iso, n_bins = var_n_bins)
ax.plot(mean_predicted_value, fraction_of_positives, "b-", label = "Isotonic Calibration (Brier loss={:.3f})".format(clf_score))

ax.legend(loc = 'lower right')
ax.set_title('Original vs Isotonic Calibration', size = 16)
plt.subplots_adjust(top = 0.85)

plt.gcf().set_size_inches(14, 7)
plt.show()

#Learning Curve - X_train, Y_train

import matplotlib.pyplot as plt

import numpy as np

from sklearn.model_selection import learning_curve

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time

start = time.time()
train_sizes_lc_lgbm, train_scores_lc_lgbm, test_scores_lc_lgbm = learning_curve( estimator = classifier_lgbm,
                                                                                 X = X_train,
                                                                                 y = Y_train,
                                                                                 train_sizes = np.linspace(0.1, 1.0, 5),
                                                                                 cv = 10,
                                                                                 scoring = make_scorer(matthews_corrcoef),
                                                                                 shuffle = True,
                                                                                 random_state = 42 )
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 288.69 min

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

plt.grid()
plt.xlabel('Number of Training Samples')
plt.ylabel('Matthews Correlation Coefficient (MCC)')
plt.legend(loc = 'lower right')
plt.ylim([0.5, 1.03])
plt.tight_layout()
#plt.savefig('images/06_05.png', dpi = 300)
plt.show()

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
                                                                                 train_sizes = np.linspace(0.1, 1.0, 10),
                                                                                 cv = 10,
                                                                                 scoring = make_scorer(matthews_corrcoef),
                                                                                 shuffle = True,
                                                                                 random_state = 42 )
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 252.40 min

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

plt.grid()
plt.xlabel('Number of Training Samples')
plt.ylabel('Matthews Correlation Coefficient (MCC)')
plt.legend(loc = 'lower right')
plt.ylim([0.5, 1.03])
plt.tight_layout()
#plt.savefig('images/06_05.png', dpi = 300)
plt.show()

#Validation Curve - X_train, Y_train

from sklearn.model_selection import validation_curve

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time

param_range_vc = [3500, 4000, 4500, 5000, 5500, 6000]

start = time.time()
train_scores_vc_lgbm, test_scores_vc_lgbm = validation_curve( estimator = classifier_lgbm, 
                                                              X = X_train, 
                                                              y = Y_train, 
                                                              param_name = 'n_estimators', 
                                                              param_range = param_range_vc,
                                                              scoring = make_scorer(matthews_corrcoef),
                                                              cv = 10 )
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 202.01 min

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

plt.grid()
#plt.xscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('Parameter n Estimator')
plt.ylabel('Matthews Correlation Coefficient (MCC)')
plt.ylim([0.5, 1.03])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi = 300)
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
#Tempo de Execução: 351.31 min

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

plt.grid()
#plt.xscale('log')
plt.legend(loc = 'lower right')
plt.xlabel('Parameter n Estimator')
plt.ylabel('Matthews Correlation Coefficient (MCC)')
plt.ylim([0.5, 1.03])
plt.tight_layout()
# plt.savefig('images/06_06.png', dpi = 300)
plt.show()

#Análise da Importância das Features (Variáveis)

# Get numerical feature importances
importances_lgbm = list(classifier_lgbm.feature_importances_)
# List of tuples with variable and importance
feature_importances_lgbm = [(feature, round(importance, 2)) for feature, importance in zip(X_list, importances_lgbm)]

# Sort the feature importances by most important first
feature_importances_lgbm = sorted(feature_importances_lgbm, key = lambda x: x[1], reverse = True)

# Print out the feature and importances 
[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_lgbm];

#Análise da Importância das Features (Variáveis) - Data Frame

feature_importances_lgbm_df = pd.DataFrame({'Feature': X_list, 'Importance': np.round(classifier_lgbm.feature_importances_,3)})

feature_importances_lgbm_df = feature_importances_lgbm_df.sort_values('Importance', ascending = False).set_index('Feature')

feature_importances_lgbm_df.head(478)

#Análise Gráfica da Importância das Features (Variáveis) - Op1

import matplotlib.pyplot as plt

importances_lgbm_array = classifier_lgbm.feature_importances_
ind_importances_lgbm_array = np.argsort(importances_lgbm_array)
# Seleciona as n Features Mais Importantes e Faz o Gráfico
quant = 30
plt.figure(figsize = (10, 10))
plt.title('Feature Importances')
plt.barh(range(quant), importances_lgbm_array[ind_importances_lgbm_array][-quant:], color = 'b', align = 'center')
plt.yticks(range(quant), [X_list[i] for i in ind_importances_lgbm_array[-quant:]])
plt.xlabel('Relative Importance')
plt.show()

#Análise Gráfica da Importância das Features (Variáveis) - Op2

import matplotlib.pyplot as plt

plt.figure(figsize = (84, 7))
# list of x locations for plotting
x_values_lgbm = list(range(len(importances_lgbm)))
# Make a bar chart
plt.bar(x_values_lgbm, importances_lgbm, orientation = 'vertical', color = 'r', edgecolor = 'k', linewidth = 1.2)
# Tick labels for x axis
plt.xticks(x_values_lgbm, X_list, rotation = 'vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

#Análise Gráfica da Importância das Features (Variáveis) - Op3

plt.figure(figsize = (84, 7))
# list of x locations for plotting
x_values_lgbm = list(range(len(importances_lgbm)))
# Make a bar chart
plt.bar(x_values_lgbm, importances_lgbm, orientation = 'vertical')
# Tick labels for x axis
plt.xticks(x_values_lgbm, X_list, rotation = 'vertical')
# Axis labels and title
plt.ylabel('Importance'); plt.xlabel('Variable'); plt.title('Variable Importances');

#Gráfico de Importâncias - Cumulativo

import matplotlib.pyplot as plt

# List of features sorted from most to least important
sorted_importances_lgbm = [importance[1] for importance in feature_importances_lgbm]
sorted_features_lgbm = [importance[0] for importance in feature_importances_lgbm]
# Cumulative importances
cumulative_importances_lgbm = np.cumsum(sorted_importances_lgbm)
# Make a line graph
plt.figure(figsize = (84, 7))
plt.plot(x_values_lgbm, cumulative_importances_lgbm, 'g-')
# Draw line at 95% of importance retained
plt.hlines(y = 0.95, xmin = 0, xmax = len(sorted_importances_lgbm), color = 'r', linestyles = 'dashed')
# Format x ticks and labels
plt.xticks(x_values_lgbm, sorted_features_lgbm, rotation = 'vertical')
# Axis labels and title
plt.xlabel('Variable'); plt.ylabel('Cumulative Importance'); plt.title('Cumulative Importances');

#Criação de Novas Bases X_train e X_test com Features Mais Importantes

#Nomes das Features Mais Importantes
important_feature_names = [feature[0] for feature in feature_importances_lgbm[0:478]]
#Indices das Features Mais Importantes
important_indices = [X_list.index(feature) for feature in important_feature_names]
#Criação da Nova Base X_train
X_train_important_features = X_train[:, important_indices]
#Criação da Nova Base X_test
X_test_important_features = X_test[:, important_indices]

#Validação das Operações Realizadas
print('Important Train Features Shape:', X_train_important_features.shape)
print('Important Test Features Shape:', X_test_important_features.shape)

Important Train Features Shape: (59012, 478)
Important Test Features Shape: (25291, 478)

#Parametrização de Novo Classificador

from lightgbm import LGBMClassifier

#classifier_lgbm = LGBMClassifier( max_depth = 1880, 
#                                  learning_rate = 0.1,
#                                  num_leaves = 100,
#                                  n_estimators = 4500,
#                                  min_data_in_leaf = 140 )

classifier_lgbm_if = LGBMClassifier( max_depth = 1880, 
                                     learning_rate = 0.1,
                                     num_leaves = 100,
                                     n_estimators = 4500,
                                     min_data_in_leaf = 140 )

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(classifier_lgbm_if.get_params())

# Fit e Predição

import time
start = time.time()

classifier_lgbm_if.fit(X_train_important_features, Y_train)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 4.52 min
#Tempo de Execução: 13.46 min

Y_pred_lgbm_if = classifier_lgbm_if.predict(X_test_important_features)

#All Features
Accuracy Score : 0.9546478984619035
Balanced Accuracy Score : 0.8555189629936164
Precision Score : 0.8582422033211826
Recall Score : 0.7266803840877915
F1 Score : 0.7870009285051067
Matthews Correlation Coefficient : 0.7651219344788733
Cohen’s Kappa Score : 0.7618186742639614

#Reduced Features
Accuracy Score : 0.9546478984619035
Balanced Accuracy Score : 0.8555189629936164
Precision Score : 0.8582422033211826
Recall Score : 0.7266803840877915
F1 Score : 0.7870009285051067
Matthews Correlation Coefficient : 0.7651219344788733
Cohen’s Kappa Score : 0.7618186742639614

#Análise de Métricas

from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm_if = accuracy_score(Y_test, Y_pred_lgbm_if)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm_if))

#Balanced Accuracy

mtrc_balanced_accuracy_score_lgbm_if = balanced_accuracy_score(Y_test, Y_pred_lgbm_if)
print('Balanced Accuracy Score : ' + str(mtrc_balanced_accuracy_score_lgbm_if))

#Precision Score
#Número de vezes que uma classe foi predita corretamente dividida pelo número de vezes que a classe foi predita.

mtrc_precision_score_lgbm_if = precision_score(Y_test, Y_pred_lgbm_if)
print('Precision Score : ' + str(mtrc_precision_score_lgbm_if))

#Recall Score
#Número de vezes que uma classe foi predita corretamente (TP) dividido pelo número de vezes que a classe aparece no dado de teste (TP + FN).

mtrc_recall_score_lgbm_if = recall_score(Y_test, Y_pred_lgbm_if)
print('Recall Score : ' + str(mtrc_recall_score_lgbm_if))

#F1 Score
#Média harmônica entre precisão e revocação. Com essa informação podemos dizer a performance do classificador com um indicador apenas.

mtrc_f1_score_lgbm_if = f1_score(Y_test, Y_pred_lgbm_if)
print('F1 Score : ' + str(mtrc_f1_score_lgbm_if))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm_if = matthews_corrcoef(Y_test, Y_pred_lgbm_if)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm_if))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm_if = cohen_kappa_score(Y_test, Y_pred_lgbm_if)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm_if))

#CrossValScore - 10 KFolds - Seed 21 - X_train, Y_train

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

kfold_lgbm = KFold(n_splits = 10, shuffle = True, random_state = 21)

#skf_lgbm = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 21)

start = time.time()
scores_lgbm = cross_val_score(classifier_lgbm, X_train, Y_train, cv = kfold_lgbm, scoring = make_scorer(matthews_corrcoef)) 
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

display_scores(scores_lgbm)

#Tempo de Execução: 32.75 min
Scores: [0.75322082 0.78091384 0.75657978 0.74190464 0.77751352 0.78002321
 0.74965155 0.73152315 0.75233359 0.75060775]
Mean: 0.7574271840303514
Standard Deviation: 0.01590268586863797

#CrossValScore - 10 KFolds - Seed 21 - X, Y

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

kfold_lgbm = KFold(n_splits = 10, shuffle = True, random_state = 21)

#skf_lgbm = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 21)

start = time.time()
scores_lgbm = cross_val_score(classifier_lgbm, X, Y, cv = kfold_lgbm, scoring = make_scorer(matthews_corrcoef)) 
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

display_scores(scores_lgbm)

#Tempo de Execução: 44.47 min
Scores: [0.7625348  0.75042917 0.76281597 0.7768494  0.76055565 0.76342349
 0.7813639  0.76166769 0.77031238 0.76337046]
Mean: 0.7653322899200401
Standard Deviation: 0.00833340614570456

#CrossValScore - 10 KFolds - Seed 42 - X_train, Y_train

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

kfold_lgbm = KFold(n_splits = 10, shuffle = True, random_state = 42)

#skf_lgbm = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

start = time.time()
scores_lgbm = cross_val_score(classifier_lgbm, X_train, Y_train, cv = kfold_lgbm, scoring = make_scorer(matthews_corrcoef)) 
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

display_scores(scores_lgbm)

#Tempo de Execução: 38.03 min
Scores: [0.75989547 0.73667452 0.75448709 0.78309003 0.76297575 0.73949739
 0.75440817 0.74116896 0.75067762 0.76196894]
Mean: 0.7544843941006902
Standard Deviation: 0.01307395705785211

#CrossValScore - 10 KFolds - Seed 42 - X, Y

from sklearn.model_selection import cross_val_score

from sklearn.model_selection import KFold

from sklearn.model_selection import StratifiedKFold

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

def display_scores(scores):
    print("Scores:", scores)
    print("Mean:", scores.mean())
    print("Standard Deviation:", scores.std())

kfold_lgbm = KFold(n_splits = 10, shuffle = True, random_state = 42)

#skf_lgbm = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

start = time.time()
scores_lgbm = cross_val_score(classifier_lgbm, X, Y, cv = kfold_lgbm, scoring = make_scorer(matthews_corrcoef)) 
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

display_scores(scores_lgbm)

Tempo de Execução: 41.90 min
Scores: [0.75945617 0.78188393 0.76230049 0.75890242 0.75995801 0.76253148
 0.76208016 0.77305743 0.75913343 0.77788111]
Mean: 0.7657184628414038
Standard Deviation: 0.00812656081962441

#CrossValidate - 10 KFolds - X_train, Y_train - Múltiplas Métricas

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
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
                                             X = X_train, 
                                             y = Y_train, 
                                             cv = skf_lgbm, 
                                             scoring = metricas_cross_validate, 
                                             return_train_score = False )
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
#Tempo de Execução: 34.52 min

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
print('Test Average Precision LGBM: ' + str(cv_test_average_precision_lgbm))

cv_test_average_precision_std_lgbm = result_lgbm_cross_validate['test_average_precision'].std()
print('STD Test Average Precision LGBM: ' + str(cv_test_average_precision_std_lgbm))

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

#CrossValidate - 10 KFolds - X, Y - Múltiplas Métricas

from sklearn.model_selection import cross_validate
from sklearn.metrics import accuracy_score
from sklearn.metrics import balanced_accuracy_score
from sklearn.metrics import average_precision_score
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from sklearn.metrics import roc_auc_score
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
#Tempo de Execução: 46.46 min

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
print('Test Average Precision LGBM: ' + str(cv_test_average_precision_lgbm))

cv_test_average_precision_std_lgbm = result_lgbm_cross_validate['test_average_precision'].std()
print('STD Test Average Precision LGBM: ' + str(cv_test_average_precision_std_lgbm))

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

Fit Time LGBM: 277.01844930648804
Score Time LGBM: 1.671333384513855
Test Accuracy LGBM: 0.9548652384797502
STD Test Accuracy LGBM: 0.0019329130215183944
Test Balanced Accuracy LGBM: 0.858164284125046
STD Test Balanced Accuracy LGBM: 0.006667598527929935
Test Average Precision LGBM: 0.8791628324238235
STD Test Average Precision LGBM: 0.009966631473748303
Test F1 LGBM: 0.7890875249275481
STD Test F1 LGBM: 0.009323204294300431
Test Precision LGBM: 0.8554732398865225
STD Test Precision LGBM: 0.013239772221887618
Test Recall LGBM: 0.7324848592728211
STD Test Recall LGBM: 0.013688031965650408
Test ROC AUC LGBM: 0.9779940308655328
STD Test ROC AUC LGBM: 0.0018446095868624567
Test MCC Score LGBM: 0.7669504228841302
STD Test MCC Score LGBM: 0.01013870155450071
Test Cohen Kappa Score LGBM: 0.7639899001981696
STD Test Cohen Kappa Score LGBM: 0.01033681765787603

#Permutation Test Score

import numpy as np

import matplotlib.pyplot as plt

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import permutation_test_score

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

skf_lgbm = StratifiedKFold(n_splits = 10, shuffle = True, random_state = 42)

n_classes = np.unique(Y_train).size

import time

start = time.time()

score_lgbm, permutation_scores_lgbm, pvalue_lgbm = permutation_test_score(
    classifier_lgbm, X_train, Y_train, scoring = make_scorer(matthews_corrcoef), cv = skf_lgbm, n_permutations = 100, n_jobs = 4 )

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

print("Classification Score %s (pvalue : %s)" % (score_lgbm, pvalue_lgbm))

# View Histogram of Permutation Scores
plt.figure(figsize = (14, 7))
plt.hist(permutation_scores_lgbm, 20, label = 'Permutation Scores', edgecolor = 'black')
ylim = plt.ylim()

#plt.plot(2 * [score_lgbm], ylim, '--g', linewidth = 3, label = 'Classification Score' (pvalue %s)' % pvalue_lgbm)
#plt.plot(2 * [1. / n_classes], ylim, '--k', linewidth = 3, label = 'Luck')

#plt.vlines(score_lgbm, ylim[0], ylim[1], linestyle = '--', color = 'g', linewidth = 3, label = 'Classification Score (pvalue %s)' % pvalue_lgbm)
#plt.vlines(1.0 / n_classes, ylim[0], ylim[1], linestyle = '--', color = 'k', linewidth = 3, label = 'Luck')

plt.ylim(ylim)
plt.legend()
plt.xlabel('Score')
plt.show()

#OTIMIZAÇÃO DE HIPERPARAMETROS - LIGHT GBM

#TESTE 01: Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from lightgbm import LGBMClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

#Criação das Variáveis dos Parâmetros a Serem Testados

#max_depth:It describes the maximum depth of tree. This parameter is used to handle model overfitting. 
#Any time you feel that your model is overfitted, my first advice will be to lower max_depth.
max_depth_lgbm = [int(x) for x in np.linspace(start = 200, stop = 3000, num = 11)]

#learning_rate:  This determines the impact of each tree on the final outcome. 
#GBM works by starting with an initial estimate which is updated using the output of each tree. 
#The learning parameter controls the magnitude of this change in the estimates. 
#Typical values: 0.1, 0.001, 0.003…
learning_rate_lgbm = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.15]

#num_leaves: This parameter is used to set the number of leaves to be formed in a tree.
#In case of Light GBM, since splitting takes place leaf-wise rather than depth-wise, num_leaves must be smaller than 2^(max_depth), otherwise, it may lead to overfitting.
#This is the main parameter to control the complexity of the tree model. 
#Ideally, the value of num_leaves should be less than or equal to 2^(max_depth). 
#Value more than this will result in overfitting.
num_leaves_lgbm = [int(x) for x in np.linspace(100, 3000, num = 30)]

#min_data_in_leaf:  It is the minimum number of the records a leaf may have. 
#The default value is 20, optimum value. 
#A very small value may cause overfitting.
#It is also one of the most important parameters in dealing with overfitting.
#Setting it to a large value can avoid growing too deep a tree, but may cause under-fitting. 
#In practice, setting it to hundreds or thousands is enough for a large dataset.
min_data_in_leaf_lgbm = [int(x) for x in np.linspace(20, 1000, num = 50)]

#n_estimators:
n_estimators_lgbm = [int(x) for x in np.linspace(500, 5000, num = 10)]

# Criação e Fit do Random Grid

random_grid_lgbm = { 'max_depth': max_depth_lgbm,
                     'learning_rate': learning_rate_lgbm,
                     'num_leaves': num_leaves_lgbm,
                     'min_data_in_leaf': min_data_in_leaf_lgbm,
                     'n_estimators': n_estimators_lgbm }

pprint(random_grid_lgbm)

classifier_lgbm_rscv = LGBMClassifier()

classifier_lgbm_rscv_random = RandomizedSearchCV( estimator = classifier_lgbm_rscv, 
                                                  param_distributions = random_grid_lgbm, 
                                                  n_iter = 30, 
                                                  cv = 3, 
                                                  random_state = 42,
                                                  scoring = make_scorer(matthews_corrcoef) )

import time

start = time.time()

classifier_lgbm_rscv_random.fit(X_train, Y_train)

end = time.time()

print("Tempo de Execução: {} min".format((end - start)/60))

Tempo de Execução: 189.00136020183564 min

classifier_lgbm_rscv_random.best_params_

{'num_leaves': 100, 'n_estimators': 4500, 'min_data_in_leaf': 140, 'max_depth': 1880, 'learning_rate': 0.1}

classifier_lgbm_rscv_random.best_score_

0.750779135590919

classifier_lgbm_rscv_random.best_estimator_

LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.1, max_depth=1880,
               min_child_samples=20, min_child_weight=0.001,
               min_data_in_leaf=140, min_split_gain=0.0, n_estimators=4500,
               n_jobs=-1, num_leaves=100, objective=None, random_state=None,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)

classifier_lgbm = classifier_lgbm_rscv_random.best_estimator_

Accuracy Score : 0.9546478984619035
Balanced Accuracy Score : 0.8555189629936164
Precision Score : 0.8582422033211826
Recall Score : 0.7266803840877915
F1 Score : 0.7870009285051067
Matthews Correlation Coefficient : 0.7651219344788733
Cohen’s Kappa Score : 0.7618186742639614

classifier_lgbm_rscv_random.best_index_

23

#TESTE 02: Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from lightgbm import LGBMClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

#Criação das Variáveis dos Parâmetros a Serem Testados

#max_depth:It describes the maximum depth of tree. This parameter is used to handle model overfitting. 
#Any time you feel that your model is overfitted, my first advice will be to lower max_depth.
max_depth_lgbm = [int(x) for x in np.linspace(start = 300, stop = 3000, num = 19)]

#learning_rate:  This determines the impact of each tree on the final outcome. 
#GBM works by starting with an initial estimate which is updated using the output of each tree. 
#The learning parameter controls the magnitude of this change in the estimates. 
#Typical values: 0.1, 0.001, 0.003…
learning_rate_lgbm = [0.001, 0.003, 0.01, 0.03, 0.05, 0.1, 0.15]

#num_leaves: This parameter is used to set the number of leaves to be formed in a tree.
#In case of Light GBM, since splitting takes place leaf-wise rather than depth-wise, num_leaves must be smaller than 2^(max_depth), otherwise, it may lead to overfitting.
#This is the main parameter to control the complexity of the tree model. 
#Ideally, the value of num_leaves should be less than or equal to 2^(max_depth). 
#Value more than this will result in overfitting.
num_leaves_lgbm = [int(x) for x in  np.linspace(100, 3000, num = 59)]

#min_data_in_leaf:  It is the minimum number of the records a leaf may have. 
#The default value is 20, optimum value. 
#A very small value may cause overfitting.
#It is also one of the most important parameters in dealing with overfitting.
#Setting it to a large value can avoid growing too deep a tree, but may cause under-fitting. 
#In practice, setting it to hundreds or thousands is enough for a large dataset.
min_data_in_leaf_lgbm = [int(x) for x in np.linspace(20, 1000, num = 50)]

#n_estimators:
n_estimators_lgbm = [int(x) for x in np.linspace(200, 5000, num = 25)]

# Criação e Fit do Random Grid

random_grid_lgbm = { 'max_depth': max_depth_lgbm,
                     'learning_rate': learning_rate_lgbm,
                     'num_leaves': num_leaves_lgbm,
                     'min_data_in_leaf': min_data_in_leaf_lgbm,
                     'n_estimators': n_estimators_lgbm }

pprint(random_grid_lgbm)

classifier_lgbm_rscv = LGBMClassifier()

classifier_lgbm_rscv_random = RandomizedSearchCV( estimator = classifier_lgbm_rscv, 
                                                  param_distributions = random_grid_lgbm, 
                                                  n_iter = 90, 
                                                  cv = 3, 
                                                  random_state = 42,
                                                  scoring = make_scorer(matthews_corrcoef) )

import time

start = time.time()

classifier_lgbm_rscv_random.fit(X_train, Y_train)

end = time.time()

print("Tempo de Execução: {} min".format((end - start)/60))

Tempo de Execução: 624.8046803553899 min

classifier_lgbm_rscv_random.best_params_

{'num_leaves': 3000, 'n_estimators': 4200, 'min_data_in_leaf': 100, 'max_depth': 1950, 'learning_rate': 0.05}

classifier_lgbm_rscv_random.best_score_

0.7465495529665019

classifier_lgbm_rscv_random.best_estimator_

LGBMClassifier(boosting_type='gbdt', class_weight=None, colsample_bytree=1.0,
               importance_type='split', learning_rate=0.05, max_depth=1950,
               min_child_samples=20, min_child_weight=0.001,
               min_data_in_leaf=100, min_split_gain=0.0, n_estimators=4200,
               n_jobs=-1, num_leaves=3000, objective=None, random_state=None,
               reg_alpha=0.0, reg_lambda=0.0, silent=True, subsample=1.0,
               subsample_for_bin=200000, subsample_freq=0)

classifier_lgbm = classifier_lgbm_rscv_random.best_estimator_

Accuracy Score : 0.9541734213751928
Balanced Accuracy Score : 0.8486894651738435
Precision Score : 0.8671124111993314
Recall Score : 0.7115912208504801
F1 Score : 0.7816914673196459
Matthews Correlation Coefficient : 0.7609812900138212
Cohen’s Kappa Score : 0.7563684784231534

classifier_lgbm_rscv_random.best_index_

0

#Grid Search With Cross Validation

from sklearn.model_selection import GridSearchCV

from lightgbm import LGBMClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

classifier_lgbm_gscv = LGBMClassifier()

classifier_lgbm_gscv.get_params().keys()

grid_param_lgbm = {
    'max_depth': [500, 1000, 1500, 2000, 2500],
    'learning_rate': [0.01, 0.03, 0.05, 0.1, 0.15],
    'num_leaves': [1000, 2000, 3000, 4000, 5000],
    'min_data_in_leaf': [100, 200, 300, 400, 500],
    'n_estimators': [1000, 3000, 5000, 7000, 9000],
    'colsample_bytree' : [0.6, 0.7, 0.8, 0.9, 1.0],
}

classifier_lgbm_gscv_gd_sr = GridSearchCV( estimator = classifier_lgbm_gscv,  
                                           param_grid = grid_param_lgbm,
                                           scoring = make_scorer(matthews_corrcoef),
                                           cv = 3,
                                           return_train_score = True,
                                           n_jobs = -1 )

classifier_lgbm_gscv_gd_sr.fit(X, Y) 

best_parameters_lgbm = classifier_lgbm_gscv_gd_sr.best_params_
  
print(best_parameters_lgbm)

best_result_lgbm = classifier_lgbm_gscv_gd_sr.best_score_
 
print(best_result_lgbm)

pd.set_option('max_columns',200)

gd_sr_results_lgbm = pd.DataFrame(classifier_lgbm_gscv_gd_sr.cv_results_)

#ETAPA 17: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO CATBOOST

#Criação do Modelo

from catboost import CatBoostClassifier

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 500, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.15 )

classifier_cbc.fit(X_train, Y_train)

Y_pred_cbc = classifier_cbc.predict(X_test)

#Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_cbc)))

print('Precision Score : ' + str(precision_score(Y_test, Y_pred_cbc)))

print('Recall Score : ' + str(recall_score(Y_test, Y_pred_cbc)))

print('F1 Score : ' + str(f1_score(Y_test, Y_pred_cbc)))

#Classification Report

from sklearn.metrics import classification_report  

print(classification_report(Y_test,Y_pred_cbc))  

#Matriz de Confusão

from sklearn.metrics import confusion_matrix

print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_cbc)))

#Análise da Importância das Features (Variáveis)

importances_cbc = list(classifier_cbc.feature_importances_)

feature_importances_cbc = [(feature, round(importance, 6)) for feature, importance in zip(X_list, importances_cbc)]

feature_importances_cbc = sorted(feature_importances_cbc, key = lambda x: x[1], reverse = True)

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_cbc];

#OTIMIZAÇÃO DE HIPERPARAMETROS - CATBOOST

#Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from catboost import CatBoostClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

#Criação das Variáveis dos Parâmetros a Serem Testados

#depth: Defines the depth of the trees.
depth_cbc = [int(x) for x in np.linspace(start = 1, stop = 9, num = 5)]

#iterations: The maximum number of trees that can be built.
# The final number of trees may be less than or equal to this number.
iterations_cbc = [int(x) for x in np.linspace(100, 1000, num = 10)]

#l2_leaf_reg_cbc
l2_leaf_reg_cbc = [int(x) for x in np.linspace(1, 9, num = 5)]

#learning_rate: Defines the learning rate.
#Used for reducing the gradient step.
learning_rate_cbc = [0.01, 0.03, 0.05, 0.1, 0.15]

# Criação e Fit do Random Grid

random_grid_cbc = { 'depth': depth_cbc,
                    'iterations': iterations_cbc,
                    'l2_leaf_reg': l2_leaf_reg_cbc,
                    'learning_rate': learning_rate_cbc }

classifier_cbc_rscv = CatBoostClassifier()

classifier_cbc_rscv_random = RandomizedSearchCV( estimator = classifier_cbc_rscv, 
                                                 param_distributions = random_grid_cbc, 
                                                 n_iter = 10, 
                                                 cv = 3, 
                                                 verbose = 2, 
                                                 random_state = 42,
                                                 scoring = make_scorer(matthews_corrcoef) )

classifier_cbc_rscv_random.fit(X, Y)

classifier_cbc_rscv_random.best_params_

classifier_cbc_rscv_random.best_score_

classifier_cbc_rscv_random.best_estimator_

classifier_cbc_rscv_random.best_index_

#Grid Search With Cross Validation

from sklearn.model_selection import GridSearchCV

from catboost import CatBoostClassifier

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

classifier_cbc_gscv = CatBoostClassifier()

classifier_cbc_gscv.get_params().keys()

grid_param_cbc = {
    'depth': [4, 7, 10],
    'iterations': [100, 200, 300, 400, 500],
    'l2_leaf_reg': [1,4,9],    
    'learning_rate': [0.03, 0.1, 0.15] 
}

classifier_cbc_gscv_gd_sr = GridSearchCV( estimator = classifier_cbc_gscv,  
                                          param_grid = grid_param_cbc,
                                          scoring = make_scorer(matthews_corrcoef),
                                          cv = 3 )

classifier_cbc_gscv_gd_sr.fit(X, Y) 

best_parameters_cbc = classifier_cbc_gscv_gd_sr.best_params_  

print(best_parameters_cbc)

best_result_cbc = classifier_cbc_gscv_gd_sr.best_score_  

print(best_result_cbc)

pd.set_option('max_columns',200)

gd_sr_results_cbc = pd.DataFrame(classifier_cbc_gscv_gd_sr.cv_results_)

#ETAPA 18: APLICAÇÃO DO PCA

#Bloco 01: Aplicação do PCA 

#Variance Caused by Each of the Principal Components

from sklearn.decomposition import PCA

pca = PCA()
#pca = PCA(n_components=0.99, whiten=True)

pca.get_params().keys()

import time
start = time.time()
 
X_train_pca_new = pca.fit_transform(X_train_pca)
X_test_pca_new = pca.transform(X_test_pca)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))
#Tempo de Execução: 0.8740171154340108 min

print('Original Number of Features:', X_train_pca.shape[1])
print('Reduced Number of Features:', X_train_pca_new.shape[1])

print('Original Number of Features:', X_test_pca.shape[1])
print('Reduced Number of Features:', X_test_pca_new.shape[1])

#Original Number of Features: 627
#Reduced Number of Features: 627

#Original Number of Features: 627
#Reduced Number of Features: 627

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(pca.get_params())

covariance_pca = pca.get_covariance()

explained_variance_pca = pca.explained_variance_ratio_ 

for i in explained_variance_pca:
    print(format(i*100, 'f'))

plt.figure(1, figsize = (14, 7))
plt.plot(explained_variance_pca, linewidth = 2)
plt.axis('tight')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')

plt.figure(1, figsize = (14, 7))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(explained_variance_pca, linewidth=5)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')

plt.figure(1, figsize = (14, 7))
plt.plot(np.cumsum(explained_variance_pca))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Name Trends PCA')

#1° Teste

pca_new = PCA(n_components = 400)

import time
start = time.time()

X_train_pca_new = pca_new.fit_transform(X_train_pca)
X_test_pca_new = pca_new.transform(X_test_pca)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))
#Tempo de Execução: 0.48103355169296264 min

print('Original Number of Features:', X_train_pca.shape[1])
print('Reduced Number of Features:', X_train_pca_new.shape[1])

print('Original Number of Features:', X_test_pca.shape[1])
print('Reduced Number of Features:', X_test_pca_new.shape[1])

Original Number of Features: 627
Reduced Number of Features: 400

Original Number of Features: 627
Reduced Number of Features: 400

Accuracy Score : 0.9447234193982049
Balanced Accuracy Score : 0.817998045842242
Precision Score : 0.831151832460733
Recall Score : 0.6532921810699589
F1 Score : 0.7315668202764977
Matthews Correlation Coefficient : 0.707567611944278
Cohen’s Kappa Score : 0.7012484914901898

#2° Teste

pca_new = PCA(n_components = 500)

import time
start = time.time()

X_train_pca_new = pca_new.fit_transform(X_train_pca)  
X_test_pca_new = pca_new.transform(X_test_pca)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))
#Tempo de Execução: 0.5051625728607178 min

print('Original Number of Features:', X_train_pca.shape[1])
print('Reduced Number of Features:', X_train_pca_new.shape[1])

print('Original Number of Features:', X_test_pca.shape[1])
print('Reduced Number of Features:', X_test_pca_new.shape[1])

Original Number of Features: 627
Reduced Number of Features: 500

Original Number of Features: 627
Reduced Number of Features: 500

Accuracy Score : 0.9445652603693013
Balanced Accuracy Score : 0.8192507529254891
Precision Score : 0.8271391529818496
Recall Score : 0.6563786008230452
F1 Score : 0.7319311663479924
Matthews Correlation Coefficient : 0.7073084854000639
Cohen’s Kappa Score : 0.7014735434350658

#3° Teste

pca_new = PCA(0.99)

import time
start = time.time()

X_train_pca_new = pca_new.fit_transform(X_train_pca)
X_test_pca_new = pca_new.transform(X_test_pca)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))
#Tempo de Execução: 0.19058539072672526 min

pca_new.n_components_

print('Original Number of Features:', X_train_pca.shape[1])
print('Reduced Number of Features:', X_train_pca_new.shape[1])

print('Original Number of Features:', X_test_pca.shape[1])
print('Reduced Number of Features:', X_test_pca_new.shape[1])

Accuracy Score : 0.9449606579415603
Balanced Accuracy Score : 0.8191759738219494
Precision Score : 0.831304347826087
Recall Score : 0.6556927297668038
F1 Score : 0.7331288343558281
Matthews Correlation Coefficient : 0.7090757301890313
Cohen’s Kappa Score : 0.7029214545678426

#Bloco 02: Parametrização do Modelo

from lightgbm import LGBMClassifier

#classifier_lgbm_pca = LGBMClassifier( max_depth = 1000, 
#                                      learning_rate = 0.1,
#                                      num_leaves = 2000,
#                                      min_data_in_leaf = 200,
#                                      n_estimators = 2000 )

classifier_lgbm_pca = LGBMClassifier( max_depth = 1880, 
                                      learning_rate = 0.1,
                                      num_leaves = 100,
                                      n_estimators = 4500,
                                      min_data_in_leaf = 140 )

#Bloco 03: Fit e Predição
					
import time
start = time.time()
			  
classifier_lgbm_pca.fit(X_train_pca_new, Y_train)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))
#Tempo de Execução: 19.891622447967528 min
#Tempo de Execução: 17.928581190109252 min
#Tempo de Execução: 14.361983533700307 min

Y_pred_lgbm_pca = classifier_lgbm_pca.predict(X_test_pca_new)
    
#Bloco 04: Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm_pca = accuracy_score(Y_test, Y_pred_lgbm_pca)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm_pca))

#Balanced Accuracy

mtrc_balanced_accuracy_score_lgbm_pca = balanced_accuracy_score(Y_test, Y_pred_lgbm_pca)
print('Balanced Accuracy Score : ' + str(mtrc_balanced_accuracy_score_lgbm_pca))

#Precision Score

mtrc_precision_score_lgbm_pca = precision_score(Y_test, Y_pred_lgbm_pca)
print('Precision Score : ' + str(mtrc_precision_score_lgbm_pca))

#Recall Score

mtrc_recall_score_lgbm_pca = recall_score(Y_test, Y_pred_lgbm_pca)
print('Recall Score : ' + str(mtrc_recall_score_lgbm_pca))

#F1 Score

mtrc_f1_score_lgbm_pca = f1_score(Y_test, Y_pred_lgbm_pca)
print('F1 Score : ' + str(mtrc_f1_score_lgbm_pca))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm_pca = matthews_corrcoef(Y_test, Y_pred_lgbm_pca)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm_pca))  

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm_pca = cohen_kappa_score(Y_test, Y_pred_lgbm_pca)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm_pca))

#ETAPA 19: APLICAÇÃO DO SPARCE PCA

from sklearn.decomposition import SparsePCA

#Bloco 01: Aplicação do SparcePCA

#1° Teste

spca = SparsePCA()
#spca = SparsePCA(n_components = 5, normalize_components = True, random_state = 0)

spca.get_params().keys()

import time
start = time.time()
 
X_train_spca_new = spca.fit_transform(X_train_spca)
X_test_spca_new = spca.transform(X_test_spca)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))

print('Original Number of Features:', X_train_spca.shape[1])
print('Reduced Number of Features:', X_train_spca_new.shape[1])

print('Original Number of Features:', X_test_spca.shape[1])
print('Reduced Number of Features:', X_test_spca_new.shape[1])

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(spca.get_params())

#2° Teste

import time
start = time.time()

spca_new = SparsePCA(n_components = 400)

X_train_spca_new = spca_new.fit_transform(X_train_spca)
X_test_spca_new = spca_new.transform(X_test_spca)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))

print('Original Number of Features:', X_train_spca.shape[1])
print('Reduced Number of Features:', X_train_spca_new.shape[1])

print('Original Number of Features:', X_test_spca.shape[1])
print('Reduced Number of Features:', X_test_spca_new.shape[1])

#3° Teste

spca_new = SparsePCA(n_components = 500)

import time
start = time.time()

X_train_spca_new = spca_new.fit_transform(X_train_spca)
X_test_spca_new = spca_new.transform(X_test_spca)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))

print('Original Number of Features:', X_train_spca.shape[1])
print('Reduced Number of Features:', X_train_spca_new.shape[1])

print('Original Number of Features:', X_test_spca.shape[1])
print('Reduced Number of Features:', X_test_spca_new.shape[1])

#Bloco 02: Parametrização do Modelo

from lightgbm import LGBMClassifier

#classifier_lgbm_spca = LGBMClassifier( max_depth = 1000, 
#                                       learning_rate = 0.1,
#                                       num_leaves = 2000,
#                                       min_data_in_leaf = 200,
#                                       n_estimators = 2000 )

classifier_lgbm_spca = LGBMClassifier( max_depth = 1880, 
                                       learning_rate = 0.1,
                                       num_leaves = 100,
                                       n_estimators = 4500,
                                       min_data_in_leaf = 140 )

#Bloco 03: Fit e Predição
							
import time
start = time.time()
	  
classifier_lgbm_spca.fit(X_train_spca_new, Y_train)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))

Y_pred_lgbm_spca = classifier_lgbm_spca.predict(X_test_spca_new)
    
#Bloco 04: Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm_spca = accuracy_score(Y_test, Y_pred_lgbm_spca)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm_spca))

#Balanced Accuracy

mtrc_balanced_accuracy_score_lgbm_spca = balanced_accuracy_score(Y_test, Y_pred_lgbm_spca)
print('Balanced Accuracy Score : ' + str(mtrc_balanced_accuracy_score_lgbm_spca))

#Precision Score

mtrc_precision_score_lgbm_spca = precision_score(Y_test, Y_pred_lgbm_spca)
print('Precision Score : ' + str(mtrc_precision_score_lgbm_spca))

#Recall Score

mtrc_recall_score_lgbm_spca = recall_score(Y_test, Y_pred_lgbm_spca)
print('Recall Score : ' + str(mtrc_recall_score_lgbm_spca))

#F1 Score

mtrc_f1_score_lgbm_spca = f1_score(Y_test, Y_pred_lgbm_spca)
print('F1 Score : ' + str(mtrc_f1_score_lgbm_spca))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm_spca = matthews_corrcoef(Y_test, Y_pred_lgbm_spca)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm_spca))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm_spca = cohen_kappa_score(Y_test, Y_pred_lgbm_spca)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm_spca))

#ETAPA 20: APLICAÇÃO DO INCREMENTAL PCA

#Bloco 01: Aplicação do IncrementalPCA 

#Variance Caused by Each of the Principal Components

from sklearn.decomposition import IncrementalPCA

ipca = IncrementalPCA()

ipca.get_params().keys()

import time
start = time.time()

X_train_ipca_new = ipca.fit_transform(X_train_ipca)
X_test_ipca_new = ipca.transform(X_test_ipca)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))
#Tempo de Execução: 0.4079192757606506 min

print('Original Number of Features:', X_train_ipca.shape[1])
print('Reduced Number of Features:', X_train_ipca_new.shape[1])

print('Original Number of Features:', X_test_ipca.shape[1])
print('Reduced Number of Features:', X_test_ipca_new.shape[1])

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(ipca.get_params())

covariance_ipca = ipca.get_covariance()

explained_variance_ipca = ipca.explained_variance_ratio_ 

for i in explained_variance_ipca:
    print(format(i*100, 'f'))

plt.figure(1, figsize = (14, 7))
plt.plot(explained_variance_ipca, linewidth = 2)
plt.axis('tight')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')

plt.figure(1, figsize = (14, 7))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(explained_variance_ipca, linewidth = 5)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')

plt.figure(1, figsize = (14, 7))
plt.plot(np.cumsum(explained_variance_ipca))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Name Trends IPCA')

#1° Teste

ipca_new = IncrementalPCA(n_components = 400)

import time
start = time.time()

X_train_ipca_new = ipca_new.fit_transform(X_train_ipca)
X_test_ipca_new = ipca_new.transform(X_test_ipca)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))
#Tempo de Execução: 0.2747618476549784 min

print('Original Number of Features:', X_train_ipca.shape[1])
print('Reduced Number of Features:', X_train_ipca_new.shape[1])

print('Original Number of Features:', X_test_ipca.shape[1])
print('Reduced Number of Features:', X_test_ipca_new.shape[1])

Original Number of Features: 627
Reduced Number of Features: 400

Original Number of Features: 627
Reduced Number of Features: 400

Accuracy Score : 0.945039737456012
Balanced Accuracy Score : 0.8205627591174871
Precision Score : 0.8294473229706391
Recall Score : 0.6587791495198903
F1 Score : 0.7343272171253822
Matthews Correlation Coefficient : 0.7099394460948084
Cohen’s Kappa Score : 0.7041256367009046

#2° Teste

ipca_new = IncrementalPCA(n_components = 500)

import time
start = time.time()

X_train_ipca_new = ipca_new.fit_transform(X_train_ipca)
X_test_ipca_new = ipca_new.transform(X_test_ipca)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))
#Tempo de Execução: 0.29941542943318683 min

print('Original Number of Features:', X_train_ipca.shape[1])
print('Reduced Number of Features:', X_train_ipca_new.shape[1])

print('Original Number of Features:', X_test_ipca.shape[1])
print('Reduced Number of Features:', X_test_ipca_new.shape[1])

Original Number of Features: 627
Reduced Number of Features: 500

Original Number of Features: 627
Reduced Number of Features: 500

Accuracy Score : 0.9443675615831719
Balanced Accuracy Score : 0.8186916568958779
Precision Score : 0.8261997405966277
Recall Score : 0.6553497942386831
F1 Score : 0.7309236947791163
Matthews Correlation Coefficient : 0.7062070400786051
Cohen’s Kappa Score : 0.700359808851374

#3° Teste

ipca_new = IncrementalPCA(n_components = 378)

import time
start = time.time()

X_train_ipca_new = ipca_new.fit_transform(X_train_ipca)
X_test_ipca_new = ipca_new.transform(X_test_ipca)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))
#Tempo de Execução: 0.22842663526535034 min

print('Original Number of Features:', X_train_ipca.shape[1])
print('Reduced Number of Features:', X_train_ipca_new.shape[1])

print('Original Number of Features:', X_test_ipca.shape[1])
print('Reduced Number of Features:', X_test_ipca_new.shape[1])

Original Number of Features: 627
Reduced Number of Features: 378

Original Number of Features: 627
Reduced Number of Features: 378

Accuracy Score : 0.9444466410976237
Balanced Accuracy Score : 0.8179907426565818
Precision Score : 0.828335506301608
Recall Score : 0.6536351165980796
F1 Score : 0.7306881349434542
Matthews Correlation Coefficient : 0.7063044776723942
Cohen’s Kappa Score : 0.7001962434268956

#Bloco 02: Parametrização do Modelo

from lightgbm import LGBMClassifier

#classifier_lgbm_ipca = LGBMClassifier( max_depth = 1000, 
#                                       learning_rate = 0.1,
#                                       num_leaves = 2000,
#                                       min_data_in_leaf = 200,
#                                       n_estimators = 2000 )

classifier_lgbm_ipca = LGBMClassifier( max_depth = 1880, 
                                       learning_rate = 0.1,
                                       num_leaves = 100,
                                       n_estimators = 4500,
                                       min_data_in_leaf = 140 )

#Bloco 03: Fit e Predição

import time
start = time.time()
								  
classifier_lgbm_ipca.fit(X_train_ipca_new, Y_train)

end = time.time()
print("Tempo de Execução: {} min".format((end - start)/60))
#Tempo de Execução: 16.1908074537913 min
#Tempo de Execução: 21.414798378944397 min
#Tempo de Execução: 14.769559462865194 min

Y_pred_lgbm_ipca = classifier_lgbm_ipca.predict(X_test_ipca_new)
    
#Bloco 04: Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm_ipca = accuracy_score(Y_test, Y_pred_lgbm_ipca)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm_ipca))

#Balanced Accuracy

mtrc_balanced_accuracy_score_lgbm_ipca = balanced_accuracy_score(Y_test, Y_pred_lgbm_ipca)
print('Balanced Accuracy Score : ' + str(mtrc_balanced_accuracy_score_lgbm_ipca))

#Precision Score

mtrc_precision_score_lgbm_ipca = precision_score(Y_test, Y_pred_lgbm_ipca)
print('Precision Score : ' + str(mtrc_precision_score_lgbm_ipca))

#Recall Score

mtrc_recall_score_lgbm_ipca = recall_score(Y_test, Y_pred_lgbm_ipca)
print('Recall Score : ' + str(mtrc_recall_score_lgbm_ipca))

#F1 Score

mtrc_f1_score_lgbm_ipca = f1_score(Y_test, Y_pred_lgbm_ipca)
print('F1 Score : ' + str(mtrc_f1_score_lgbm_ipca))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm_ipca = matthews_corrcoef(Y_test, Y_pred_lgbm_ipca)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm_ipca)) 

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm_ipca = cohen_kappa_score(Y_test, Y_pred_lgbm_ipca)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm_ipca)) 

#ETAPA XX: APLICAÇÃO DO ANOVA F-VALUE 

#ETAPA XX: APLICAÇÃO DO VARIANCETHRESHOLD

#ETAPA XX: REMOVING CORRELATED FEATURES USING CORR() METHOD

#ETAPA XX: APLICAÇÃO DO SMOTE

#ETAPA XX: APLICAÇÃO DO SMOTETomek