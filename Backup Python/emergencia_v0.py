 #Importação do Pandas
import pandas as pd

#Leitura da Base de Dados
df = pd.read_excel('BASE_DADOS_EMERGENCIA_16_A_19.xlsx')

#Nome Das Colunas da Base de Dados
df.columns.values

#Visualização das n Primeiras Linhas
df.head(10)

#Estatísticas Descritivas da Base de Dados
df.info()

df.describe()

#Valores Nulos ou Não Existentes

df.isnull().sum()

df.isna().sum()

#Preencher Valores Nulos
df['COD_CID_ADM'].fillna('SEM REGISTRO', inplace = True)

df['COD_CID_ADM_PRINCIPAL'].fillna('SEM REGISTRO', inplace = True)

#Exclusão das Colunas Que Não Serão Utilizadas

df_trat = df.drop('DESC_DIA_DA_SEMANA', axis = 1)

df_trat = df_trat.drop('REGISTRO', axis = 1)

df_trat = df_trat.drop('NOME_PACIENTE', axis = 1)

df_trat = df_trat.drop('DESC_SEXO', axis = 1)

df_trat = df_trat.drop('DESC_FAIXA_ETARIA', axis = 1)

df_trat = df_trat.drop('NUM_TRATAMENTO', axis = 1)

df_trat = df_trat.drop('DATA_ADMISSAO', axis = 1)

df_trat = df_trat.drop('DESC_DATA_ADMISSAO', axis = 1)

df_trat = df_trat.drop('DESC_DATA_ALTA', axis = 1)

df_trat = df_trat.drop('DESC_CONVENIO', axis = 1)

df_trat = df_trat.drop('COD_EMPRESA', axis = 1)

df_trat = df_trat.drop('DESC_EMPRESA', axis = 1)

df_trat = df_trat.drop('COD_PLANTONISTA', axis = 1)

df_trat = df_trat.drop('NOME_PLANTONISTA', axis = 1)

df_trat = df_trat.drop('COD_CID_ADM', axis = 1)

df_trat = df_trat.drop('DESC_STATUS_INT', axis = 1)

df_trat = df_trat.drop('NUM_INT_POSTERIOR', axis = 1)

df_trat = df_trat.drop('TEMPO_ATENDIMENTO', axis = 1)

df_trat = df_trat.drop('CONTADOR', axis = 1)

#Criação das Bases X e Y

X = df_trat

X.info()

X.describe()

Y = df_trat['CONT_STATUS_INT']

X = X.drop('CONT_STATUS_INT', axis = 1)

#Aplicação do One Hot Encoding

X = pd.get_dummies(X, columns=['APEL_PLANTONISTA'], prefix=['PLANTONISTA'])

X = pd.get_dummies(X, columns=['COD_CID_ADM_PRINCIPAL'], prefix=['COD_CID'])

X = pd.get_dummies(X, columns=['COD_CLAC_RISCO'], prefix=['CLAC_RISCO'])

X = pd.get_dummies(X, columns=['CID_PREVALENTE'], prefix=['CID_PREVALENTE'])

#Transformação da Base Y em Array

import numpy as np

Y = np.array(Y)

#Transformação da Base X em Array

X_list = list(X.columns)

X = np.array(X)

#Criação da Base de Teste e Treinamento

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

print('Training X Data Dimensions:', X_train.shape)
print('Training Y Data Dimensions:', Y_train.shape)
print('Testing X Data Dimensions:', X_test.shape)
print('Testing Y Data Dimensions:', Y_test.shape)

#RANDOM FOREST

#Criação do Modelo
from sklearn.ensemble import RandomForestClassifier
classifier_rf = RandomForestClassifier(n_estimators = 40, criterion = 'entropy', random_state = 42)
classifier_rf.fit(X_train, Y_train)
Y_pred = classifier_rf.predict(X_test)

#Análise de Métricas
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred)))
print('Precision Score : ' + str(precision_score(Y_test, Y_pred)))
print('Recall Score : ' + str(recall_score(Y_test, Y_pred)))
print('F1 Score : ' + str(f1_score(Y_test, Y_pred)))

from sklearn.metrics import classification_report  
print(classification_report(Y_test,Y_pred))  

#Matriz de Confusão
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred)))

#Análise da Importância das Features (Variáveis)
importances = list(classifier_rf.feature_importances_)

feature_importances = [(feature, round(importance, 6)) for feature, importance in zip(X_list, importances)]

feature_importances = sorted(feature_importances, key = lambda x: x[1], reverse = True)

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances];

#TESTE DE VÁRIOS MODELOS

from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.neural_network import MLPClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from catboost import CatBoostClassifier

from sklearn.metrics import accuracy_score
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import KFold

models = []
models.append(('KNN', KNeighborsClassifier()))
models.append(('SVC', SVC()))
models.append(('LR', LogisticRegression()))
models.append(('MLP', MLPClassifier()))
models.append(('GNB', GaussianNB()))
models.append(('RF', RandomForestClassifier()))
models.append(('GB', GradientBoostingClassifier()))
models.append(('XGB', XGBClassifier()))
models.append(('LGBM', LGBMClassifier()))
models.append(('CB', CatBoostClassifier()))

#Treino e Teste dos Modelos
names = []
scores = []
for name, model in models:
    model.fit(X_train, Y_train)
    Y_pred = model.predict(X_test)
    scores.append(accuracy_score(Y_test, Y_pred))
    names.append(name)
tr_split = pd.DataFrame({'Name': names, 'Score': scores})
print(tr_split)

#Plotagem das Acurácias
import seaborn as sns
import matplotlib.pyplot as plt

axis = sns.barplot(x = 'Name', y = 'Score', data = tr_split)
axis.set(xlabel='Classifier', ylabel='Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha="center") 
    
plt.show()

#Treino e Teste dos Modelos com KFold
names = []
scores = []
for name, model in models:
    
    kfold = KFold(n_splits = 10, random_state = 10) 
    score = cross_val_score(model, X, Y, cv = kfold, scoring = 'accuracy').mean()   
    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

#Plotagem das Acurácias com KFold
import seaborn as sns
import matplotlib.pyplot as plt

axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel = 'Classifier', ylabel = 'Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha = "center") 
    
plt.show()







