#Importação do Pandas
import pandas as pd

#Leitura da Base de Dados
df = pd.read_excel('BaseMCV.xls')

#Nome Das Colunas da Base de Dados
df.columns.values

#Visualização das n Primeiras Linhas
df.head(10)

#Estatísticas Descritivas da Base de Dados
df.info()

df.describe()

#Conversão da Coluna Admissão Emergência em DateTime
df['Admissão Emergência'] = pd.to_datetime(df['Admissão Emergência'])

# Coluna 'Data'
#df['Data Admissão Emergência'] = df['Admissão Emergência'].dt.strftime('%Y-%m-%d')

# Coluna 'Hora'
#df['Hora Admissão Emergência'] = df['Admissão Emergência'].dt.strftime('%H:%M:%S')

#Extração do Ano a Partir da Coluna Admissão Emergência - Coluna Ano
df['Ano Admissão Emergência'] = df['Admissão Emergência'].dt.strftime('%Y')

df['Ano Admissão Emergência'] = pd.to_numeric(df['Ano Admissão Emergência'])

#Extração do Mês a Partir da Coluna Admissão Emergência - Coluna Mês
df['Mês Admissão Emergência'] = df['Admissão Emergência'].dt.strftime('%m')

df['Mês Admissão Emergência'] = pd.to_numeric(df['Mês Admissão Emergência'])

#Valores Nulos ou Não Existentes

df.isnull().sum()
df.isna().sum()

#Seleção das Colunas de Interesse
features = ['CID Adm. Emergência', 'Plantonista', 'Ano Admissão Emergência', 'Mês Admissão Emergência', 'Dia da Semana', 'Número da Semana', 
            'Grupo DT', 'IAM C/SST', 'Diagnóstico Final']

X = df[features]

X.info()

X.describe()

Y = df['Admissão Hospitalar']

#Aplicação do One Hot Encoding

X = pd.get_dummies(X, columns=['CID Adm. Emergência'], prefix=['CID_ADM'])

X = pd.get_dummies(X, columns=['Plantonista'], prefix=['PLANTONISTA'])

X = pd.get_dummies(X, columns=['Dia da Semana'], prefix=['DIA_DA_SEMANA'])

X = pd.get_dummies(X, columns=['Número da Semana'], prefix=['NUM_SEMANA'])

X = pd.get_dummies(X, columns=['Grupo DT'], prefix=['GRUPO_DT'])

X = pd.get_dummies(X, columns=['IAM C/SST'], prefix=['IAM_C_SST'])

X = pd.get_dummies(X, columns=['Diagnóstico Final'], prefix=['DIAG_FINAL'])

#Aplicação do Label Encoder na Base Y

from sklearn.preprocessing import LabelEncoder
import numpy as np

Y = np.array(Y)

lbl_Y = LabelEncoder()

Y = lbl_Y.fit_transform(Y)

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

#OTIMIZAÇÃO DE HIPERPARAMETROS - RANDOM FOREST

#Random Search With Cross Validation
import numpy as np
from sklearn.model_selection import RandomizedSearchCV

# Número de Árvores
n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)]
# Número de Variáveis a Serem Consideradas em Cada Split
max_features = ['auto', 'sqrt']
# Número Máximo de Níveis na Árvore
max_depth = [int(x) for x in np.linspace(10, 100, num = 10)]
max_depth.append(None)
# Número Mínimo de Amostras Necessárias para Dividir um Nó 
min_samples_split = [2, 5, 10]
# Número Mínimo de Amostras Requeridas em Cada Nó Folha
min_samples_leaf = [1, 2, 4]
# Método de Seleção de Amostras para Treinar cada Árvore
bootstrap = [True, False]

# Criação do Random Grid
random_grid = {'n_estimators': n_estimators,
               'max_features': max_features,
               'max_depth': max_depth,
               'min_samples_split': min_samples_split,
               'min_samples_leaf': min_samples_leaf,
               'bootstrap': bootstrap}

classifier_rf_rscv = RandomForestClassifier()

classifier_rf_rscv_random = RandomizedSearchCV( estimator = classifier_rf_rscv, 
                                                param_distributions = random_grid, 
                                                n_iter = 100, 
                                                cv = 3, 
                                                verbose = 2, 
                                                random_state = 42, 
                                                n_jobs = -1 )

classifier_rf_rscv_random.fit(X_train, Y_train)

classifier_rf_rscv_random.best_params_

classifier_rf_rscv_random.best_score_

classifier_rf_rscv_random.best_estimator_

classifier_rf_rscv_random.best_index_

#Grid Search With Cross Validation
from sklearn.model_selection import GridSearchCV

classifier_rf_gscv = RandomForestClassifier()

classifier_rf_gscv.get_params().keys()

grid_param = {  
    'n_estimators': [100, 300, 500, 800, 1000],
    'criterion': ['gini', 'entropy'],
    'bootstrap': [True, False]
}

grid_param = {
    'max_depth': range(1,25),
    'criterion': ['gini', 'entropy'],
    'max_leaf_nodes': [None, 2, 3, 5, 7, 9, 12, 15]
}

grid_param = {
    'bootstrap': [True],
    'max_depth': [80, 90, 100, 110],
    'max_features': [2, 3],
    'min_samples_leaf': [3, 4, 5],
    'min_samples_split': [8, 10, 12],
    'n_estimators': [100, 200, 300, 1000]
}

grid_param = {
    'criterion': ['entropy', 'gini'],
    'n_estimators': [25, 50, 75],
    'bootstrap': [False, True],
    'max_depth': [3, 5, 10],
    'max_features': ['auto', 0.1, 0.2, 0.3]
}

classifier_rf_gscv_gd_sr = GridSearchCV( estimator = classifier_rf_gscv,  
                                         param_grid = grid_param,
                                         scoring = 'accuracy',
                                         cv = 3,
                                         return_train_score = True,
                                          n_jobs = -1 )

classifier_rf_gscv_gd_sr.fit(X_train, Y_train) 

best_parameters = classifier_rf_gscv_gd_sr.best_params_  
print(best_parameters)

best_result = classifier_rf_gscv_gd_sr.best_score_  
print(best_result)

pd.set_option('max_columns',200)
gd_sr_results = pd.DataFrame(classifier_rf_gscv_gd_sr.cv_results_)


