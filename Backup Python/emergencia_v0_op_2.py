#ETAPA 01: TRATAMENTO DOS DADOS

#Importação do Pandas
import pandas as pd

#Leitura da Base de Dados
df = pd.read_excel('BASE_DADOS_EMERGENCIA_16_A_19.xlsx')

#Checa os tipos:
df.dtypes

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

#X = pd.get_dummies(X, columns=['APEL_PLANTONISTA'], prefix=['PLANTONISTA'])

#X = pd.get_dummies(X, columns=['COD_CID_ADM_PRINCIPAL'], prefix=['COD_CID'])

#X = pd.get_dummies(X, columns=['COD_CLAC_RISCO'], prefix=['CLAC_RISCO'])

#X = pd.get_dummies(X, columns=['CID_PREVALENTE'], prefix=['CID_PREVALENTE'])

#Aplicação do Label Encoder na Base X

from sklearn.preprocessing import LabelEncoder

lbl_enc_X = LabelEncoder()

X['APEL_PLANTONISTA'] = lbl_enc_X.fit_transform(X['APEL_PLANTONISTA'])

X['COD_CID_ADM_PRINCIPAL'] = lbl_enc_X.fit_transform(X['COD_CID_ADM_PRINCIPAL'])

X['COD_CLAC_RISCO'] = lbl_enc_X.fit_transform(X['COD_CLAC_RISCO'])

X['CID_PREVALENTE'] = lbl_enc_X.fit_transform(X['CID_PREVALENTE'])

#Transformação da Base Y em Array

import numpy as np

Y = np.array(Y)

#Transformação da Base X em Array

X_list = list(X.columns)

X = np.array(X)

#Escalonamento da Base X

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X = scaler.fit_transform(X)

#from sklearn.preprocessing import MinMaxScaler
#scaler = MinMaxScaler()
#X = scaler.fit_transform(X)

#from sklearn.preprocessing import Normalizer
#scaler = Normalizer()
#X = scaler.fit_transform(X)

#ETAPA 02: DIVISÃO DA BASE DE DADOS EM TESTE E TREINAMENTO

#Criação da Base de Teste e Treinamento

from sklearn.model_selection import train_test_split

X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size = 0.3, random_state = 42)

print('Training X Data Dimensions:', X_train.shape)
print('Training Y Data Dimensions:', Y_train.shape)
print('Testing X Data Dimensions:', X_test.shape)
print('Testing Y Data Dimensions:', Y_test.shape)

#ETAPA 03: TESTE DE VÁRIOS MODELOS

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

#Treino e Teste dos Modelos com 10 KFold
names = []
scores = []
for name, model in models:
    
    kfold = KFold(n_splits = 10, random_state = 10) 
    score = cross_val_score(model, X, Y, cv = kfold, scoring = 'accuracy').mean()   
    names.append(name)
    scores.append(score)
kf_cross_val = pd.DataFrame({'Name': names, 'Score': scores})
print(kf_cross_val)

#Plotagem das Acurácias com 10 KFold
import seaborn as sns
import matplotlib.pyplot as plt

axis = sns.barplot(x = 'Name', y = 'Score', data = kf_cross_val)
axis.set(xlabel = 'Classifier', ylabel = 'Accuracy')
for p in axis.patches:
    height = p.get_height()
    axis.text(p.get_x() + p.get_width()/2, height + 0.005, '{:1.4f}'.format(height), ha = "center") 
    
plt.show()

#ETAPA 04: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO KNN

from sklearn.neighbors import KNeighborsClassifier
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

from sklearn.metrics import classification_report  
print(classification_report(Y_test,Y_pred_knc))  

#Matriz de Confusão
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_knc)))

#ETAPA 05: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO SVC

#Criação do Modelo
from sklearn.svm import SVC
classifier_svc = SVC( kernel = 'linear', 
                      random_state = 1 )
classifier_svc.fit(X_train, Y_train)
Y_pred_svc = classifier_svc.predict(X_test)

#Análise de Métricas
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_svc)))
print('Precision Score : ' + str(precision_score(Y_test, Y_pred_svc)))
print('Recall Score : ' + str(recall_score(Y_test, Y_pred_svc)))
print('F1 Score : ' + str(f1_score(Y_test, Y_pred_svc)))

from sklearn.metrics import classification_report  
print(classification_report(Y_test,Y_pred_svc))  

#Matriz de Confusão
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_svc)))

#ETAPA 06: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO REGRESSÃO LOGÍSTICA

#Criação do Modelo
from sklearn.linear_model import LogisticRegression
classifier_lr = LogisticRegression( random_state = 1 )
classifier_lr.fit(X_train, Y_train)
Y_pred_lr = classifier_lr.predict(X_test)

#Análise de Métricas
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_lr)))
print('Precision Score : ' + str(precision_score(Y_test, Y_pred_lr)))
print('Recall Score : ' + str(recall_score(Y_test, Y_pred_lr)))
print('F1 Score : ' + str(f1_score(Y_test, Y_pred_lr)))

from sklearn.metrics import classification_report  
print(classification_report(Y_test,Y_pred_lr))  

#Matriz de Confusão
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_lr)))

#ETAPA 07: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO MLP

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

from sklearn.metrics import classification_report  
print(classification_report(Y_test,Y_pred_mlp))  

#Matriz de Confusão
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_mlp)))

#ETAPA 08: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO NAIVE BAYES

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

from sklearn.metrics import classification_report  
print(classification_report(Y_test,Y_pred_gnb))

#Matriz de Confusão
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_gnb)))

#ETAPA 09: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO RANDOM FOREST

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
                                                scoring = 'f1',
                                                n_jobs = -1 )

classifier_rf_rscv_random.fit(X, Y)

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

#ETAPA 10: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO GRADIENT BOOSTING

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

#ETAPA 11: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO XGB

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

#ETAPA 12: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO LGBM

#Criação do Modelo
from lightgbm import LGBMClassifier

classifier_lgbm = LGBMClassifier( max_depth = 50, 
                                  learning_rate = 0.1,
                                  num_leaves = 900,
                                  n_estimators = 300 )

classifier_lgbm.fit(X_train, Y_train)

Y_pred_lgbm = classifier_lgbm.predict(X_test)

#Análise de Métricas
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_lgbm)))
print('Precision Score : ' + str(precision_score(Y_test, Y_pred_lgbm)))
print('Recall Score : ' + str(recall_score(Y_test, Y_pred_lgbm)))
print('F1 Score : ' + str(f1_score(Y_test, Y_pred_lgbm)))

from sklearn.metrics import classification_report  
print(classification_report(Y_test,Y_pred_lgbm))  

#Matriz de Confusão
from sklearn.metrics import confusion_matrix
print('Confusion Matrix : \n' + str(confusion_matrix(Y_test, Y_pred_lgbm)))

#Análise da Importância das Features (Variáveis)
importances_lgbm = list(classifier_lgbm.feature_importances_)

feature_importances_lgbm = [(feature, round(importance, 6)) for feature, importance in zip(X_list, importances_lgbm)]

feature_importances_lgbm = sorted(feature_importances_lgbm, key = lambda x: x[1], reverse = True)

[print('Variable: {:20} Importance: {}'.format(*pair)) for pair in feature_importances_lgbm];

#ETAPA 12: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO CATBOOST

#Criação do Modelo
from catboost import CatBoostClassifier

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 500, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.15,
                                     one_hot_max_size = 50 )

classifier_cbc.fit(X_train, Y_train)

Y_pred_cbc = classifier_cbc.predict(X_test)

#Análise de Métricas
from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score
print('Accuracy Score : ' + str(accuracy_score(Y_test, Y_pred_cbc)))
print('Precision Score : ' + str(precision_score(Y_test, Y_pred_cbc)))
print('Recall Score : ' + str(recall_score(Y_test, Y_pred_cbc)))
print('F1 Score : ' + str(f1_score(Y_test, Y_pred_cbc)))

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

#Grid Search With Cross Validation

from catboost import CatBoostClassifier

from sklearn.model_selection import GridSearchCV

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
                                          scoring = 'accuracy',
                                          cv = 10 )

classifier_cbc_gscv_gd_sr.fit(X_train, Y_train) 

best_parameters_cbc = classifier_cbc_gscv_gd_sr.best_params_  
print(best_parameters_cbc)

best_result_cbc = classifier_cbc_gscv_gd_sr.best_score_  
print(best_result_cbc)

pd.set_option('max_columns',200)
gd_sr_results_cbc = pd.DataFrame(classifier_cbc_gscv_gd_sr.cv_results_)