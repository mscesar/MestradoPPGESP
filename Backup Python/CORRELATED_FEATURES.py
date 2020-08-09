#ETAPA XX: REMOVING CORRELATED FEATURES USING CORR() METHOD

import numpy as np
X = np.load('X.npy')

correlated_features_X = set()  
correlation_matrix_X = X.corr() 

for i in range(len(correlation_matrix_X .columns)):  
    for j in range(i):
        if abs(correlation_matrix_X.iloc[i, j]) > 0.95:
            colname = correlation_matrix_X.columns[i]
            correlated_features_X.add(colname)
            
len(correlated_features_X)         

print(correlated_features_X) 

X_corr = X.copy()

X_corr.drop(labels = correlated_features_X, axis = 1, inplace = True)

#Transformação da Base Y em Array

import numpy as np

Y = np.array(Y)

#Transformação da Base X_corr em Array

X_corr_list = list(X_corr.columns)

X_corr = np.array(X_corr)

#Escalonamento da Base X_corr - Opção 01 - StandardScaler

from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()

X_corr = scaler.fit_transform(X_corr)

#Criação da Base de Teste e Treinamento - Com Balanceamento em Y

from sklearn.model_selection import train_test_split

X_corr_train, X_corr_test, Y_corr_train, Y_corr_test = train_test_split(X_corr, Y, test_size = 0.3, random_state = 42, stratify = Y)

print('Training X_corr_train Data Dimensions:', X_corr_train.shape)
print('Training Y_corr_train Data Dimensions:', Y_corr_train.shape)
print('Testing X_corr_test Data Dimensions:', X_corr_test.shape)
print('Testing Y_corr_test Data Dimensions:', Y_corr_test.shape)

#Training X_corr_train Data Dimensions: (59012, 334)
#Training Y_corr_train Data Dimensions: (59012,)
#Testing X_corr_test Data Dimensions: (25291, 334)
#Testing Y_corr_test Data Dimensions: (25291,)

print('Labels counts in Y_corr:', np.bincount(Y))
#Labels counts in Y_corr: [74584  9719]
print('Labels counts in Y_corr_train:', np.bincount(Y_corr_train))
#Labels counts in Y_corr_train: [52209  6803]
print('Labels counts in Y_corr_test:', np.bincount(Y_corr_test))
#Labels counts in Y_corr_test: [22375  2916]

#ETAPA 02: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO LIGHT GBM

#Parametrização do Modelo

from lightgbm import LGBMClassifier

classifier_lgbm_corr = LGBMClassifier( max_depth = 500, 
                                       learning_rate = 0.01,
                                       num_leaves = 1000,
                                       min_data_in_leaf = 200,
                                       n_estimators = 2000,
                                       objective = 'binary',
                                       metric = 'binary_logloss',
                                       random_state = 42 )

#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(classifier_lgbm_corr.get_params())

# Fit e Predição

import time
start = time.time()

classifier_lgbm_corr.fit(X_corr_train, Y_corr_train)

end = time.time()
print("Tempo de Execução: {} sec".format(end - start))

Y_pred_lgbm_corr = classifier_lgbm_corr.predict(X_corr_test)

#Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm_corr = accuracy_score(Y_corr_test, Y_pred_lgbm_corr)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm_corr))

#Precision Score

mtrc_precision_score_lgbm_corr = precision_score(Y_corr_test, Y_pred_lgbm_corr)
print('Precision Score : ' + str(mtrc_precision_score_lgbm_corr))

#Recall Score

mtrc_recall_score_lgbm_corr = recall_score(Y_corr_test, Y_pred_lgbm_corr)
print('Recall Score : ' + str(mtrc_recall_score_lgbm_corr))

#F1 Score

mtrc_f1_score_lgbm_corr = f1_score(Y_corr_test, Y_pred_lgbm_corr)
print('F1 Score : ' + str(mtrc_f1_score_lgbm_corr))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm_corr = matthews_corrcoef(Y_corr_test, Y_pred_lgbm_corr)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm_corr))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm_corr = cohen_kappa_score(Y_corr_test, Y_pred_lgbm_corr)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm_corr))

#Accuracy Score : 0.9513265588549287
#Precision Score : 0.8574459058124735
#Recall Score : 0.6930727023319616
#F1 Score : 0.7665465579366584
#Matthews Correlation Coefficient : 0.7449396884850108
#Cohen’s Kappa Score : 0.7397179910056747