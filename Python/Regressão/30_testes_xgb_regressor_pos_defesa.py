from sklearn.model_selection import StratifiedKFold

import numpy as np

from xgboost import XGBRegressor

from sklearn.metrics import r2_score

folds_reg = 10

import time
start = time.time()

resultados30 = []
for i in range(21,51):
    print(str(i))
    k_fold_reg  = StratifiedKFold(n_splits = folds_reg, shuffle = True, random_state = i)
    resultados1 = []
    for indice_treinamento, indice_teste in k_fold_reg.split(X_reg, np.zeros(shape=(Y_reg.shape[0], 1))):
        #XGB 06
        #regressor = XGBRegressor( n_estimators = 500, min_child_weight = 70, max_depth = 45, learning_rate = 0.01, n_jobs = 5 )
        #XGB 07
        #regressor = XGBRegressor( n_estimators = 400, min_child_weight = 60, max_depth = 50, learning_rate = 0.01, n_jobs = 5 )
        #XGB 08
        regressor = XGBRegressor( n_estimators = 4000, min_child_weight = 40, max_depth = 450, learning_rate = 0.001, n_jobs = 6 )        
        #XGB 09
        #regressor = XGBRegressor( n_estimators = 500, min_child_weight = 50, max_depth = 60, learning_rate = 0.01, n_jobs = 6 )
        #XGB 10
        #regressor = XGBRegressor( n_estimators = 400, min_child_weight = 50, max_depth = 500, learning_rate = 0.01, n_jobs = 5 )
        #XGB 11
        #regressor = XGBRegressor( n_estimators = 550, min_child_weight = 30, max_depth = 375, learning_rate = 0.01, n_jobs = 4 )
        regressor.fit(X_reg[indice_treinamento], Y_reg[indice_treinamento])
        previsoes = regressor.predict(X_reg[indice_teste])
        escore = coefficient_of_determination_by_correlation(Y_reg[indice_teste], previsoes)
        resultados1.append(escore)
    resultados1 = np.asarray(resultados1)
    media = resultados1.mean()
    resultados30.append(media)
    
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

resultados30 = np.asarray(resultados30)    
resultados30.mean()

for i in range(resultados30.size):
    print(str(resultados30[i]).replace('.', ','))

#XGB 06
#Tempo de Execução: 1027.61 min
0.21493024411126707

#XGB 07
#Tempo de Execução: 815.73 min
0.21342002818041855

#XGB 08
#Tempo de Execução: 3001.70 min
0.2128530269188035

#XGB 09
#Tempo de Execução: 1112.34 min
0.2142215083803626

#XGB 10
#Tempo de Execução: 862.79 min
0.2133033214598154

#XGB 11
#Tempo de Execução: 1417.13 min
0.21128742322072183


























