from sklearn.model_selection import StratifiedKFold

import numpy as np

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import BayesianRidge

from sklearn.ensemble import RandomForestRegressor

from sklearn.svm import SVR

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
        #regressor = XGBRegressor(n_jobs = 3)
        #regressor = RandomForestRegressor(n_jobs = 3)
        #regressor = LGBMRegressor()
        #regressor = CatBoostRegressor()
        #regressor = GradientBoostingRegressor()
        #regressor = BayesianRidge()
        #regressor = SVR()
        #regressor = ExtraTreesRegressor(n_jobs = 3)
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

#CATBOOST
#Tempo de Execução: 314.67 min
0.21021578584660314
#BAYESIANRIDGE
#Tempo de Execução: 5.32 min
0.17515191720049128
#LIGHTGBM
#Tempo de Execução: 41.80 min
0.20326982152763923
#XGBOOST
#Tempo de Execução: 56.64 min
0.19958733335515527
#RANDOMFOREST
#Tempo de Execução: 285.39 min
0.1973231675045023
#SVR
#Tempo de Execução: 313.58 min
0.18540360531390226
#GRADIENTBOOST
#Tempo de Execução: 212.97 min
0.19816409806306268
#EXTRATREES
#Tempo de Execução: 230.63 min
0.18839198876901048
