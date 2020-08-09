from sklearn.model_selection import StratifiedKFold

import numpy as np

from xgboost import XGBRegressor

from lightgbm import LGBMRegressor

from catboost import CatBoostRegressor

from sklearn.ensemble import GradientBoostingRegressor

from sklearn.linear_model import BayesianRidge

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
        #regressor = XGBRegressor()
        #regressor = LGBMRegressor()
        #regressor = CatBoostRegressor()
        #regressor = GradientBoostingRegressor()
        #regressor = BayesianRidge()
        regressor = SVR()
        regressor.fit(X_reg[indice_treinamento], Y_reg[indice_treinamento])
        previsoes = regressor.predict(X_reg[indice_teste])
        escore = r2_score(Y_reg[indice_teste], previsoes)
        resultados1.append(escore)
    resultados1 = np.asarray(resultados1)
    media = resultados1.mean()
    resultados30.append(media)
    
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

#XGB
#Tempo de Execução: 116.73 min
#LIGHT GBM
#Tempo de Execução: 49.57 min
#CATBOOST
#Tempo de Execução: 457.88 min
#GRADIENTBOOST
#Tempo de Execução: 224.92 min
#BAYESIANRIDGE
#Tempo de Execução: 15.59 min
#SVR
#Tempo de Execução: 403.97 min
    
resultados30 = np.asarray(resultados30)    
resultados30.mean()

#XGB
#0.19668740578676633
#LIGHT GBM
#0.19794776950387666
#CATBOOST
#0.20701557580268673
#GRADIENTBOOST
#0.1950725578064766
#BAYESIANRIDGE
#0.1731090775434205
#SVR
#0.15134841617026307

for i in range(resultados30.size):
    print(str(resultados30[i]).replace('.', ','))