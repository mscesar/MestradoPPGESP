#ETAPA 03: TESTE DE VÁRIOS MODELOS

from sklearn.linear_model import LinearRegression
from sklearn.linear_model import Lasso
from sklearn.linear_model import LassoLarsIC
from sklearn.linear_model import Ridge
from sklearn.kernel_ridge import KernelRidge
from sklearn.linear_model import BayesianRidge
from sklearn.linear_model import ElasticNet
from sklearn.neighbors import KNeighborsRegressor
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import ExtraTreesRegressor
from pyearth import Earth
from rgf.sklearn import RGFRegressor
from rgf.sklearn import FastRGFRegressor
from sklearn.ensemble import RandomForestRegressor
from sklearn.ensemble import AdaBoostRegressor
from sklearn.ensemble import GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor
from sklearn.neural_network import MLPRegressor

folds_reg   = 10

seed_reg = 42

# Criação do Dicionário de Modelos

models_reg = {}
models_reg["LinearR"]          = LinearRegression()
models_reg["Lasso"]            = Lasso(random_state = seed_reg)
models_reg["LassoLarsIC"]      = LassoLarsIC()
models_reg["Ridge"]            = Ridge(random_state = seed_reg)
models_reg["KernelRidge"]      = KernelRidge()
models_reg["BayesianRidge"]    = BayesianRidge()
models_reg["ElasticNet"]       = ElasticNet(random_state = seed_reg)
models_reg["KNN"]              = KNeighborsRegressor()
models_reg["SVR"]              = SVR()
models_reg["DecisionTree"]     = DecisionTreeRegressor(random_state = seed_reg)
models_reg["ExtraTrees"]       = ExtraTreesRegressor(random_state = seed_reg)
models_reg["Earth"]            = Earth()
models_reg["RGFRegressor"]     = RGFRegressor()
#models_reg["FastRGFRegressor"] = FastRGFRegressor()
models_reg["RandomForest"]     = RandomForestRegressor(random_state = seed_reg)
models_reg["AdaBoost"]         = AdaBoostRegressor(random_state = seed_reg)
models_reg["GradientBoost"]    = GradientBoostingRegressor(random_state = seed_reg)
models_reg["XGBoost"]          = XGBRegressor(random_state = seed_reg)
models_reg["LightGBM"]         = LGBMRegressor(random_state = seed_reg)
models_reg["CatBoost"]         = CatBoostRegressor(random_state = seed_reg)
models_reg["MLPRegressor"]     = MLPRegressor(random_state = seed_reg)

#TESTE03-1: Treino e Teste dos Modelos com CrossValScore + 10 KFold(shuffle = False) + R2 Score - X_reg, Y_reg

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

#from sklearn.metrics import r2_score

folds_reg   = 10

metric_reg  = custom_r2_by_correlation

import time

model_results_reg = []
model_std_reg = []
model_names_reg   = []

start = time.time()
for model_name_reg in models_reg:
	print(model_name_reg)    
	model_reg   = models_reg[model_name_reg]
	k_fold_reg  = KFold(n_splits = folds_reg, random_state = seed_reg, shuffle = False)
	results_reg = cross_val_score(model_reg, X_reg, Y_reg, cv = k_fold_reg, scoring = metric_reg, n_jobs = 6)
    
	score_reg = round(results_reg.mean(),10)
	std_reg = round(results_reg.std(),10)
	model_results_reg.append(score_reg)
	model_std_reg.append(std_reg)
	model_names_reg.append(model_name_reg)
	#print("{}: {}, {}".format(model_name_reg, score_reg, std_reg))
end = time.time()
kf_cross_val_reg = pd.DataFrame({'Name': model_names_reg, 'R² Score': model_results_reg, 'Desvio Padrão': model_std_reg})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_val_reg)
#

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_regressor = pd.ExcelWriter('V5/cross_val_score_regressor_r2_score_X_reg_Y_reg_kfold.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val_reg.to_excel(cross_val_score_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_regressor.save()

#TESTE04-1: Treino e Teste dos Modelos com CrossValScore + 10 KFold(shuffle = True) + R2 Score - X_reg, Y_reg

from sklearn.model_selection import KFold

from sklearn.model_selection import cross_val_score

#from sklearn.metrics import r2_score

folds_reg   = 10

metric_reg  = custom_r2_by_correlation

import time

model_results_reg = []
model_std_reg = []
model_names_reg   = []

start = time.time()
for model_name_reg in models_reg:
	print(model_name_reg)    
	model_reg   = models_reg[model_name_reg]
	k_fold_reg  = KFold(n_splits = folds_reg, random_state = seed_reg, shuffle = True)
	results_reg = cross_val_score(model_reg, X_reg, Y_reg, cv = k_fold_reg, scoring = metric_reg)
    
	score_reg = round(results_reg.mean(),10)
	std_reg = round(results_reg.std(),10)
	model_results_reg.append(score_reg)
	model_std_reg.append(std_reg)
	model_names_reg.append(model_name_reg)
	#print("{}: {}, {}".format(model_name_reg, score_reg, std_reg))
end = time.time()
kf_cross_val_reg = pd.DataFrame({'Name': model_names_reg, 'R² Score': model_results_reg, 'Desvio Padrão': model_std_reg})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_val_reg)
#

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_regressor = pd.ExcelWriter('V5/cross_val_score_regressor_r2_score_X_reg_Y_reg_kfold_shuffle.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val_reg.to_excel(cross_val_score_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_regressor.save()

#TESTE06-1: Treino e Teste dos Modelos com CrossValScore + ShuffleSplit(n_splits = 10) + R2 Score - X_reg, Y_reg

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_val_score

#from sklearn.metrics import r2_score

folds_reg   = 10

metric_reg  = custom_r2_by_correlation

import time

model_results_reg = []
model_std_reg = []
model_names_reg   = []

start = time.time()
for model_name_reg in models_reg:
	print(model_name_reg)    
	model_reg   = models_reg[model_name_reg]
	shuffle_reg = ShuffleSplit(n_splits = folds_reg, test_size = 0.3, random_state = seed_reg)
	results_reg = cross_val_score(model_reg, X_reg, Y_reg, cv = shuffle_reg, scoring = metric_reg)
    
	score_reg = round(results_reg.mean(),10)
	std_reg = round(results_reg.std(),10)
	model_results_reg.append(score_reg)
	model_std_reg.append(std_reg)
	model_names_reg.append(model_name_reg)
	#print("{}: {}, {}".format(model_name_reg, score_reg, std_reg))
end = time.time()
kf_cross_val_reg = pd.DataFrame({'Name': model_names_reg, 'R² Score': model_results_reg, 'Desvio Padrão': model_std_reg})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_val_reg)
#

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_regressor = pd.ExcelWriter('V5/cross_val_score_regressor_r2_score_shuffle_X_reg_Y_reg.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val_reg.to_excel(cross_val_score_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_regressor.save()

#TESTE07-1: #Treino e Teste dos Modelos com CrossValScore + 10 StratifiedKFold(shuffle = False) + R2 Score - X_reg, Y_reg

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

#from sklearn.metrics import r2_score

folds_reg   = 10

metric_reg  = custom_r2_by_correlation

import time

model_results_reg = []
model_std_reg = []
model_names_reg   = []

start = time.time()
for model_name_reg in models_reg:
	print(model_name_reg)    
	model_reg   = models_reg[model_name_reg]
	skfold_reg = StratifiedKFold(n_splits = folds_reg, shuffle = False, random_state = seed_reg)
	results_reg = cross_val_score(model_reg, X_reg, Y_reg, cv = skfold_reg, scoring = metric_reg)
    
	score_reg = round(results_reg.mean(),10)
	std_reg = round(results_reg.std(),10)
	model_results_reg.append(score_reg)
	model_std_reg.append(std_reg)
	model_names_reg.append(model_name_reg)
	#print("{}: {}, {}".format(model_name_reg, score_reg, std_reg))
end = time.time()
kf_cross_val_reg = pd.DataFrame({'Name': model_names_reg, 'R² Score': model_results_reg, 'Desvio Padrão': model_std_reg})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_val_reg)
#

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_regressor = pd.ExcelWriter('V5/cross_val_score_regressor_r2_score_X_reg_Y_reg_stratifiedkfold.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val_reg.to_excel(cross_val_score_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_regressor.save()

#TESTE08-1: Treino e Teste dos Modelos com CrossValScore + 10 StratifiedKFold(shuffle = True) - X_reg, Y_reg

from sklearn.model_selection import StratifiedKFold

from sklearn.model_selection import cross_val_score

#from sklearn.metrics import r2_score

folds_reg   = 10

metric_reg  = custom_r2_by_correlation

model_results_reg = []
model_std_reg = []
model_names_reg   = []

start = time.time()
for model_name_reg in models_reg:
	print(model_name_reg)    
	model_reg   = models_reg[model_name_reg]
	skfold_reg = StratifiedKFold(n_splits = folds_reg, shuffle = True, random_state = seed_reg)
	results_reg = cross_val_score(model_reg, X_reg, Y_reg, cv = skfold_reg, scoring = metric_reg)
    
	score_reg = round(results_reg.mean(),10)
	std_reg = round(results_reg.std(),10)
	model_results_reg.append(score_reg)
	model_std_reg.append(std_reg)
	model_names_reg.append(model_name_reg)
	#print("{}: {}, {}".format(model_name_reg, score_reg, std_reg))
end = time.time()
kf_cross_val_reg = pd.DataFrame({'Name': model_names_reg, 'R² Score': model_results_reg, 'Desvio Padrão': model_std_reg})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_val_reg)
#

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_val_score_regressor = pd.ExcelWriter('V5/cross_val_score_regressor_r2_score_X_reg_Y_reg_stratifiedkfold_shuffle.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_val_reg.to_excel(cross_val_score_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_val_score_regressor.save()
