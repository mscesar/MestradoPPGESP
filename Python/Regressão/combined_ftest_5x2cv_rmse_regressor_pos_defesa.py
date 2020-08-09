models_reg_A = {}
models_reg_A["BayesianRidge"]    = BayesianRidge()
models_reg_A["SVR"]              = SVR()
models_reg_A["ExtraTrees"]       = ExtraTreesRegressor()
models_reg_A["RandomForest"]     = RandomForestRegressor()
models_reg_A["GradientBoost"]    = GradientBoostingRegressor()
models_reg_A["XGBoost"]          = XGBRegressor()
models_reg_A["LightGBM"]         = LGBMRegressor()
models_reg_A["CatBoost"]         = CatBoostRegressor()

models_reg_B = {}
models_reg_B["BayesianRidge"]    = BayesianRidge()
models_reg_B["SVR"]              = SVR()
models_reg_B["ExtraTrees"]       = ExtraTreesRegressor()
models_reg_B["RandomForest"]     = RandomForestRegressor()
models_reg_B["GradientBoost"]    = GradientBoostingRegressor()
models_reg_B["XGBoost"]          = XGBRegressor()
models_reg_B["LightGBM"]         = LGBMRegressor()
models_reg_B["CatBoost"]         = CatBoostRegressor()

model_reg_A = []
model_reg_B = []
f_valor_teste_reg = []
p_valor_teste_reg = []

from mlxtend.evaluate import combined_ftest_5x2cv

from sklearn.metrics import mean_squared_error

import time
start = time.time()
      
for model_reg_name_A in models_reg_A:
	model_reg_tst_A = models_reg_A[model_reg_name_A]
	for model_reg_name_B in models_reg_B:
		if model_reg_name_A != model_reg_name_B:            
			model_reg_tst_B = models_reg_B[model_reg_name_B]
			print(model_reg_name_A, model_reg_name_B)
			f, p = combined_ftest_5x2cv( estimator1 = model_reg_tst_A,
									     estimator2 = model_reg_tst_B,
                                         X = X_reg, 
                                         y = Y_reg,
                                         scoring = 'neg_mean_squared_error',
                                         random_seed = 42 )
			model_reg_A.append(model_reg_name_A)
			model_reg_B.append(model_reg_name_B)
			f_valor_teste_reg.append(f)
			p_valor_teste_reg.append(p)
        
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
Tempo de Execução: 288.52 min 
        
Resultado = pd.DataFrame({'Modelo A': model_reg_A, 'Modelo B': model_reg_B, 'Valor p': p_valor_teste_reg, 'Valor f': f_valor_teste_reg})

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
resultado_combined_ftest_regressao = pd.ExcelWriter('V5/resultado_combined_ftest_rmse_regressao.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
Resultado.to_excel(resultado_combined_ftest_regressao, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
resultado_combined_ftest_regressao.save()