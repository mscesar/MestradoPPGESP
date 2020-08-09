models_reg_A = {}
models_reg_A["XGB 06"]          = XGBRegressor( n_estimators = 500, min_child_weight = 70, max_depth = 45, learning_rate = 0.01, n_jobs = 5 )
models_reg_A["XGB 07"]          = XGBRegressor( n_estimators = 400, min_child_weight = 60, max_depth = 50, learning_rate = 0.01, n_jobs = 5 )
models_reg_A["XGB 08"]          = XGBRegressor( n_estimators = 4000, min_child_weight = 40, max_depth = 450, learning_rate = 0.001, n_jobs = 5 )
models_reg_A["XGB 09"]          = XGBRegressor( n_estimators = 500, min_child_weight = 50, max_depth = 60, learning_rate = 0.01, n_jobs = 5 )
models_reg_A["XGB 10"]          = XGBRegressor( n_estimators = 400, min_child_weight = 50, max_depth = 500, learning_rate = 0.01, n_jobs = 5 )
models_reg_A["XGB 11"]          = XGBRegressor( n_estimators = 550, min_child_weight = 30, max_depth = 375, learning_rate = 0.01, n_jobs = 5 )

models_reg_B = {}
models_reg_B["XGB 06"]          = XGBRegressor( n_estimators = 500, min_child_weight = 70, max_depth = 45, learning_rate = 0.01, n_jobs = 5 )
models_reg_B["XGB 07"]          = XGBRegressor( n_estimators = 400, min_child_weight = 60, max_depth = 50, learning_rate = 0.01, n_jobs = 5 )
models_reg_B["XGB 08"]          = XGBRegressor( n_estimators = 4000, min_child_weight = 40, max_depth = 450, learning_rate = 0.001, n_jobs = 5 )
models_reg_B["XGB 09"]          = XGBRegressor( n_estimators = 500, min_child_weight = 50, max_depth = 60, learning_rate = 0.01, n_jobs = 5 )
models_reg_B["XGB 10"]          = XGBRegressor( n_estimators = 400, min_child_weight = 50, max_depth = 500, learning_rate = 0.01, n_jobs = 5 )
models_reg_B["XGB 11"]          = XGBRegressor( n_estimators = 550, min_child_weight = 30, max_depth = 375, learning_rate = 0.01, n_jobs = 5 )

model_reg_A = []
model_reg_B = []
f_valor_teste_reg = []
p_valor_teste_reg = []

from mlxtend.evaluate import combined_ftest_5x2cv

import time
start = time.time()

for model_reg_name_A in models_reg_A:
	model_reg_tst_A   = models_reg_A[model_reg_name_A]
	for model_reg_name_B in models_reg_B:
		if model_reg_name_A != model_reg_name_B:        
			model_reg_tst_B = models_reg_B[model_reg_name_B]
			print(model_reg_name_A, model_reg_name_B)
			f, p = combined_ftest_5x2cv( estimator1 = model_reg_tst_A,
										 estimator2 = model_reg_tst_B,
                         	             X = X_reg, 
                          	             y = Y_reg,
                          	             scoring = custom_r2_by_correlation,
                          	             random_seed = 42 )
			model_reg_A.append(model_reg_name_A)
			model_reg_B.append(model_reg_name_B)
			f_valor_teste_reg.append(f)
			p_valor_teste_reg.append(p)
        
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))
Tempo de Execução: 826.12 min
        
Resultado = pd.DataFrame({'Modelo A': model_reg_A, 'Modelo B': model_reg_B, 'Valor p': p_valor_teste_reg, 'Valor f': f_valor_teste_reg})

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
resultado_combined_ftest_regressao = pd.ExcelWriter('V5/resultado_combined_ftest_r2_score_xgboost_regressor.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
Resultado.to_excel(resultado_combined_ftest_regressao, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
resultado_combined_ftest_regressao.save()