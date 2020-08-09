models_A = {}

models_A["LightGBM 04"] = LGBMClassifier( max_depth = 100, learning_rate = 0.3, num_leaves = 500, n_estimators = 500, n_jobs = 3 )
models_A["LightGBM 07"] = LGBMClassifier( max_depth = 1000, learning_rate = 0.15, num_leaves = 2000, min_data_in_leaf = 200, n_estimators = 2000, n_jobs = 3 )
models_A["LightGBM 08"] = LGBMClassifier( max_depth = 1000, learning_rate = 0.1, num_leaves = 2000, min_data_in_leaf = 200, n_estimators = 2000, n_jobs = 3 )
models_A["LightGBM 28"] = LGBMClassifier( max_depth = 1880, learning_rate = 0.1, num_leaves = 100, n_estimators = 4500, min_data_in_leaf = 140, n_jobs = 3 )

models_B = {}
models_B["LightGBM 04"] = LGBMClassifier( max_depth = 100, learning_rate = 0.3, num_leaves = 500, n_estimators = 500, n_jobs = 3 )
models_B["LightGBM 07"] = LGBMClassifier( max_depth = 1000, learning_rate = 0.15, num_leaves = 2000, min_data_in_leaf = 200, n_estimators = 2000, n_jobs = 3 )
models_B["LightGBM 08"] = LGBMClassifier( max_depth = 1000, learning_rate = 0.1, num_leaves = 2000, min_data_in_leaf = 200, n_estimators = 2000, n_jobs = 3 )
models_B["LightGBM 28"] = LGBMClassifier( max_depth = 1880, learning_rate = 0.1, num_leaves = 100, n_estimators = 4500, min_data_in_leaf = 140, n_jobs = 3 )

model_A = []
model_B = []
f_valor_teste = []
p_valor_teste = []

from mlxtend.evaluate import combined_ftest_5x2cv

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

import time
start = time.time()

for model_name_A in models_A:
	model_tst_A   = models_A[model_name_A]
	for model_name_B in models_B:
		if model_name_A != model_name_B:        
			model_tst_B   = models_B[model_name_B]
			print(model_name_A, model_name_B)
			f, p = combined_ftest_5x2cv( estimator1 = model_tst_A,
										 estimator2 = model_tst_B,
            	                         X = X, 
            	                         y = Y,
            	                         scoring = make_scorer(matthews_corrcoef),
            	                         random_seed = 42 )
			model_A.append(model_name_A)
			model_B.append(model_name_B)
			f_valor_teste.append(f)
			p_valor_teste.append(p)
        
end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

#Tempo de Execução: 751.73 min

Resultado = pd.DataFrame({'Modelo A': model_A, 'Modelo B': model_B, 'Valor p': p_valor_teste, 'Valor f': f_valor_teste})

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
resultado_combined_ftest_classificacao_1 = pd.ExcelWriter('resultado_combined_ftest_lightgbm_classifier.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
Resultado.to_excel(resultado_combined_ftest_classificacao_1, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
resultado_combined_ftest_classificacao_1.save()