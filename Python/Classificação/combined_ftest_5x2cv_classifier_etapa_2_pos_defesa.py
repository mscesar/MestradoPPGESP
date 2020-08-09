models_A = {}
models_A["RGF"]     = RGFClassifier()
models_A["FastRGF"] = FastRGFClassifier()

models_B = {}
#models_B["KNN"]     = KNeighborsClassifier()
#models_B["SVC"]     = SVC()
models_B["LR"]      = LogisticRegression()
models_B["PRT"]     = Perceptron()
models_B["MLP"]     = MLPClassifier()
models_B["GNB"]     = GaussianNB()
models_B["BNB"]     = BernoulliNB()
models_B["ETC"]     = ExtraTreesClassifier()
#models_B["RGF"]     = RGFClassifier()
#models_B["FastRGF"] = FastRGFClassifier()
models_B["RF"]      = RandomForestClassifier()
models_B["AB"]      = AdaBoostClassifier()
models_B["GB"]      = GradientBoostingClassifier()
models_B["XGB"]     = XGBClassifier()
models_B["LGBM"]    = LGBMClassifier()
models_B["CB"]      = CatBoostClassifier()

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

Tempo de Execução: 1095.64 min

Resultado = pd.DataFrame({'Modelo A': model_A, 'Modelo B': model_B, 'Valor p': p_valor_teste, 'Valor f': f_valor_teste})

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
resultado_combined_ftest_classificacao_2 = pd.ExcelWriter('V3/resultado_combined_ftest_classificacao_2.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
Resultado.to_excel(resultado_combined_ftest_classificacao_2, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
resultado_combined_ftest_classificacao_2.save()