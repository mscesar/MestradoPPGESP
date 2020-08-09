#Treino e Teste dos Modelos com CrossValidate + ShuffleSplit(n_splits = 10) + Várias Métricas - X_reg, Y_reg

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_validate

from sklearn.metrics import explained_variance_score

from sklearn.metrics import mean_absolute_error

from sklearn.metrics import median_absolute_error

from sklearn.metrics import mean_squared_error

from sklearn.metrics import r2_score

from sklearn.metrics import max_error

import time

metricas_cross_validate_reg = ['explained_variance', 
                               'neg_mean_absolute_error',
                               'neg_median_absolute_error',                               
                               'neg_mean_squared_error',
                               'r2',
                               'max_error']

lst_names_reg = []
lst_fit_time_reg = []
lst_score_time_reg = []
lst_test_explained_variance_reg = []
lst_test_neg_mean_absolute_error_reg = []
lst_test_neg_median_absolute_error_reg = []
lst_test_neg_mean_squared_error_reg = []
lst_test_root_mean_squared_error_reg = []
lst_test_r2_reg = []
lst_test_max_error_reg = []

start = time.time()
for model_name_reg in models_reg:
    model_reg = models_reg[model_name_reg]
    shuffle_reg = ShuffleSplit(n_splits = folds_reg, test_size = 0.3, random_state = seed_reg) 
    result_reg = cross_validate( estimator = model_reg, 
                                 X = X_reg, 
                                 y = Y_reg, 
                                 cv = shuffle_reg, 
                                 scoring = metricas_cross_validate_reg, 
                                 return_train_score = False )
    var_name_reg = model_name_reg
    var_fit_time_reg   = result_reg['fit_time'].mean()
    var_score_time_reg = result_reg['score_time'].mean()
    var_test_explained_variance_reg = result_reg['test_explained_variance'].mean()
    var_test_neg_mean_absolute_error_reg = result_reg['test_neg_mean_absolute_error'].mean()
    var_test_neg_median_absolute_error_reg = result_reg['test_neg_median_absolute_error'].mean()    
    var_test_neg_mean_squared_error_reg = result_reg['test_neg_mean_squared_error'].mean()
    var_mean_squared_error_reg = result_reg['test_neg_mean_squared_error']*-1
    var_test_root_mean_squared_error_reg = np.sqrt(var_mean_squared_error_reg).mean()
    var_test_r2_reg = result_reg['test_r2'].mean()
    var_test_max_error_reg = result_reg['test_max_error'].mean()
    lst_names_reg.append(var_name_reg)
    lst_fit_time_reg.append(var_fit_time_reg)
    lst_score_time_reg.append(var_score_time_reg)
    lst_test_explained_variance_reg.append(var_test_explained_variance_reg)
    lst_test_neg_mean_absolute_error_reg.append(var_test_neg_mean_absolute_error_reg*-1)
    lst_test_neg_median_absolute_error_reg.append(var_test_neg_median_absolute_error_reg*-1)    
    lst_test_neg_mean_squared_error_reg.append(var_test_neg_mean_squared_error_reg*-1)
    lst_test_root_mean_squared_error_reg.append(var_test_root_mean_squared_error_reg)
    lst_test_r2_reg.append(var_test_r2_reg)
    lst_test_max_error_reg.append(var_test_max_error_reg)
end = time.time()    
kf_cross_validate_reg = pd.DataFrame({'Name': lst_names_reg, 
                                      'Fit Time': lst_fit_time_reg, 
                                      'Score Time': lst_score_time_reg, 
                                      'Test Explained Variance': lst_test_explained_variance_reg, 
                                      'Test Mean Absolute Error (MAE)': lst_test_neg_mean_absolute_error_reg, 
                                      'Test Median Absolute Error': lst_test_neg_median_absolute_error_reg,                                      
                                      'Test Mean Squared Error (MSE)': lst_test_neg_mean_squared_error_reg, 
                                      'Test Root Mean Squared Error (RMSE)': lst_test_root_mean_squared_error_reg,
                                      'Test R2 Score': lst_test_r2_reg,
                                      'Test Max Error': lst_test_max_error_reg})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))   
print(kf_cross_validate_reg)
#Tempo de Execução: 87.65 min

# Create a Pandas Excel Writer Using XlsxWriter as the Engine.
cross_validate_regressor = pd.ExcelWriter('V3/cross_validate_regressor_shuffle_X_reg_Y_reg.xlsx', engine = 'xlsxwriter')
# Convert the Dataframe to an XlsxWriter Excel Object.
kf_cross_validate_reg.to_excel(cross_validate_regressor, sheet_name = 'Dados')
# Close the Pandas Excel Writer and Output the Excel File.
cross_validate_regressor.save()