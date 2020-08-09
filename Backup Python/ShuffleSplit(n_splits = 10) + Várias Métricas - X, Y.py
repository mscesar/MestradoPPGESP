#Treino e Teste dos Modelos com CrossValidate + ShuffleSplit(n_splits = 10) + Várias Métricas - X, Y 

from sklearn.model_selection import ShuffleSplit

from sklearn.model_selection import cross_validate

from sklearn.metrics import make_scorer

from sklearn.metrics import matthews_corrcoef

from sklearn.metrics import cohen_kappa_score

import time

metricas_cross_validate = { 'accuracy': 'accuracy',
                            'balanced_accuracy': 'balanced_accuracy',
                            'average_precision': 'average_precision',
                            'f1': 'f1',
                            'precision': 'precision',
                            'recall': 'recall',
                            'roc_auc': 'roc_auc',
                            'ms_matthews_corrcoef': make_scorer(matthews_corrcoef),
                            'ms_cohen_kappa_score': make_scorer(cohen_kappa_score) }

lst_names = []
lst_fit_time = []
lst_score_time = []
lst_test_accuracy = []
lst_test_balanced_accuracy = []
lst_test_average_precision = []
lst_test_f1 = []
lst_test_precision = []
lst_test_recall = []
lst_test_roc_auc = []
lst_test_matthews_corrcoef = []
lst_test_cohen_kappa_score = []

start = time.time()
for var_name, model in models:
    shuffle = ShuffleSplit(n_splits = 10, test_size = 0.3, random_state = 42) 
    result = cross_validate( estimator = model, 
                             X = X, 
                             y = Y, 
                             cv = shuffle, 
                             scoring = metricas_cross_validate, 
                             return_train_score = False )    
    var_fit_time   = result['fit_time'].mean()
    var_score_time = result['score_time'].mean()
    var_test_accuracy = result['test_accuracy'].mean()
    var_test_balanced_accuracy = result['test_balanced_accuracy'].mean()
    var_test_average_precision = result['test_average_precision'].mean()
    var_test_f1 = result['test_f1'].mean()
    var_test_precision = result['test_precision'].mean()
    var_test_recall = result['test_recall'].mean()
    var_test_roc_auc = result['test_roc_auc'].mean()
    var_test_matthews_corrcoef = result['test_ms_matthews_corrcoef'].mean()
    var_test_cohen_kappa_score = result['test_ms_cohen_kappa_score'].mean()
    lst_names.append(var_name)
    lst_fit_time.append(var_fit_time)
    lst_score_time.append(var_score_time)
    lst_test_accuracy.append(var_test_accuracy)
    lst_test_balanced_accuracy.append(var_test_balanced_accuracy)
    lst_test_average_precision.append(var_test_average_precision)
    lst_test_f1.append(var_test_f1)
    lst_test_precision.append(var_test_precision)
    lst_test_recall.append(var_test_recall)
    lst_test_roc_auc.append(var_test_roc_auc)
    lst_test_matthews_corrcoef.append(var_test_matthews_corrcoef)
    lst_test_cohen_kappa_score.append(var_test_cohen_kappa_score)
end = time.time()    
kf_cross_validate = pd.DataFrame({'Name': lst_names, 
                                  'Fit Time': lst_fit_time, 
                                  'Score Time': lst_score_time, 
                                  'Test Accuracy': lst_test_accuracy, 
                                  'Test Balanced Accuracy': lst_test_balanced_accuracy, 
                                  'Test Average Precision': lst_test_average_precision, 
                                  'Test F1': lst_test_f1, 
                                  'Test Precision': lst_test_precision, 
                                  'Test Recall': lst_test_recall,
                                  'Test ROC AUC': lst_test_roc_auc, 
                                  'Test Matthews Correlation Coefficient': lst_test_matthews_corrcoef,
                                  'Test Cohen’s Kappa Score': lst_test_cohen_kappa_score})

print("Tempo de Execução: {:.2f} min".format((end - start)/60))
print(kf_cross_validate)
#Tempo de Execução: 5466.32 min

# Create a Pandas Excel writer using XlsxWriter as the engine.
cross_validate_classifier = pd.ExcelWriter('V3/cross_validate_classifier_shuffle_X_Y.xlsx', engine = 'xlsxwriter')
# Convert the dataframe to an XlsxWriter Excel object.
kf_cross_validate.to_excel(cross_validate_classifier, sheet_name = 'Dados')
# Close the Pandas Excel writer and output the Excel file.
cross_validate_classifier.save()