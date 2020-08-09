#ETAPA XX: APLICAÇÃO DO KERNEL PCA

#Bloco 01: Aplicação do KernelPCA 

#Variance Caused by Each of the Principal Components

from sklearn.decomposition import KernelPCA

kpca_linear = KernelPCA(kernel = 'linear')

kpca_linear.get_params().keys()

X_train_kpca_new = kpca_linear.fit_transform(X_train_kpca)
X_test_kpca_new = kpca_linear.transform(X_test_kpca)

print('Original Number of Features:', X_train_kpca.shape[1])
print('Reduced Number of Features:', X_train_kpca_new.shape[1])

print('Original Number of Features:', X_test_kpca.shape[1])
print('Reduced Number of Features:', X_test_kpca_new.shape[1])

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(kpca_linear.get_params())

explained_variance_kpca_linear = kpca_linear.explained_variance_ratio_

kpca_rbf = KernelPCA(kernel = 'rbf')
#kpca = KernelPCA(kernel = "rbf", gamma = 15, n_components = 1)

kpca_rbf.get_params().keys()
 
X_train_kpca_new = kpca_rbf.fit_transform(X_train_kpca)
X_test_kpca_new = kpca_rbf.transform(X_test_kpca)

print('Original Number of Features:', X_train_kpca.shape[1])
print('Reduced Number of Features:', X_train_kpca_new.shape[1])

print('Original Number of Features:', X_test_kpca.shape[1])
print('Reduced Number of Features:', X_test_kpca_new.shape[1])

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(kpca_rbf.get_params())

explained_variance_kpca_rbf = kpca_rbf.explained_variance_ratio_ 

#1° Teste

kpca_linear_new = KernelPCA(n_components = 250, kernel = 'linear')

X_train_kpca_new = kpca_linear_new.fit_transform(X_train_kpca)
X_test_kpca_new = kpca_linear_new.transform(X_test_kpca)

kpca_rbf_new = KernelPCA(n_components = 250, kernel = 'rbf')

X_train_kpca_new = kpca_rbf_new.fit_transform(X_train_kpca)
X_test_kpca_new = kpca_rbf_new.transform(X_test_kpca)

#2° Teste

kpca_linear_new = KernelPCA(n_components = 300, kernel = 'linear')

X_train_kpca_new = kpca_linear_new.fit_transform(X_train_kpca)
X_test_kpca_new = kpca_linear_new.transform(X_test_kpca)

kpca_rbf_new = KernelPCA(n_components = 300, kernel = 'rbf')

X_train_kpca_new = kpca_rbf_new.fit_transform(X_train_kpca)
X_test_kpca_new = kpca_rbf_new.transform(X_test_kpca)

#3° Teste

kpca_linear_new = KernelPCA(0.99, kernel = 'linear')

X_train_kpca_new = kpca_linear_new.fit_transform(X_train_kpca)
X_test_kpca_new = kpca_linear_new.transform(X_test_kpca)

kpca_linear_new.n_components_

kpca_new = KernelPCA(0.99, kernel = 'rbf')

X_train_kpca_new = kpca_rbf_new.fit_transform(X_train_kpca)
X_test_kpca_new = kpca_rbf_new.transform(X_test_kpca)

kpca_rbf_new.n_components_

#Bloco 02: Parametrização do Modelo

from lightgbm import LGBMClassifier

classifier_lgbm_kpca = LGBMClassifier( max_depth = 500, 
                                       learning_rate = 0.01,
                                       num_leaves = 1000,
                                       min_data_in_leaf = 200,
                                       n_estimators = 2000,
                                       objective = 'binary',
                                       metric = 'binary_logloss',
                                       random_state = 42)

#Bloco 03: Fit e Predição
								  
classifier_lgbm_kpca.fit(X_train_kpca_new, Y_train)
Y_pred_lgbm_kpca = classifier_lgbm_kpca.predict(X_test_kpca_new)
    
#Bloco 04: Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm_kpca = accuracy_score(Y_test, Y_pred_lgbm_kpca)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm_kpca))

#Precision Score

mtrc_precision_score_lgbm_kpca = precision_score(Y_test, Y_pred_lgbm_kpca)
print('Precision Score : ' + str(mtrc_precision_score_lgbm_kpca))

#Recall Score

mtrc_recall_score_lgbm_kpca = recall_score(Y_test, Y_pred_lgbm_kpca)
print('Recall Score : ' + str(mtrc_recall_score_lgbm_kpca))

#F1 Score

mtrc_f1_score_lgbm_kpca = f1_score(Y_test, Y_pred_lgbm_kpca)
print('F1 Score : ' + str(mtrc_f1_score_lgbm_kpca))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm_kpca = matthews_corrcoef(Y_test, Y_pred_lgbm_kpca)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm_kpca))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm_kpca = cohen_kappa_score(Y_test, Y_pred_lgbm_kpca)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm_kpca))