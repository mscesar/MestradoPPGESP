#ETAPA XX: APLICAÇÃO DO LDA

#Bloco 01: Aplicação do LDA 

#Variance Caused by Each of the Principal Components

from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA

lda = LDA()

lda.get_params().keys()
 
X_train_lda_new = lda.fit_transform(X_train_lda, Y_train)
X_test_lda_new = lda.transform(X_test_lda)

print('Original Number of Features:', X_train_lda.shape[1])
print('Reduced Number of Features:', X_train_lda_new.shape[1])

print('Original Number of Features:', X_test_lda.shape[1])
print('Reduced Number of Features:', X_test_lda_new.shape[1])

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(lda.get_params())

explained_variance_lda = lda.explained_variance_ratio_ 

for i in explained_variance_pca:
    print(format(i*100, 'f'))

plt.figure(1, figsize = (14, 7))
plt.plot(explained_variance_lda, linewidth = 2)
plt.axis('tight')
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')

plt.figure(1, figsize = (14, 7))
plt.clf()
plt.axes([.2, .2, .7, .7])
plt.plot(explained_variance_lda, linewidth=5)
plt.xlabel('Number of Components')
plt.ylabel('Explained Variance')

plt.figure(1, figsize = (14, 7))
plt.plot(np.cumsum(explained_variance_lda))
plt.xlabel('Number of Components')
plt.ylabel('Cumulative Explained Variance')
plt.title('Name Trends LDA')

#1° Teste

lda_new = LDA(n_components = 250)

X_train_lda_new = lda_new.fit_transform(X_train_lda, Y_train)  
X_test_lda_new = lda_new.transform(X_test_lda)

print('Original Number of Features:', X_train_lda.shape[1])
print('Reduced Number of Features:', X_train_lda_new.shape[1])

print('Original Number of Features:', X_test_lda.shape[1])
print('Reduced Number of Features:', X_test_lda_new.shape[1])

#2° Teste

lda_new = LDA(n_components = 300)

X_train_lda_new = lda_new.fit_transform(X_train_lda, Y_train)
X_test_lda_new = lda_new.transform(X_test_lda)

print('Original Number of Features:', X_train_lda.shape[1])
print('Reduced Number of Features:', X_train_lda_new.shape[1])

print('Original Number of Features:', X_test_lda.shape[1])
print('Reduced Number of Features:', X_test_lda_new.shape[1])

#Bloco 02: Parametrização do Modelo

from lightgbm import LGBMClassifier

classifier_lgbm_lda = LGBMClassifier( max_depth = 500, 
                                      learning_rate = 0.01,
                                      num_leaves = 1000,
                                      min_data_in_leaf = 200,
                                      n_estimators = 2000,
                                      objective = 'binary',
                                      metric = 'binary_logloss',
                                      random_state = 42)

#Bloco 03: Fit e Predição
								  
classifier_lgbm_lda.fit(X_train_lda_new, Y_train)
Y_pred_lgbm_lda = classifier_lgbm_lda.predict(X_test_lda_new)
    
#Bloco 04: Análise de Métricas

from sklearn.metrics import accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_lgbm_lda = accuracy_score(Y_test, Y_pred_lgbm_lda)
print('Accuracy Score : ' + str(mtrc_accuracy_score_lgbm_lda))

#Precision Score

mtrc_precision_score_lgbm_lda = precision_score(Y_test, Y_pred_lgbm_lda)
print('Precision Score : ' + str(mtrc_precision_score_lgbm_lda))

#Recall Score

mtrc_recall_score_lgbm_lda = recall_score(Y_test, Y_pred_lgbm_lda)
print('Recall Score : ' + str(mtrc_recall_score_lgbm_lda))

#F1 Score

mtrc_f1_score_lgbm_lda = f1_score(Y_test, Y_pred_lgbm_lda)
print('F1 Score : ' + str(mtrc_f1_score_lgbm_lda))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_lgbm_lda = matthews_corrcoef(Y_test, Y_pred_lgbm_lda)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_lgbm_lda))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_lgbm_lda = cohen_kappa_score(Y_test, Y_pred_lgbm_lda)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_lgbm_lda))