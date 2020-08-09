#ETAPA 17: CRIAÇÃO E AJUSTES DE HIPERPARAMETROS PARA O MODELO CATBOOST

#Parametrização do Modelo

from catboost import CatBoostClassifier

#1° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 500, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.15 )

#Accuracy Score : 0.9485983156063422
#Precision Score : 0.8412162162162162
#Recall Score : 0.6831275720164609
#F1 Score : 0.7539742619227858

#2° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 1000, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.01 )

#Accuracy Score : 0.9455932940571745
#Precision Score : 0.8496821071752951
#Recall Score : 0.6416323731138546
#F1 Score : 0.7311449785072294

#3° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 1000, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.15 )
#Accuracy Score : 0.94938911075086
#Precision Score : 0.848381601362862
#Recall Score : 0.6831275720164609
#F1 Score : 0.756838905775076

#4° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 2000, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.15 )

#Accuracy Score : 0.9498635878375706
#Precision Score : 0.8512361466325661
#Recall Score : 0.6848422496570644
#F1 Score : 0.7590269859369061

#5° Teste

classifier_cbc = CatBoostClassifier( iterations = 200,
                                     learning_rate = 0.01,
                                     depth = 10,
                                     random_seed = 42,
                                     bagging_temperature = 0.2,
                                     od_type = 'Iter',
                                     metric_period = 75,
                                     od_wait = 100 )

#Accuracy Score : 0.9369736269819303
#Precision Score : 0.8777142857142857
#Recall Score : 0.5267489711934157
#F1 Score : 0.6583797685383627

#6° Teste ******************

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 2000, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.15,
                                     random_seed = 42 )

#Accuracy Score : 0.9499822071092483
#Balanced Accuracy Score : 0.8354348882298397
#Precision Score : 0.850828729281768
#Recall Score : 0.686556927297668
#F1 Score : 0.7599164926931107
#Matthews Correlation Coefficient : 0.73760656051932
#Cohen’s Kappa Score : 0.7323548838173148

#7° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 2000, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.01,
                                     random_seed = 42 )

#Accuracy Score : 0.9480842987624056
#Balanced Accuracy Score : 0.8257132216014897
#Precision Score : 0.8507658643326039
#Recall Score : 0.6666666666666666
#F1 Score : 0.7475485483560852
#Matthews Correlation Coefficient : 0.7257332243184327
#Cohen’s Kappa Score : 0.7190896120339216

#8° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 2000, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.3,
                                     random_seed = 42 )

#Accuracy Score : 0.947965679490728
#Balanced Accuracy Score : 0.8302689457510479
#Precision Score : 0.8404255319148937
#Recall Score : 0.6772976680384087
#F1 Score : 0.750094948727687
#Matthews Correlation Coefficient : 0.7266627766986594
#Cohen’s Kappa Score : 0.7214284036739191

#9° Teste

classifier_cbc = CatBoostClassifier( depth = 11, 
                                     iterations = 2000, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.15,
                                     random_seed = 42 )

#Accuracy Score : 0.9489541734213752
#Balanced Accuracy Score : 0.8309767263642703
#Precision Score : 0.8491620111731844
#Recall Score : 0.6776406035665294
#F1 Score : 0.7537669273316803
#Matthews Correlation Coefficient : 0.7314434797249222
#Cohen’s Kappa Score : 0.7256928733781278

#10° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 4000, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.3,
                                     random_seed = 42 )

#Accuracy Score : 0.9480842987624056
#Balanced Accuracy Score : 0.8315289560199555
#Precision Score : 0.8391874735505713
#Recall Score : 0.6800411522633745
#F1 Score : 0.7512786512597083
#Matthews Correlation Coefficient : 0.7276320098014164
#Cohen’s Kappa Score : 0.7226506721910442

#11° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 4000, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.15,
                                     random_seed = 42 )

#Accuracy Score : 0.9499031275947966
#Balanced Accuracy Score : 0.8350919527017189
#Precision Score : 0.8507018290089323
#Recall Score : 0.6858710562414266
#F1 Score : 0.7594456047085628
#Matthews Correlation Coefficient : 0.737133679403788
#Cohen’s Kappa Score : 0.7318444242774473

#12° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 4000, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.1,
                                     random_seed = 42 )

#Accuracy Score : 0.9499426673520225
#Balanced Accuracy Score : 0.8346669348843981
#Precision Score : 0.851962457337884
#Recall Score : 0.6848422496570644
#F1 Score : 0.7593155893536121
#Matthews Correlation Coefficient : 0.7371847154865997
#Cohen’s Kappa Score : 0.7317502953155708

#13° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 4000, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.001,
                                     random_seed = 42 )

#Accuracy Score : 0.941441619548456
#Balanced Accuracy Score : 0.7845295614256922
#Precision Score : 0.8677601230138391
#Recall Score : 0.5805898491083676
#F1 Score : 0.6957057735771524
#Matthews Correlation Coefficient : 0.6811670180732218
#Cohen’s Kappa Score : 0.6647127084016538

#14° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 2000, 
                                     l2_leaf_reg = 50, 
                                     learning_rate = 0.15,
                                     random_seed = 42 )
#Accuracy Score : 0.9489146336641493
#Balanced Accuracy Score : 0.8343841720884966
#Precision Score : 0.8420387531592249
#Recall Score : 0.6855281207133059
#F1 Score : 0.7557655954631379
#Matthews Correlation Coefficient : 0.7323720301963581
#Cohen’s Kappa Score : 0.7275736050044995

#15° Teste

classifier_cbc = CatBoostClassifier()

#Accuracy Score : 0.9501799058953778
#Balanced Accuracy Score : 0.8348010130966886
#Precision Score : 0.8541488451668092
#Recall Score : 0.6848422496570644
#F1 Score : 0.7601827179291967
#Matthews Correlation Coefficient : 0.738329224892732
#Cohen’s Kappa Score : 0.7327602168124642

#16° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 2000, 
                                     l2_leaf_reg = 3.5, 
                                     learning_rate = 0.15,
                                     random_seed = 42 )

#Accuracy Score : 0.9502985251670555
#Balanced Accuracy Score : 0.8327803526680001
#Precision Score : 0.8595578673602081
#Recall Score : 0.6800411522633745
#F1 Score : 0.7593337162550258
#Matthews Correlation Coefficient : 0.7382889211268905
#Cohen’s Kappa Score : 0.7320409530747893

#17° Teste -- New Best

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 3000, 
                                     l2_leaf_reg = 3.5, 
                                     learning_rate = 0.15,
                                     random_seed = 42 )

#Accuracy Score : 0.9508916215254438
#Balanced Accuracy Score : 0.8352032477335601
#Precision Score : 0.8607758620689655
#Recall Score : 0.6848422496570644
#F1 Score : 0.7627960275019098
#Matthews Correlation Coefficient : 0.741787570446258
#Cohen’s Kappa Score : 0.7358018822838966

#18° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 3000, 
                                     l2_leaf_reg = 9, 
                                     learning_rate = 0.15,
                                     random_seed = 42 )

#Accuracy Score : 0.9499822071092483
#Balanced Accuracy Score : 0.8357331310205302
#Precision Score : 0.8502333474756045
#Recall Score : 0.6872427983539094
#F1 Score : 0.7600986155888487
#Matthews Correlation Coefficient : 0.7376999988937857
#Cohen’s Kappa Score : 0.7325290484339386

#19° Teste

classifier_cbc = CatBoostClassifier( depth = 6, 
                                     iterations = 500, 
                                     l2_leaf_reg = 3, 
                                     learning_rate = 0.03,
                                     loss_function = 'Logloss',
                                     random_seed = 42 )
#Accuracy Score : 0.9457909928433039
#Balanced Accuracy Score : 0.8133821489604647
#Precision Score : 0.8519362186788155
#Recall Score : 0.6412894375857339
#F1 Score : 0.7317550381530034
#Matthews Correlation Coefficient : 0.7110397563084447
#Cohen’s Kappa Score : 0.7022699200722549

#20° Teste

classifier_cbc = CatBoostClassifier( depth = 10, 
                                     iterations = 3000, 
                                     l2_leaf_reg = 2, 
                                     learning_rate = 0.15,
                                     random_seed = 42 )

#Accuracy Score : 0.9491123324502787
#Balanced Accuracy Score : 0.8297240192810232
#Precision Score : 0.8533622559652928
#Recall Score : 0.6745541838134431
#F1 Score : 0.753495498946562
#Matthews Correlation Coefficient : 0.7317937510285027
#Cohen’s Kappa Score : 0.7255557222856217



#Parâmetros Utilizados pelo Modelo

from pprint import pprint
print('Parameters Currently In Use:\n')
pprint(classifier_cbc.get_params())

# Fit e Predição

classifier_cbc.fit(X_train, Y_train)

Y_pred_cbc = classifier_cbc.predict(X_test)

#Accuracy Score : 0.9499822071092483
#Balanced Accuracy Score : 0.8354348882298397
#Precision Score : 0.850828729281768
#Recall Score : 0.686556927297668
#F1 Score : 0.7599164926931107
#Matthews Correlation Coefficient : 0.73760656051932
#Cohen’s Kappa Score : 0.7323548838173148

#Análise de Métricas

from sklearn.metrics import accuracy_score, balanced_accuracy_score, recall_score, precision_score, f1_score

#Accuracy Score

mtrc_accuracy_score_cbc = accuracy_score(Y_test, Y_pred_cbc)
print('Accuracy Score : ' + str(mtrc_accuracy_score_cbc))

#Balanced Accuracy

mtrc_balanced_accuracy_score_cbc = balanced_accuracy_score(Y_test, Y_pred_cbc)
print('Balanced Accuracy Score : ' + str(mtrc_balanced_accuracy_score_cbc))

#Precision Score

mtrc_precision_score_cbc = precision_score(Y_test, Y_pred_cbc)
print('Precision Score : ' + str(mtrc_precision_score_cbc))

#Recall Score

mtrc_recall_score_cbc = recall_score(Y_test, Y_pred_cbc)
print('Recall Score : ' + str(mtrc_recall_score_cbc))

#F1 Score

mtrc_f1_score_cbc = f1_score(Y_test, Y_pred_cbc)
print('F1 Score : ' + str(mtrc_f1_score_cbc))

#Matthews Correlation Coefficient

from sklearn.metrics import matthews_corrcoef

mtrc_matthews_corrcoef_cbc = matthews_corrcoef(Y_test, Y_pred_cbc)
print('Matthews Correlation Coefficient : ' + str(mtrc_matthews_corrcoef_cbc))

#Cohen’s Kappa

from sklearn.metrics import cohen_kappa_score

mtrc_cohen_kappa_score_cbc = cohen_kappa_score(Y_test, Y_pred_cbc)
print('Cohen’s Kappa Score : ' + str(mtrc_cohen_kappa_score_cbc))


#OTIMIZAÇÃO DE HIPERPARAMETROS - CATBOOST

#Randomized Search With Cross Validation

import numpy as np

from sklearn.model_selection import RandomizedSearchCV

from catboost import CatBoostClassifier

#Criação das Variáveis dos Parâmetros a Serem Testados

#depth: Defines the depth of the trees.
#depth_cbc = [int(x) for x in np.linspace(start = 8, stop = 16, num = 5)]
depth_cbc = 10

#iterations: The maximum number of trees that can be built.
# The final number of trees may be less than or equal to this number.
iterations_cbc = [int(x) for x in np.linspace(1000, 5000, num = 5)]

#l2_leaf_reg_cbc
l2_leaf_reg_cbc = [int(x) for x in np.linspace(1, 9, num = 5)]

#learning_rate: Defines the learning rate.
#Used for reducing the gradient step.
learning_rate_cbc = [0.01, 0.03, 0.05, 0.1, 0.15]

# Criação e Fit do Random Grid

random_grid_cbc = { 'depth': depth_cbc,
                    'iterations': iterations_cbc,
                    'l2_leaf_reg': l2_leaf_reg_cbc,
                    'learning_rate': learning_rate_cbc }

classifier_cbc_rscv = CatBoostClassifier()

classifier_cbc_rscv_random = RandomizedSearchCV( estimator = classifier_cbc_rscv, 
                                                 param_distributions = random_grid_cbc, 
                                                 n_iter = 50, 
                                                 cv = 3, 
                                                 verbose = 2, 
                                                 random_state = 42,
                                                 scoring = 'f1' )

classifier_cbc_rscv_random.fit(X_train, Y_train)

classifier_cbc_rscv_random.best_params_

classifier_cbc_rscv_random.best_score_

classifier_cbc_rscv_random.best_estimator_

classifier_cbc_rscv_random.best_index_

#Grid Search With Cross Validation

from sklearn.model_selection import GridSearchCV

from catboost import CatBoostClassifier

classifier_cbc_gscv = CatBoostClassifier()

classifier_cbc_gscv.get_params().keys()

grid_param_cbc = {
    'depth': [4, 7, 10],
    'iterations': [100, 200, 300, 400, 500],
    'l2_leaf_reg': [1,4,9],    
    'learning_rate': [0.03, 0.1, 0.15] 
}

classifier_cbc_gscv_gd_sr = GridSearchCV( estimator = classifier_cbc_gscv,  
                                          param_grid = grid_param_cbc,
                                          scoring = 'accuracy',
                                          cv = 3 )

classifier_cbc_gscv_gd_sr.fit(X, Y) 

best_parameters_cbc = classifier_cbc_gscv_gd_sr.best_params_  

print(best_parameters_cbc)

best_result_cbc = classifier_cbc_gscv_gd_sr.best_score_  

print(best_result_cbc)

pd.set_option('max_columns',200)

gd_sr_results_cbc = pd.DataFrame(classifier_cbc_gscv_gd_sr.cv_results_)






















































































































































































































































































































































































































