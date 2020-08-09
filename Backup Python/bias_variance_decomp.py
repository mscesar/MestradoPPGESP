from mlxtend.evaluate import bias_variance_decomp

import time
start = time.time()

avg_expected_loss, avg_bias, avg_var = bias_variance_decomp( classifier_lgbm, 
                                                             X_train, 
                                                             Y_train, 
                                                             X_test, 
                                                             Y_test, 
                                                             loss = '0-1_loss',
                                                             random_seed = 42)

end = time.time()
print("Tempo de Execução: {:.2f} min".format((end - start)/60))

print('Average expected loss: %.3f' % avg_expected_loss)
print('Average bias: %.3f' % avg_bias)
print('Average variance: %.3f' % avg_var)