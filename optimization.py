# data analysis and parameter optimization functions 

import numpy as np
from implementations import *

def ridge_regression_best_lambda(y, tx):
    # array of lambdas 
    lambdas = np.logspace(-4, 0, 30)
    losses = []
    
    for lambda_ in lambdas: 
        
        w, loss = ridge_regression(y, tx, lambda_) 
        losses.append(loss)
        print("Current lambda={i}, the loss={l}".format(i=lambda_, l=loss))
        
    ind_lambda = np.argmin(losses)
    best_lambda = lambdas[ind_lambda]
    
    return best_lambda
    
        
    