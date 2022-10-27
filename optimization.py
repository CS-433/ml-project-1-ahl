# data analysis and parameter optimization functions 

import numpy as np
from implementations import *

def ridge_regression_best_lambda(y, tx, log1, log2, step):
    # array of lambdas 
    lambdas = np.linspace(log1, log2, step)
    losses = []
    
    for lambda_ in lambdas: 
        
        w, loss = ridge_regression(y, tx, lambda_) 
        losses.append(loss)
        print("Current lambda={i}, the loss={l}".format(i=lambda_, l=loss))
        
    ind_lambda = np.argmin(losses)
    best_lambda = lambdas[ind_lambda]
    
    return best_lambda

#return standardize dataset, its mean and its std
def standardize(dataset):
    
    mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)
    
    standardize_dataset = (dataset - mean)/std
    
    return standardize_dataset, mean, std

# in case we want to standardize the test set given the mean and std of train set 
def standardize_test(dataset, mean, std): 
    return (dataset - mean)/std

def dataClean(tx): 
     
    # we remove the columns where there is a majority of -999 (NaN) elements 
    # each of these columns has 177457 datas that are equal to -999
    median = np.median(tx, axis=0) 
    indx = np.where(median == -999)
    tx_optimized = np.delete(tx, indx[0], 1)
    
    # we then standardize the dataset 
    standardize_dataset, _, _ = standardize(tx_optimized)
    
    return standardize_dataset
    
    
    
    
        
    