# data analysis and parameter optimization functions 

import numpy as np
from implementations import *

def best_lambda(y, tx, start, end, inter):
    # array of lambdas 
    lambdas = np.linspace(start, end, inter)
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
    tx_clean = delete_useless_features(tx)
    tx_clean = replace_outliers(tx)
    
    return tx_clean
        
def delete_useless_features(tx):
    # we remove the columns where there is a majority of -999 (NaN) elements 
    # each of these columns has 177457 datas that are equal to -999
    median = np.median(tx, axis=0) 
    indx = np.where(median == -999)
    tx_optimized = np.delete(tx, indx[0], 1)
    
    # we then standardize the dataset 
    standardize_dataset, _, _ = standardize(tx_optimized)
    
    # add column of 1 to our dataset
    np.c_[np.ones((standardize_dataset.shape[0],1)), standardize_dataset]
    
    return standardize_dataset

def replace_outliers(tx):
    tx_clean = np.copy(tx.T)

    for i in range(tx_clean.shape[0]):
        line = tx.T[i]
        med = np.median(line)
        std = np.std(line)
        low = np.max([-999, med-r*std])
        up = med+r*std
        tx_clean[i] = np.where(np.logical_and(line > low, line < up), line, med)

    return tx_clean.T
    
    
    
    
        
    