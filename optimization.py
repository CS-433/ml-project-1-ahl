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
    tx, _ = delete_useless_features(tx)
    tx = replace_outliers(tx, 2)
    tx, _, _ = standardize(tx)
    tx = add_col_one(tx)
    
    return tx_clean

def add_col_one(tx)
    # add column of 1 to our dataset
    return np.c_[np.ones((tx.shape[0],1)), tx]
        
def delete_useless_features(tx):
    # we remove the columns where there is a majority of -999 (NaN) elements 
    # each of these columns has 177457 datas that are equal to -999
    median = np.median(tx, axis=0) 
    indx = np.where(median == -999)[0]
    tx = np.delete(tx, indx, 1)
    
    return tx, indx

def replace_outliers(tx, r):
    tx_clean = np.copy(tx.T)

    for i in range(tx_clean.shape[0]):
        line = tx.T[i]
        med = np.median(line)
        std = np.std(line)
        low = np.max([-999, med-r*std])
        up = med+r*std
        tx_clean[i] = np.where(np.logical_and(line > low, line < up), line, med)

    return tx_clean.T
    
    
    
    
        
    