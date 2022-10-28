# data analysis and parameter optimization functions 

import numpy as np
from implementations import *

def best_lambda(y, tx, start, end, inter):
    # array of lambdas 
    lambdas = np.logspace(start, end, inter)
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
    
    return standardize_dataset

def split_train(tx, y): 
    # we split our training set into 4 parts, each part contain the datas for each section of PRI_jet_num, column number 22 of the dataset
    # this is the unique feature that contains integer number as well 
    
    col = tx[:,22]
    tx = np.delete(tx, 22, 1) 
    
    tx_0 = tx[col == 0]
    tx_1 = tx[col == 1]
    tx_2 = tx[col == 2]
    tx_3 = tx[col == 3]
    
    #we also split the labels in the same way 
    y_0 = y[col == 0]
    y_1 = y[col == 1]
    y_2 = y[col == 2]
    y_3 = y[col == 3]
    
    return [tx_0, tx_1, tx_2, tx_3], [y_0, y_1, y_2, y_3] 

def add_col_one(tx):
    # add column of 1 to our dataset
    return np.c_[np.ones((tx.shape[0],1)), tx]
        
def delete_useless_features(tx):
    # we remove the columns where there is a majority of -999 (NaN) elements 
    # each of these columns has 177457 datas that are equal to -999
    median = np.median(tx, axis=0) 
    indx = np.where(median == -999)[0]
    tx = np.delete(tx, indx, 1)
    
    return tx, indx

def miss_to_nan(tx):
    return np.where(tx > -999, tx, np.nan)

def replace_outliers(tx, r=2):
    tx_clean = np.copy(tx.T)

    for i in range(tx_clean.shape[0]):
        line = tx.T[i]
        med = np.median(line)
        std = np.std(line)
        
        low = med-r*std
        up = med+r*std
        
        line = np.where(np.logical_and(line > low, line < up), line, np.nan)
        tx_clean[i] = np.where(np.isnan(line), line, np.nanmedian(line))

    return tx_clean.T

def outliers_to_nan(tx, r=2):
    tx_clean = np.copy(tx.T)

    for i in range(tx_clean.shape[0]):
        line = tx.T[i]
        med = np.nanmedian(line)
        std = np.nanstd(line)
        
        low = med-r*std
        up = med+r*std
        
        tx_clean[i] = np.where(np.logical_and(line > low, line < up), line, np.nan)

    return tx_clean.T

def nan_to_median(tx):
    tx_clean = np.copy(tx.T)
    
    for i in range(tx_clean.shape[0]):
        line = tx.T[i]        
        tx_clean[i] = np.where(~np.isnan(line), line, np.nanmedian(line))

    return tx_clean.T
    

#calculates the nb of corrected claissification
def calculate_accuracy(true_pred, y_pred): 
    nb_true = np.sum(y_pred == true_pred)
    return nb_true/len(true_pred)

def dataClean(tx, y): 
    tx_train, y_train = split_train(tx, y)
    
    for i in range(4): 
        tx_train[i], indx = delete_useless_features(tx_train[i])
        tx_train[i] = replace_outliers(tx_train[i], r=2)
        #tx_train[i] = standardize(tx_train[i])
        tx_train[i] = add_col_one(tx_train[i])
        
    return tx_train, y_train

def dataClean_without_splitting(tx):
    # tx, deleted = delete_useless_features(tx)
    tx = miss_to_nan(tx)
    # tx = replace_outliers(tx, r=1.5)
    tx = outliers_to_nan(tx, r=3)
    tx = nan_to_median(tx)
    tx = standardize(tx)
    tx = add_col_one(tx)
        
    return tx #, deleted
