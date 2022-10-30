# data analysis and parameter optimization functions 

import numpy as np
from implementations import *

def build_poly(x, degree):
    """polynomial basis functions for input data x, for j=0 up to j=degree.
    
    Args:
        x: numpy array of shape (N,), N is the number of samples.
        degree: integer.
        
    Returns:
        poly: numpy array of shape (N,d+1)
    """
    result = np.ones((x.shape[0], 1))
    
    for deg in range(1, degree+1):
        result = np.c_[result,np.power(x,deg)]
    
    return result

def build_k_indices (y, k_fold, seed):
    """build k indices for k-fold."""
    num_row = y.shape[0]
    interval = int(num_row / k_fold)
    np.random.seed(seed)
    indices = np.random.permutation(num_row)
    k_indices = [indices[k * interval: (k + 1) * interval]
                 for k in range(k_fold)]
    return np.array(k_indices)

def cross_validation (y,tx, k_indices, k, lambda_ ):
    te_ind = k_indices[k]
    tr_ind = k_indices[~(np.arange(k_indices.shape[0])==k)]
    tr_ind = tr_ind.reshape(-1)
    y_te   = y[te_ind]
    y_tr   = y[tr_ind]
    tx_te  = tx[te_ind]
    tx_tr  = tx[tr_ind]
    w,loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
    _,loss_te = ridge_regression(y_te,tx_te, lambda_)
    return loss_tr, loss_te,w

def best_lambda_degree(y, tx,k_fold, lambdas, degrees,seed):
    k_indices = build_k_indices(y, k_fold, seed)
    best_lambdas = []
    best_rmses   = []
    
    for degree in degrees : 
        rmse_te = []
        for lambda_ in lambdas : 
            rmse_temp = []
            for k in range(k_fold): 
                _, loss_te,_ = cross_validation (y, tx, k_indices, k, lambda_)
                rmse_temp.append(loss_te)
            rmse_te.append(np.mean(rmse_temp))
        
        indice_lambda = np.argmin(rmse_te)
        best_lambdas.append ( lambdas[indice_lambda])
        best_rmses.append( rmse_te[indice_lambda])
    
    indice_deg = np.argmin(best_rmses)
    best_lambda = best_lambdas[indice_deg]
    best_degree = degrees[indice_deg]
    return  best_lambda, best_degree

def best_lambda(y, tx, start, end, inter):
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
    
    txs = []
    ys = []
    ids = []
    
    for i in range(4):
        ids.append(np.where(col == i))
        txs.append(tx[ids[i]])
        ys.append(y[ids[i]])
    
    return txs, ys, ids 

def add_bias(tx):
    # add column of 1 to our dataset
    return np.c_[np.ones((tx.shape[0],1)), tx]
        
def remove_useless_features(tx):
    # we remove the columns where there is a majority of -999 (NaN) elements 
    # each of these columns has 177457 datas that are equal to -999
    
    # median = np.median(tx, axis=0) 
    # indx = np.where(median == -999)[0]
    
    std_ = np.std(tx, axis=0) 
    indx = np.where(std_ == 0)[0]
    
    tx = np.delete(tx, indx, 1)
    
    return tx, indx

def miss_to_nan(tx):
    return np.where(tx > -999, tx, np.nan)

def replace_outliers(tx, r):
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

def outliers_to_nan(tx, r):
    tx_clean = np.copy(tx.T)

    for i in range(tx_clean.shape[0]):
        line = tx.T[i]
        med = np.nanmedian(line)
        std = np.nanstd(line)
        
        low = med-r*std
        up = med+r*std
        
        tx_clean[i] = np.where(np.logical_and(line >= low, line <= up), line, np.nan)

    return tx_clean.T

def nan_to_median(tx):
    tx_clean = np.copy(tx.T)
    
    for i in range(tx_clean.shape[0]):
        line = tx.T[i]        
        tx_clean[i] = np.where(~np.isnan(line), line, np.nanmedian(line))

    return tx_clean.T

def useless_features_to_one(tx):
    all_std = np.std(tx, axis=0)
    idx = [i for i, std in enumerate(all_std) if std==0]
    tx[:, idx] = 1
    return tx

#calculates the nb of corrected claissification
def calculate_accuracy(true_pred, y_pred): 
    nb_true = np.sum(y_pred == true_pred)
    return nb_true/len(true_pred)

def dataClean(tx, y, r=1.5): 
    tx_train, y_train, ids_train = split_train(tx, y)
    
    for i in range(4): 
        tx_train[i], _ = remove_useless_features(tx_train[i])
        # tx_train[i] = useless_features_to_one(tx_train[i])
        tx_train[i] = miss_to_nan(tx_train[i])
        # tx_train[i] = replace_outliers(tx_train[i], r=1.5)
        tx_train[i] = outliers_to_nan(tx_train[i], r=r)
        tx_train[i] = nan_to_median(tx_train[i])
        tx_train[i] = standardize(tx_train[i])
        tx_train[i] = add_bias(tx_train[i])

    return tx_train, y_train, ids_train

def dataClean_without_splitting(tx, r=1.5):
    # tx, deleted = delete_useless_features(tx)
    tx = miss_to_nan(tx)
    # tx = replace_outliers(tx, r=1.5)
    tx = outliers_to_nan(tx, r=r)
    tx = nan_to_median(tx)
    tx = standardize(tx)
    tx = add_bias(tx)
        
    return tx #, deleted
