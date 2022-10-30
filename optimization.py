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

def cross_validation(y, tx, k_indices, k, lambda_, degree):
    """return the loss of ridge regression for a fold corresponding to k_indices
    
    Args:
        y:          shape=(N,)
        x:          shape=(N,)
        k_indices:  2D array returned by build_k_indices()
        k:          scalar, the k-th fold (N.B.: not to confused with k_fold which is the fold nums)
        lambda_:    scalar, cf. ridge_regression()
        degree:     scalar, cf. build_poly()

    Returns:
        train and test mean square errors
    """
    te_ind = k_indices[k]
    tr_ind = k_indices[~(np.arange(k_indices.shape[0])==k)]
    tr_ind = tr_ind.reshape(-1)
    y_te   = y[te_ind]
    y_tr   = y[tr_ind]
    tx_te  = tx[te_ind]
    tx_tr  = tx[tr_ind]
    tx_tr = build_poly(tx_tr, degree)
    tx_te = build_poly(tx_te, degree)
    w,loss_tr = ridge_regression(y_tr, tx_tr, lambda_)
    _,loss_te = ridge_regression(y_te,tx_te, lambda_)
    
    return loss_tr, loss_te,w

def best_lambda_degree(y, tx, k_fold, lambdas, degrees, seed):
    """cross validation over regularisation parameter lambda and degree.
    
    Args:
        degrees: shape = (d,), where d is the number of degrees to test 
        k_fold: integer, the number of folds
        lambdas: shape = (p, ) where p is the number of values of lambda to test
    Returns:
        best_degree : integer, value of the best degree
        best_lambda : scalar, value of the best lambda
        
    """
    k_indices = build_k_indices(y, k_fold, seed)
    best_lambdas = []
    best_rmses   = []
    
    for degree in degrees : 
        rmse_te = []
        for lambda_ in lambdas : 
            rmse_temp = []
            for k in range(k_fold): 
                _, loss_te,_ = cross_validation(y, tx, k_indices, k, lambda_, degree)
                rmse_temp.append(loss_te)
            rmse_te.append(np.mean(rmse_temp))
        
        indice_lambda = np.argmin(rmse_te)
        best_lambdas.append ( lambdas[indice_lambda])
        best_rmses.append( rmse_te[indice_lambda])
    
    indice_deg = np.argmin(best_rmses)
    best_lambda = best_lambdas[indice_deg]
    best_degree = degrees[indice_deg]
    
    return  best_lambda, best_degree


def standardize(dataset):
    """normalizes the dataset by substracing it by its mean and dividing it its standard deviation."""
    mean = np.mean(dataset, axis=0)
    std = np.std(dataset, axis=0)
    standardize_dataset = (dataset - mean)/std

    return standardize_dataset

def split_train(tx, y):
    """splits the dataset into four parts according to the feature PRI_jet_num. (column number 22 of the features)."""
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
    """adds a vector of ones as a bias to our dataset."""
    return np.c_[np.ones((tx.shape[0],1)), tx]
        
def remove_useless_features(tx):
    """removes the features from the dataset where the standard deviation is 0."""
    
    std_ = np.std(tx, axis=0) 
    indx = np.where(std_ == 0)[0]
    tx = np.delete(tx, indx, 1)
    
    return tx, indx

def miss_to_nan(tx):
    """transfroms the missing values (-999) to NaN values."""
    return np.where(tx > -999, tx, np.nan)

def replace_outliers(tx, r):
    """replaces the outliers by the median."""
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
    """transfroms the outliers by NaN values."""
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
    """replaces the NaN values by the median."""
    tx_clean = np.copy(tx.T)
    
    for i in range(tx_clean.shape[0]):
        line = tx.T[i]        
        tx_clean[i] = np.where(~np.isnan(line), line, np.nanmedian(line))

    return tx_clean.T

def useless_features_to_one(tx):
    """replaces the features with a zero standard deviation to one."""
    all_std = np.std(tx, axis=0)
    idx = [i for i, std in enumerate(all_std) if std==0]
    tx[:, idx] = 1
    return tx

#calculates the nb of corrected claissification
def calculate_accuracy(true_pred, y_pred): 
    """computes the accuracy based on the true prediction and the prediction computed by the resuts of our training model."""
    nb_true = np.sum(y_pred == true_pred)
    return nb_true/len(true_pred)

def dataClean(tx, y, r=1.5): 
    """splits the dataset into 4 parts, removes the outliers, replaces them by the median, normalizes the dataset and adds a bias."""
    tx_train, y_train, ids_train = split_train(tx, y)
    
    for i in range(4): 
        tx_train[i], _ = remove_useless_features(tx_train[i])
        tx_train[i] = miss_to_nan(tx_train[i])
        tx_train[i] = outliers_to_nan(tx_train[i], r=r)
        tx_train[i] = nan_to_median(tx_train[i])
        tx_train[i] = standardize(tx_train[i])
        tx_train[i] = add_bias(tx_train[i])

    return tx_train, y_train, ids_train

def dataClean_without_splitting(tx, r=1.5):
    """removes the outliers, replaces them by the median, normalizes the dataset and adds a bias."""
    tx = miss_to_nan(tx)
    tx = outliers_to_nan(tx, r=r)
    tx = nan_to_median(tx)
    tx = standardize(tx)
    tx = add_bias(tx)
        
    return tx
