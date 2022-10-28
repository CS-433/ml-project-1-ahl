import numpy as np


###############################################################
# ## HELPERS FUNCTIONS
# ##############################################################

def compute_gradient(y, tx, w):
    """Computes the gradient at w.
    Args:
        y: shape=(N, )
        tx: shape=(N,D)
        w: shape=(2, ). The vector of model parameters.
    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx @ w
    grad = -(tx.T @ err) / len(err)
    return grad


def calculate_mse(y, tx, w):
    """Calculate the mse."""
    e = y - tx @ w
    return (1 / 2) * np.mean(e**2)


def sigmoid(t):
    """apply sigmoid function on t.
    Args:
        t: scalar or numpy array
    Returns:
        scalar or numpy array
    """
    # modified fuction to avoid overflows
    
    pos = np.where(t >= 0)
    neg = np.where(t<0)
    t[pos]=1/(1+np.exp(-t[pos]))
    t[neg]=np.exp(t[neg])/(1+np.exp(t[neg]))
    
    return t


def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
    Returns:
        a non-negative loss
    """
    N = y.shape[0]
    v = tx @ w
    #print("############################## v= ",v)    
    #sig = sigmoid(v)
    #print("############################## sig= ",sig)
    #log1 = np.log(sig)
    #log2 = np.log(1-sig)
    return -np.sum(y * np.log(sigmoid(v)) + (1 - y) * np.log(1-sigmoid(v)))/N


def calculate_gradient(y, tx, w):
    """compute the gradient of loss.
    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1)
    Returns:
        a vector of shape (D, 1)
    """
    N = len(y)
    v = tx @ w

    return 1 / N * tx.T @ (sigmoid(v) - y)

def predict(weights, dataset):
    """generates predictions given weights and a dataset"""
    
    y_pred = dataset @ weights
    y_pred[np.where(y_pred <= 1/2)] = 0
    y_pred[np.where(y_pred > 1/2)] = 1
    
    return y_pred

def predict_logistic(weights, dataset):
    """generates predictions with logistic regression given weights and a dataset"""
    
    y_pred = sigmoid(dataset @ weights)
    y_pred[np.where(y_pred <= 1/2)] = 0
    y_pred[np.where(y_pred > 1/2)] = 1
    y_pred[y_pred==0] = -1
    
    return y_pred