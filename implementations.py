# implementation of ML methods

import numpy as np


###############################################################
# ## HELPERS FUNCTIONS
# ##############################################################

def compute_gradient(y, tx, w):
    """Computes the gradient at w.

    Args:
        y: shape=(N, )
        tx: shape=(N,2)
        w: shape=(2, ). The vector of model parameters.

    Returns:
        An array of shape (2, ) (same shape as w), containing the gradient of the loss at w.
    """
    err = y - tx @ w
    grad = - (tx.T @ err) / len(err)
    return grad, err

def calculate_mse(e):
    """Calculate the mse for vector e."""
    return 1/2 * np.mean(e**2)

def sigmoid(t):
    """apply sigmoid function on t.

    Args:
        t: scalar or numpy array

    Returns:
        scalar or numpy array
    """
    return 1 / (1 + np.exp(-t))

def calculate_loss(y, tx, w):
    """compute the cost by negative log likelihood.

    Args:
        y:  shape=(N, 1)
        tx: shape=(N, D)
        w:  shape=(D, 1) 

    Returns:
        a non-negative loss
    """
    N = len(y)
    v = tx @ w
    return -1/N * np.sum(y * np.log(sigmoid(v)) + (1-y) * np.log(1-sigmoid(v)))

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
    
    return 1/N * tx.T @ (sigmoid(v) - y)


###############################################################
# ## STEP 2 - ML FUNCTIONS
# ##############################################################

def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using the Gradient Descent algorithm.
    
    Args:
        y:         numpy array of shape (N,), N is the number of samples.
        tx:        numpy array of shape (N,D), D is the number of features.
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD 
        gamma:     a scalar denoting the stepsize 
        
    Returns:
        w:    optimal weights, numpy array of shape(D,), D is the number of features
        loss: the loss, a scalar
    """
    w = initial_w
    w_array = [w]
    loss_array = []
    grad, err = compute_gradient(y, tx, w)
    loss_array.append(calculate_mse(err))
    
    for n_iter in range(max_iters):
        # compute gradient
        grad, err = compute_gradient(y, tx, w) 
        # compute loss and append it to array
        loss_array.append(calculate_mse(err))
        # update w by gradient descent
        w = w - (gamma * grad)
        w_array.append(w)
    
    return w_array[-1], loss_array[-1]

def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using Stochastic Gradient Descent algorithm.
    
    Args:
        y:         numpy array of shape (N,), N is the number of samples.
        tx:        numpy array of shape (N,D), D is the number of features.
        initial_w: shape=(2, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD 
        gamma:     a scalar denoting the stepsize 

    Returns:
        w:    optimal weights, numpy array of shape(D,), D is the number of features
        loss: the loss, a scalar
    """
    w = initial_w
    data_size = len(y)
    
    for n_iter in range(max_iters):
        indice = np.random.choice(np.arange(data_size))
        
        random_y = np.array([y[indice]])
        random_tx = np.array([tx[indice]])
        
        # compute gradient
        grad, err = compute_gradient(random_y, random_tx, w)

        # update w by stochastic gradient descent
        w = w - (gamma * grad)
        
    loss = calculate_mse(err)
    
    return w, loss
    
def least_squares(y, tx):
    """Calculate the least squares solution.
       returns mse, and optimal weights.
    
    Args:
        y: numpy array of shape (N,), N is the number of samples.
        tx: numpy array of shape (N,D), D is the number of features.
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    w = np.linalg.solve(tx.T @ tx, tx.T @ y)
    
    err = y - tx.dot(w)
    loss = calculate_mse(err)
    
    return w, loss

def ridge_regression(y, tx, lambda_): 
    """Calcule the ridge regression solution using normal equations.
       returns mse, and optimal weights.
    
    Args:
        y:       numpy array of shape (N,), N is the number of samples.
        tx:      numpy array of shape (N,D), D is the number of features.
        lambda_: paramter, a scalar
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    N = len(y)
    
    idmx = np.eye(tx.shape[1])
    X = tx.T @ tx + 2 * N * lambda_ * idmx
    
    w = np.linalg.solve(X, tx.T @ y)
    
    # loss should not include the penalty term
    loss_x = tx.T @ tx
    loss_w = np.linalg.solve(loss_x, tx.T @ y)
    err = y - tx @ loss_w
    loss = 1/2 * np.mean(err**2)
    
    # loss = np.sqrt(2 * compute_mse(y, tx, w)) 
    
    return w, loss

def logistic_regression(y, tx, initial_w, max_iters, gamma): 
    """Calcule the logistic regression solution using gradient descent or SGD.
       returns mse, and optimal weights.
    
    Args:
        y:         numpy array of shape (N,), N is the number of samples.
        tx:        numpy array of shape (N,D), D is the number of features.
        initial_w: the initial weights
        max_iters: the maximum iteration, scalar
        gamma:     the parameter, scalar
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    w = initial_w
    
    for iter in range(max_iters):
        grad = calculate_gradient(y, tx, w)
        w = w - gamma * grad
        
    loss = calculate_loss(y, tx, w)
        
    return w, loss
    
    
def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma): 
    """Calcule the regularized logistic regression solution using gradient descent or SGD.
       returns mse, and optimal weights.
    
    Args:
        y:         numpy array of shape (N,), N is the number of samples.
        tx:        numpy array of shape (N,D), D is the number of features.
        lambda_: paramter, a scalar
        initial_w: the initial weights
        max_iters: the maximum iteration, scalar
        gamma:     the parameter, scalar
    
    Returns:
        w: optimal weights, numpy array of shape(D,), D is the number of features.
        loss: scalar.
    """
    w = initial_w
    
    for iter in range(max_iters):
        gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
    
    loss = calculate_loss(y, tx, w) + lambda_ * np.sum(w**2)
        
    return w, loss
