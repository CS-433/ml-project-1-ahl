import numpy as np
from helper_functions import *


def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using the Gradient Descent algorithm.

    Args:
        y:         numpy array of shape (N,), N is the number of samples.
        tx:        numpy array of shape (N,D), D is the number of features.
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of GD
        gamma:     a scalar denoting the stepsize

    Returns:
        w:    optimal weights, numpy array of shape(D,), D is the number of features
        loss: the loss, a scalar
    """
    threshold = 1e-8
    w = initial_w
    loss = calculate_mse(y, tx, w)
    losses = [loss]

    for iter in range(max_iters):
        # compute gradient
        grad = compute_gradient(y, tx, w)
        # update w by gradient descent
        w = w - (gamma * grad)
        loss = calculate_mse(y, tx, w)

        if iter % 1 == 0:
            print(
                "Current iteration={i}, the loss={l}, the grad={we}".format(
                    i=iter, l=loss, we=np.linalg.norm(grad)
                )
            )

        losses.append(loss)

        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression using Stochastic Gradient Descent algorithm.

    Args:
        y:         numpy array of shape (N,), N is the number of samples.
        tx:        numpy array of shape (N,D), D is the number of features.
        initial_w: shape=(D, ). The initial guess (or the initialization) for the model parameters
        max_iters: a scalar denoting the total number of iterations of SGD
        gamma:     a scalar denoting the stepsize

    Returns:
        w:    optimal weights, numpy array of shape(D,), D is the number of features
        loss: the loss, a scalar
    """
    threshold = 1e-8
    w = initial_w
    data_size = len(y)
    loss = calculate_mse(y, tx, w)
    losses = [loss]

    for iter in range(max_iters):
        indice = np.random.choice(np.arange(data_size))

        random_y = np.array([y[indice]])
        random_tx = np.array([tx[indice]])

        # compute gradient
        grad = compute_gradient(random_y, random_tx, w)

        # update w by stochastic gradient descent
        w = w - (gamma * grad)

        # compute loss
        loss = calculate_mse(random_y, random_tx, w)

        if iter % 1 == 0:
            print(
                "Current iteration={i}, the loss={l}, the grad={we}".format(
                    i=iter, l=loss, we=np.mean(grad)
                )
            )

        losses.append(loss)

        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

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
    loss = calculate_mse(y, tx, w)

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
    loss = calculate_mse(y, tx, w)

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
    threshold = 1e-8
    w = initial_w
    grad = calculate_gradient(y, tx, w)
    loss = calculate_loss(y, tx, w)
    losses = [loss]
    
    print("Preiteration, the loss={l}, the grad={we}".format(l=loss, we=np.linalg.norm(grad)))

    for iter in range(max_iters):
        grad = calculate_gradient(y, tx, w)

        w = w - gamma * grad

        loss = calculate_loss(y, tx, w)

        if iter % 1 == 0:
            print(
                "Current iteration={i}, the loss={l}, the grad={we}".format(
                    i=iter, l=loss, we=np.linalg.norm(grad)
                )
            )

        losses.append(loss)

        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

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
    threshold = 1e-8
    w = initial_w
    loss = calculate_loss(y, tx, w)
    losses = [loss]

    for iter in range(max_iters):
        gradient = calculate_gradient(y, tx, w) + 2 * lambda_ * w
        w = w - gamma * gradient
        loss = calculate_loss(y, tx, w)

        losses.append(loss)

        # converge criterion
        if len(losses) > 1 and np.abs(losses[-1] - losses[-2]) < threshold:
            break

    return w, loss
