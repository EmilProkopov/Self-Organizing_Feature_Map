import numpy as np
import matplotlib.pyplot as plt


def cost_MSE(A, Y):
    """
    The MSE cost function.

    Parameters
    ----------
    A : numpy array
        probability vector corresponding to label predictions of shape:
        (1, number of examples)
    Y : numpy
        array true "label" vector of shape: (x, number of examples)

    Returns
    ----------
    cost : number
        MSE cost
    """
    m = Y.shape[1]
    return np.sum((A - Y)**2) / m


def cost_MSE_backward(A, Y):
    """
    Parameters
    ----------
    A : numpy array
        probability vector, output of the forward propagation
    Y : numpy array
        true lables

    Returns
    ----------
    derivative of the MSE cost function
    """
    m = Y.shape[1]
    return 2 * (A-Y) / m

def init_params(layer_num, layer_dims, activation_fns):
    """
    Parameters
    ----------
    layer_num : number
        number of layers in the network
    layer_dims : array of integers
        an array that contains the dimensions of each layer of the network
    activations : array of strings
        an array of strings with names of activation functions of each layer
    
    Returns
    -------
    params : dictionary
        dictionary containing parameters "layer_num", "W1", "b1", "act_fn1"...:
        later_num -- an integer
        Wl -- weight matrix of shape (layer_dims[l], layer_dims[l-1])
        bl -- bias vector of shape (layer_dims[l], 1)
        act_fn1 -- a string
    """
    
    params = {}
    
    params['layer_num'] = layer_num

    for l in range(1, layer_num+1):
        params['W' + str(l)] = np.random.randn(layer_dims[l],
                                                   layer_dims[l-1]) * 0.01
        params['b' + str(l)] = np.zeros((layer_dims[l], 1))
        params['act_fn' + str(l)] = activation_fns[l-1]
        
    return params


def linear_forward(A, W, b):
    """
    Linear part of a layer's forward propagation.

    Parameters
    ----------
    A : array or numpy array
        activations from previous layer of shape:
        (size of previous layer, number of examples)
    W : numpy array
        weights matrix: numpy array of shape:
        (size of current layer, size of previous layer)
    b : numpy array
        bias vector, numpy array of shape: (size of the current layer, 1)

    Returns
    -------
    Z : numpy array
        the input for the activation function
    cache : tuple
        a tuple containing "A", "W" and "b"
    """

    Z = np.dot(W, A) + b
    cache = (A, W, b)
    
    return Z, cache


def activation_forward(A_prev, W, b, activation):
    """
    Forward propagation for the LINEAR->ACTIVATION layer

    Parameters
    ----------
    A_prev : numpy array
        activations from previous layer of shape:
        (size of previous layer, number of examples)
    W : numpy array
        weights matrix: numpy array of shape:
        (size of current layer, size of previous layer)
    b : numpy array
        bias vector, numpy array of shape:
        (size of the current layer, 1)
    activation : string
        the activation to be used in this layer, a text string:
        "sigmoid" or "relu". "none" means that this layer doesn't have an
        activation function.

    Returns
    ----------
    A : numpy array
        the output of the activation function
    cache : tuple
        a tuple containing "linear_cache" and "activation_cache".
        stored for computing the backward pass efficiently
    """

    Z, linear_cache = linear_forward(A_prev, W, b)
    A = Z
    activation_cache = None
    
    cache = (linear_cache, activation_cache)

    return A, cache


def model_forward(X, params):
    """
    Forward propagation for the model
    
    Parameters
    ----------
    X : numpy array
        data, numpy array of shape (input size, number of examples)
    params : dictionary
        output of initialize_params()
    
    Returns
    ----------
    A : numpy array
        last post-activation value
    caches : array
        list of caches containing every cache of activation_forward()
        (there are layer_num-1 of them, indexed from 0 to layer_num-1)
    """

    caches = []
    A = X
    
    for l in range(1, params['layer_num']+1):
        A_prev = A 
        A, cache = activation_forward(A_prev,
                                      params['W{:d}'.format(l)],
                                      params['b{:d}'.format(l)],
                                      params['act_fn{:d}'.format(l)])
        caches.append(cache)
        
    return A, caches


def compute_cost(A, Y, cost_fn="cross-entrophy"):
    """
    The cost function.

    Parameters
    ----------
    A : numpy array
        probability vector corresponding to label predictions of shape:
        (1, number of examples)
    Y : numpy array
        true "label" vector of shape: (x, number of examples)
    cost_fn : string
        sting containing name of the cost function
    
    Returns
    ----------
    cost : number
    """

    return cost_MSE(A, Y)


def linear_backward(dZ, cache):
    """
    Linear portion of backward propagation for a single layer

    Parameters
    ----------
    dZ : numpy array
        gradient of the cost with respect to
        the linear output of current layer l
    cache : tuple
        tuple of values (A_prev, W, b) coming from the forward propagation
        in the current layer

    Returns
    ----------
    dA_prev : numpy array
        gradient of the cost with respect to the activation
        of the previous layer l-1 same shape as A_prev
    dW : numpy array 
        gradient of the cost with respect to W of the current layer,
        same shape as W
    db : numpy array 
        gradient of the cost with respect to b of the current layer,
        same shape as b
    """
    A_prev, W, b = cache
    m = A_prev.shape[1]

    dW = np.dot(dZ, A_prev.T) / m
    db = np.sum(dZ, axis=1, keepdims=True) / m
    dA_prev = np.dot(W.T, dZ)
    
    return dA_prev, dW, db


def activation_backward(dA, cache, activation):
    """
    Backward propagation for the LINEAR->ACTIVATION layer.
    
    Parameters
    ----------
    dA : numpy array 
        post-activation gradient for current layer l 
    cache : tuple
        tuple of values (linear_cache, activation_cache) stored for computing
        backward propagation efficiently
    activation : string
        the activation to be used in this layer: "sigmoid" or "relu"
    
    Returns
    ----------
    dA_prev : numpy array  
        gradient of the cost with respect to the activation of the previous
        layer, same shape as A_prev
    dW : numpy array
        gradient of the cost with respect to W of the current layer,
        same shape as W
    db : numpy array
        gradient of the cost with respect to b of the current layer,
        same shape as b
    """
    linear_cache, activation_cache = cache
    
    dZ = dA
    
    dA_prev, dW, db = linear_backward(dZ, linear_cache)
    
    return dA_prev, dW, db


def model_backward(A, Y, caches, activations, cost_fn="cross-entrophy"):
    """
    Backward propagation
    
    Parameters
    ----------
    A : numpy array
        probability vector, output of the forward propagation
    Y : numpy array
        true lables
    caches : numpy array
        list of caches

    Returns
    ----------
    grads : dictionary
        dictionary with the gradients
             grads["dA" + str(l)] 
             grads["dW" + str(l)]
             grads["db" + str(l)] 
    """

    grads = {}
    L = len(caches)
    Y = Y.reshape(A.shape)
    
    dA = cost_MSE_backward(A, Y)

    grads["dA" + str(L+1)] = dA
    
    for l in reversed(range(1, L+1)):
        current_cache = caches[l-1]
        dA_prev_temp, dW_temp, db_temp = activation_backward(
                grads["dA" + str(l + 1)],
                current_cache,
                activations[l-1])
        grads["dA" + str(l)] = dA_prev_temp
        grads["dW" + str(l)] = dW_temp
        grads["db" + str(l)] = db_temp

    return grads


def update_params(params, grads, learning_rate):
    """
    Update parameters using gradient descent
    
    Parameters
    ----------
    params : dictionary
        dictionary containing the parameters 
    grads : dictionary
        dictionary containing the gradients, output of model_backward
    
    Returns
    ----------
    params : dictionary
        dictionary containing the updated parameters 
    """
    for l in range(1, params['layer_num']+1):
        params["W" + str(l)] = params["W" + str(l)] - learning_rate * grads["dW" + str(l)]

        params["b" + str(l)] = params["b" + str(l)] - learning_rate * grads["db" + str(l)]

    return params


def train_model(X,
                Y,
                layers_dims,
                activations,
                learning_rate = 0.001,
                num_iterations = 3000,
                print_cost=False,
                cost_function="cross-entrophy"):
    """
    Train a neural network
    
    Parameters
    ----------
    X : numpy array
        training data
    Y : numpy array
        correct outputs for training data
    layers_dims : array
        list containing the input size and each layer size
    activations : array
        array of strings with names of activation functions for each layer
    learning_rate : number
        learning rate of the gradient descent
    num_iterations : number
        number of the learning iterations
    print_cost : boolean
        if True print the cost every 100 steps
    cost_function : string
        "cross-entrophy" or "MSE"
    
    Returns
    ----------
    params : tuple
        parameters learnt by the model. Ready to be used for prediction
    """

    costs = []

    params = init_params(len(layers_dims)-1, layers_dims, activations)

    for i in range(0, num_iterations):
        A, caches = model_forward(X, params)
        cost = compute_cost(A, Y, cost_function)
        grads = model_backward(A, Y, caches, activations, cost_function)
        
        params = update_params(params, grads, learning_rate)
                
        if print_cost and i % 100 == 0:
            print ("Cost after iteration %i: %f" %(i, cost))
            costs.append(cost)
            
    if print_cost:
        plt.plot(np.squeeze(costs))
        plt.ylabel('cost')
        plt.xlabel('iterations (per hundreds)')
        plt.title("Learning rate =" + str(learning_rate))
        plt.show()
    
    return params


def predict(X, params):
    """
    Prediction
    
    Parameters
    ----------
    X : numpy array
        data, numpy array of shape (input size, number of examples)
    params : dictionary
        params of trained model
    
    Returns
    ----------
    A : numpy array
        last post-activation value
    """
    
    A, _ = model_forward(X, params)
    return A