import numpy as np


def binary_train(X, y, loss="perceptron", w0=None, b0=None, step_size=0.5, max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: binary training labels, a N dimensional numpy array where 
    N is the number of training points, indicating the labels of 
    training data
    - loss: loss type, either perceptron or logistic
    - step_size: step size (learning rate)
	- max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of logistic or perceptron regression
    - b: scalar, which is the bias of logistic or perceptron regression
    """
    N, D = X.shape
    assert len(np.unique(y)) == 2


    w = np.zeros(D)
    if w0 is not None:
        w = w0
    
    b = 0
    if b0 is not None:
        b = b0

    if loss == "perceptron":
        ############################################
        # TODO 1 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        for i in range(max_iterations):
            gradient = 0
            error = 0
            y[np.where(y == 0)] = -1
            Z = -y*(np.dot(X, w) + b)
            preds = np.where(Z>=0, 1, 0)
            gradient = -np.dot(y*preds, X)
            error = np.sum(y*preds)
            w -= step_size*(gradient/len(X))
            b += step_size*(error/len(X))
#        print ("W: ", w)
#        print ("B: ", b)
        y[np.where(y == -1)] = 0
        
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 2 : Edit this if part               #
        #          Compute w and b here            #
        w = np.zeros(D)
        b = 0
        for i in range(max_iterations):
            gradient = 0
            error = 0
            Z = np.dot(X, w) + b
            preds = np.where(sigmoid(Z), sigmoid(Z), 0)
            gradient = np.dot((preds-y), X)
            error = -np.sum(preds - y)
            w -= step_size*(gradient/len(X))
            b += step_size*(error/len(X))  
#        print ("W: ", w)
#        print ("B: ", b)

        ############################################
        

    else:
        raise "Loss Function is undefined."

    assert w.shape == (D,)
    return w, b

def sigmoid(z):
    
    """
    Inputs:
    - z: a numpy array or a float number
    
    Returns:
    - value: a numpy array or a float number after computing sigmoid function value = 1/(1+exp(-z)).
    """

    ############################################
    # TODO 3 : Edit this part to               #
    #          Compute value                   #
    value = 1/ (1 + np.exp(-z))
    ############################################
    
    return value

def binary_predict(X, w, b, loss="perceptron"):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: D-dimensional vector, a numpy array which is the weight 
    vector of your learned model
    - b: scalar, which is the bias of your model
    - loss: loss type, either perceptron or logistic
    
    Returns:
    - preds: N dimensional vector of binary predictions: {0, 1}
    """
    N, D = X.shape
    
    if loss == "perceptron":
        ############################################
        # TODO 4 : Edit this if part               #
        #          Compute preds                   #
        Z = np.dot(X, w) + b
        preds = np.where(Z>0, 1, 0)
        ############################################
        

    elif loss == "logistic":
        ############################################
        # TODO 5 : Edit this if part               #
        #          Compute preds                   #
        Z = np.dot(X, w) + b
        preds = np.where(sigmoid(Z) > 0.5, 1, 0)
        ############################################
        

    else:
        raise "Loss Function is undefined."
    

    assert preds.shape == (N,) 
    return preds



def multiclass_train(X, y, C,
                     w0=None, 
                     b0=None,
                     gd_type="sgd",
                     step_size=0.5, 
                     max_iterations=1000):
    """
    Inputs:
    - X: training features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - y: multiclass training labels, a N dimensional numpy array where
    N is the number of training points, indicating the labels of 
    training data
    - C: number of classes in the data
    - gd_type: gradient descent type, either GD or SGD
    - step_size: step size (learning rate)
    - max_iterations: number of iterations to perform gradient descent

    Returns:
    - w: C-by-D weight matrix of multinomial logistic regression, where 
    C is the number of classes and D is the dimensionality of features.
    - b: bias vector of length C, where C is the number of classes
    """

    N, D = X.shape

    w = np.zeros((C, D))
    if w0 is not None:
        w = w0
    
    b = np.zeros(C)
    if b0 is not None:
        b = b0

    np.random.seed(42)
    if gd_type == "sgd":
        ############################################
        # TODO 6 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        new_y = np.zeros((N, C))
        new_y[np.arange(len(X)), y] = 1
        for i in range(max_iterations):
            idx = np.random.choice(N)
            preds = get_softmax(X[idx], w, b)
            gradient = -np.dot((new_y[idx]-preds).reshape(C, 1), X[idx].reshape(D, 1).T)
            error = (new_y[idx]-preds)
            w -= step_size*(gradient)
            b += step_size*(error)
#        print ("W: ", w)
#        print ("B: ", b)
        ############################################
        

    elif gd_type == "gd":
        ############################################
        # TODO 7 : Edit this if part               #
        #          Compute w and b                 #
        w = np.zeros((C, D))
        b = np.zeros(C)
        new_y = np.zeros((N, C))
        new_y[np.arange(len(X)), y] = 1
        for i in range(max_iterations):
            preds = get_softmax_gd(X, w, b)
            gradient = np.dot((preds - new_y).T, X)
            error = -np.sum((preds - new_y), axis = 0)
            w -= step_size*(gradient/N)
            b += step_size*(error/N)
#        print ("W: ", w)
#        print ("B: ", b)
        ############################################
        

    else:
        raise "Type of Gradient Descent is undefined."
    

    assert w.shape == (C, D)
    assert b.shape == (C,)

    return w, b


def multiclass_predict(X, w, b):
    """
    Inputs:
    - X: testing features, a N-by-D numpy array, where N is the 
    number of training points and D is the dimensionality of features
    - w: weights of the trained multinomial classifier, C-by-D 
    - b: bias terms of the trained multinomial classifier, length of C
    
    Returns:
    - preds: N dimensional vector of multiclass predictions.
    Outputted predictions should be from {0, C - 1}, where
    C is the number of classes
    """
    N, D = X.shape
    ############################################
    # TODO 8 : Edit this part to               #
    #          Compute preds                   #
    probabs = get_softmax(X, w, b)
    preds = np.argmax(probabs, axis = 1)
    ############################################

    assert preds.shape == (N,)
    return preds

    
def get_softmax(X, w, b):
    Z = np.dot(X, w.T) + b
    Z = Z - np.max(Z)
    numerator = np.exp(Z)
    denominator = np.sum(numerator)
    value = numerator/denominator
    return value

def get_softmax_gd(X,w,b):
    Z = np.dot(X, w.T) + b
    Z = Z - np.max(Z)
    numerator = np.exp(Z.T)
    denominator = np.sum(np.exp(Z), axis = 1)
    value = numerator/denominator
    return value.T    