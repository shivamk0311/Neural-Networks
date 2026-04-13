import numpy as np

def multi_layer_nn(X_train,Y_train,X_test,Y_test,layers,alpha,epochs,h=0.00001,seed=2):
    # This function creates and trains a multi-layer neural network.
    #
    # Parameters:
    # X_train : numpy.ndarray
    #     Training input data of shape [num_train_samples, input_dimension].
    #
    # Y_train : numpy.ndarray
    #     Desired training outputs of shape [num_train_samples, output_dimension].
    #
    # X_test : numpy.ndarray
    #     Testing input data of shape [num_test_samples, input_dimension].
    #
    # Y_test : numpy.ndarray
    #     Desired testing outputs of shape [num_test_samples, output_dimension].
    #
    # layers : list of int
    #     A list specifying the number of neurons in each layer.
    #
    # alpha : float
    #     Learning rate for gradient descent.
    #
    # epochs : int
    #     Number of training epochs.
    #
    # h : float
    #     Step size used for centered-difference approximation.
    #
    # seed : int
    #     Seed for the random number generator used to initialize weights.
    #
    # Returns:
    # A list with three elements:
    #
    # 1) weights : list of numpy.ndarray
    #    A list of 2D weight matrices, one per layer. Each matrix includes the bias
    #    in its first row.
    #
    # 2) mse_history : numpy.ndarray
    #    A 1D array containing the average mean-squared error (MSE) after each
    #    epoch. The MSE is computed using X_test while the network weights are
    #    frozen (no weight updates during evaluation).
    #
    # 3) Y_pred : numpy.ndarray
    #    A 2D array of shape [num_test_samples, output_dimension] representing
    #    the network output for X_test.
    #
    # Notes:
    # - Do NOT use any external packages other than NumPy.
    # - Bias terms must be incorporated into the weight matrices as the first row.
    # - The net input is computed as: net = X · W, where the bias is included in W.
    # - All layers use the sigmoid activation function except the output layer.
    # - The output layer uses a linear activation function.
    # - Mean-squared error (MSE) is used as the loss function.
    # - Weights are updated using gradient descent:
    #       W = W − alpha * (∂E / ∂W)
    # - Partial derivatives must be computed using the centered-difference method:
    #       (f(x + h) − f(x − h)) / (2h)
    # - Re-seed the random number generator before initializing each layer’s weights:
    #       np.random.seed(seed)
    #       np.random.randn(...)

    def sigmoid(x):
        return 1/(1 + np.exp(-x))
    
    def add_ones_col(X):
        return np.hstack([np.ones((X.shape[0],1)),X])
    
    W = []
    input_dimensions = X_train.shape[1]
    prev_layer = input_dimensions

    for i in range(len(layers)):
        np.random.seed(seed)
        weights = np.random.randn(prev_layer+1, layers[i])
        W.append(weights)
        prev_layer = layers[i]
    
    def calculate_raw_net(X, W):
        A = X
        
        for k in range(len(W)):
            Xb = add_ones_col(A)
            net  = np.dot(Xb,W[k])
            if k < len(W) - 1:
                A = sigmoid(net)
            else:
                A = net
        return A
    
    def mse(Y, Y_net):
        return np.mean((Y-Y_net)**2)
    
    def calculate_gradient(X, Y, W, k, h):
        weights = W[k]
        grad = np.zeros_like(weights)

        for r in range(weights.shape[0]):
            for c in range(weights.shape[1]):
                old = weights[r,c]

                weights[r,c] = old + h
                Yp = calculate_raw_net(X, W)
                err_plus = mse(Y, Yp)

                weights[r,c] = old - h
                Ym = calculate_raw_net(X, W)
                err_minus = mse(Y, Ym)

                grad[r,c] = (err_plus - err_minus)/(2*h)
                weights[r,c] = old
        
        return grad

    mse_history = []
    
    for epoch in range(epochs):

        for i in range(X_train.shape[0]):
            grads = []
            Xi = X_train[i:i+1, :]
            Yi = Y_train[i:i+1, :]
            for k in range(len(W)):
                grad_k = calculate_gradient(Xi, Yi, W, k, h)
                grads.append(grad_k)

            for k in range(len(W)):
                W[k] -= alpha * grads[k]
                
        Y_net = calculate_raw_net(X_test,W)
        mse_history.append(mse(Y_test, Y_net))
    
    Y_pred = calculate_raw_net(X_test, W)
    mse_history = np.array(mse_history)
    
    return W , mse_history, Y_pred
