import numpy as np
import torch
from torch import nn
from torch.optim import SGD
from torch.utils.data import DataLoader, TensorDataset

def multi_layer_nn_torch(x_train, y_train, layers, activations, alpha=0.01, batch_size=32, epochs=0, loss_func='mse', val_split=(0.8, 1.0), seed=7321):
    # This function creates and trains a multi-layer neural Network in PyTorch
    # X_train: numpy array of input for training [nof_train_samples,input_dimensions]
    # Y_train: numpy array of desired outputs for training samples [nof_train_samples,output_dimensions]
    # layers: Either a list of integers or alist of numpy weight matrices.
    # If layers is a list of integers then it represents number of nodes in each layer. In this case
    # the weight matrices should be initialized by random numbers.
    # If the layers is given as a list of weight matrices (numpy array), then the given matrices should be used and NO random
    # initialization is needed.
    # activations: list of case-insensitive activations strings corresponding to each layer. The possible activations
    # are, "linear", "sigmoid", "relu".
    # alpha: learning rate
    # epochs: number of epochs for training.
    # loss_func: is a case-insensitive string determining the loss function. The possible inputs are: "svm" , "mse",
    # "CrossEntropy". Do not use any PyTorch provided methods to compute loss. Implement the equations by yourself.
    # validation_split: a two-element list specifying the normalized start and end point to
    # extract validation set. Use floor in case of non integers.

    # return: This function should return a list containing 3 elements:
        # The first element of the return list should be a list of weight matrices.
        # Each element of the list should be a 2-d numpy array which corresponds to the weight matrix of the
        # corresponding layer (Biases should be included in each weight matrix in the first row).

        # The second element should be a one dimensional list of numbers
        # representing the error after each epoch. You should compute the mean-absolute error between the target and the prediction.
        # Be careful to not mix-up loss-function with error. Each error should
        # be calculated by using the entire validation set while the network is frozen.
        # Frozen means that the weights should not be adjusted while calculating the error.
        # In case of epochs == 0, do not compute error, instead return an empty list.

        # The third element should be a two-dimensional numpy array [nof_validation_samples,output_dimensions]
        # representing the actual output of the network when validation set is used as input.

    # Notes:

    # DO NOT use any other package other than PyTorch and numpy
    # Bias should be included in the weight matrix in the first row.
    # Use steepest descent for adjusting the weights
    # Use minibatch to calculate error and adjusting the weights
    # Do not use any random number seeding. The test case will take care of the random number seeding.
    # Use numpy for weight to initialize weights. Do not use PyTorch weight initialization.
    # Do not use any random method from PyTorch
    # Do not shuffle data
    # i.e., Initialize the weights for each layer by:
    # np.random.seed(seed)
    # np.random.randn()
    # Runtime for all the test cases will be less than 5 seconds

    def fwd_activation(Z, name):
        if name.lower() == 'linear':
            return Z 
        if name.lower() == 'sigmoid':
            return 1/(1 + torch.exp(-Z))
        if name.lower() == 'relu':
            return torch.clamp(Z, min=0.0)
    
    def bwd_activation(dA, Z, a, name):
        if name.lower() == 'linear':
            return dA 
        elif name.lower() == 'sigmoid':
            return dA * (a * (1.0 -a))
        elif name.lower() == 'relu':
            return dA * (Z>0).float()
    
    def calculate_loss_and_grad(y_pred, y_true, name):

        B = y_pred.shape[0]
        C = y_pred.shape[1]
        if name.lower() == 'mse':
            diff = y_pred - y_true 
            loss = torch.mean(diff*diff)
            n = diff.numel()
            dY = (2.0 * diff)/n 
        elif name.lower() == 'svm':
            margin = 1.0 - (y_pred*y_true)
            mask = (margin > 0).float()
            loss = torch.mean(torch.clamp(margin, min = 0.0))
            n = y_pred.numel()
            dY = (-y_true*mask)/n
            return loss, dY
        elif name.lower() == 'crossentropy':
            eps = 1e-12
            if y_true.shape[1] == 1 and C > 1:
                idx = y_true.long().squeeze(1)
                y_onehot = torch.zeros((B,C), dtype = y_pred.dtype) ##
                y_onehot.scatter_(1, idx.unsqueeze(1), 1.0)
            else:
                y_onehot = y_true
            
            logits = y_pred
            logits_shift = logits  - torch.max(logits, dim=1, keepdim = True).values
            exp_logits = torch.exp(logits_shift)
            softmax = exp_logits / (torch.sum(exp_logits, dim=1, keepdim=True) + eps)
            loss = -torch.mean(torch.sum(y_onehot * torch.log(softmax + eps), dim=1))
            dY = (softmax - y_onehot)/B

        return loss, dY
    

    X = torch.tensor(x_train, dtype = torch.float32)
    Y = torch.tensor(y_train, dtype = torch.float32)

    n = X.shape[0] #no. of samples
    inp_dim = X.shape[1] #input_dimension

    #For validation
    v_start = int(np.floor(val_split[0] * n))
    v_end = int(np.floor(val_split[1] * n))
    v_start = max(0, min(n, v_start))
    v_end = max(0, min(n, v_end))

    X_val = X[v_start:v_end]
    Y_val = Y[v_start:v_end]

    X_tr = torch.cat([X[:v_start], X[v_end:]], dim=0) if (v_start < v_end) else X
    Y_tr = torch.cat([Y[:v_start], Y[v_end:]], dim=0) if (v_start < v_end) else Y

    weights = []

    if isinstance(layers[0], (int, np.integer)):
        prev = inp_dim

        for out_dim in layers:
            np.random.seed(seed)
            W_np = np.random.randn(prev+1, out_dim).astype(np.float32)
            weights.append(torch.tensor(W_np, dtype=torch.float32))
            prev = out_dim
    else:
        for W_np in layers:
            W_np = np.array(W_np, dtype = np.float32)
            W = torch.tensor(W_np, dtype=torch.float32)
            weights.append(W)
    
    def forward_pass(Xbatch):
        caches = []  
        A = Xbatch
        for i, W in enumerate(weights):
            ones = torch.ones((A.shape[0], 1), dtype=A.dtype)
            Xb = torch.cat([ones, A], dim=1)          
            Z = Xb @ W                                
            A_next = fwd_activation(Z, activations[i])   

            caches.append((Xb, Z, A_next))
            A = A_next
        return A, caches
    
    def backward_and_update(dA_last, caches):
        dA = dA_last
        
        for i in reversed(range(len(weights))):
            Xb, Z, A = caches[i]
            
            dZ = bwd_activation(dA, Z, A, activations[i])

            B = Xb.shape[0]
            dW = Xb.t() @ dZ

            W_old = weights[i]              
            weights[i] = weights[i] - alpha * dW
            dXb = dZ @ W_old.t()  
            dA = dXb[:, 1:]  

    errors = []

    if epochs > 0:
        num_train = X_tr.shape[0]
        for epoch in range(epochs):

            for start in range(0, num_train, batch_size):
                end = min(num_train, start + batch_size)
                xb = X_tr[start:end]
                yb = Y_tr[start:end]

                y_pred, caches = forward_pass(xb)
                loss, dY = calculate_loss_and_grad(y_pred, yb, loss_func)

                backward_and_update(dY, caches)

            if X_val.shape[0] > 0:
                with torch.no_grad():
                    val_pred, _ = forward_pass(X_val)
                    mae = torch.mean(torch.abs(val_pred - Y_val)).item()
                errors.append(mae)
            else:
                errors.append(0.0)
    
    with torch.no_grad():
        y_val_pred, _ = forward_pass(X_val) if X_val.shape[0] > 0 else (torch.empty((0, Y.shape[1])), [])
    y_val_pred_np = y_val_pred.cpu().numpy()


    weights_np = [W.detach().cpu().numpy() for W in weights]

    return [weights_np, errors if epochs > 0 else [], y_val_pred_np]