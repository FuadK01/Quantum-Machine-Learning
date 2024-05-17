import QCNN_circuit
import pennylane as qml
from pennylane import numpy as np
import numpy as np0
import autograd.numpy as anp
import pickle
import datetime
import os
from tqdm import tqdm
import Benchmarking


def cross_entropy(labels, predictions):  
    """
    Calculates cross entropy loss for multiclass classification
    Parameters
    ----------
    labels : list, int
        True labels (ground truth)
    predictions : list, float
        Model predictions (output from QCNN)

    Returns
    -------
    float
        Cross Entropy Loss
    """

    epsilon = 1e-15  # small constant to avoid log(0)

    # Clip predictions to avoid log(0) or log(1)
    predictions = np0.clip(predictions, epsilon, 1 - epsilon)
    labels = np0.eye(4)[np0.asarray(labels)]

    # Compute cross entropy
    loss = np0.sum(labels * np0.log(predictions)) / len(labels)
    
    return -1 * loss

def least_squares(labels, predictions):
    """
    Finds least squares loss for multiclass classification
    Parameters
    ----------
    labels : list, int
        True labels (ground truth)
    predictions : list, float
        Model predictions (output from QCNN)

    Returns
    -------
    float
        Least squares loss
    """

    predictions = np.array(predictions)
    labels = np.eye(4)[anp.asarray(labels)]

    loss = np.sum((predictions - labels)**2) / len(labels)

    return loss


def cost(params, X, Y, U, U_params):
    """
    Computes total cost of QCNN over training set

    Parameters
    ----------
    params : list, float
        Tunable paramters of QCNN
    X : list, float
        Embedded input
    Y : list, int
        Labels
    U : string
        Unitary used in QCNN
    U_params : int
        Number of parameters used to train circuit

    Returns
    -------
    float
        Total loss
    """

    # Run QCNN Circuit to gather model predictions
    predictions = [QCNN_circuit.QCNN(x, params, U, U_params) for x in X]

    # Compute loss
    loss = cross_entropy(Y, predictions)

    return loss

def Validation_Acc(params, X, Y, U, U_params):
    """
    Computes the validation accuracy of the dataset throughout training

    Parameters
    ----------
    params : list, float
        Tunable paramters of QCNN
    X : list, float
        Embedded input
    Y : list, int
        Labels
    U : string
        Unitary used in QCNN
    U_params : int
        Number of parameters used to train circuit

    Returns
    -------
    float
        Validation accuracy
    """

    acc = 0

    predictions = [QCNN_circuit.QCNN(x, params, U, U_params) for x in X]

    for i in range(len(Y)):
        predicted_label = np.argmax(predictions[i])
        
        #if predictions[i][predicted_label] < 0.50:
        #    predicted_label = 0
        
        if predicted_label == Y[i]:
            acc += 1   

    return acc / len(Y)

def circuit_training(X_train, X_val, Y_train, Y_val, U, U_params, steps, testName, learning_rate, batch_size):
    """
    Trains the QCNN Circuit using training data

    Parameters
    ----------
    X_train, X_val : list, float
        Training/Validation data
    Y_train, Y_val : list, int
        Training/Validation labels
    U : string
        Unitary used in QCNN
    U_params : int
        Number of parameters used to train circuit
    steps : int
        Number of training loops to be performed for optimisation (epochs)
    testName : string
        File location for saving results 
    learning_rate : float
        Value used for the rate of change of variable during training
    batch_size : int
        Number of datasets used in each training loop

    Returns
    -------
    float
        Loss & Validation history, with trained parameters
    """

    smallest = float('inf')

    # Calculating Variable Count
    if U == 'U_SU4_no_pooling' or U == 'U_SU4_1D' or U == 'U_9_1D':
        total_params = U_params * 3
    elif U == 'Large':
        total_params = U_params
    else:
        total_params = U_params * 13 + 4 * 3 + 15  # This is the usual calculation for paramter account. This includes 13 repetitions \\
                                                   # of the Unitary, 4 pooling filters and a final 15 paramter densely connected layer
    
    params = np.random.randn(total_params, requires_grad=True) # Randomly initialises circuit parameters

    print("Paramter count is, " + str(len(params)) + " \n")
    
    # Defining optimizer
    opt = qml.AdamOptimizer(stepsize=learning_rate)
    
    # Initalising history lists
    loss_history = []
    val_accuracy = []

    # Saving model to QCNN_Models folder
    try:
        path = "C:\\Users\\Fuad K\\Desktop\\Physics\\5th Year\\My_QCNN\\QCNN_Models"
        os.mkdir(path)
    except:
        print("File "+path+"already created")

    # Initalising progress bar to monitor training progress    
    pbar = tqdm(total=steps)


    no_improvement = 1
    best_params = 0

    # QCNN Training Loop
    for it in range(steps):

        # Creating batches of data
        batch_index = np.random.randint(0, len(X_train), (batch_size,))
        X_batch = [X_train[i] for i in batch_index]
        Y_batch = [Y_train[i] for i in batch_index]

        # Cost function is called which then calls the quantum cicuit
        params, cost_new = opt.step_and_cost(lambda v: cost(v, X_batch, Y_batch, U, U_params), params)
        loss_history.append(cost_new)

        # Compute validation accuracy
        val_accuracy.append(Validation_Acc(params, X_batch, Y_batch, U, U_params))

        # Saving paramters corresponding to lowest cost
        if cost_new < smallest:
            print(" Saving current parameters... \n")
            #currentfile = path+"\\model"+str(testName)+str(it)+"C"+str(cost_new)+".pkl"
            #pickle.dump(params, open(currentfile,'wb'))
            smallest = cost_new
            no_improvement = 1
            best_params = params
            print("iteration: ", it, " cost: ", cost_new)
        else:
            #print("No cost improvement")
            no_improvement += 1    
        
        pbar.update(1)
    
    # Saving trained parameters
    currentfile = path+"\\model"+str(testName)+"_"+U+"_"+"C"+str(cost_new)+".pkl"
    pickle.dump(best_params, open(currentfile,'wb'))
    
    return loss_history, best_params, val_accuracy