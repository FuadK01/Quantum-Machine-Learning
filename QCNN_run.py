import Benchmarking
import numpy as np
import datetime
import os
import csv

"""
Here are possible combinations of benchmarking user could try.
Unitaries: ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4', 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
U_num_params: [2, 10, 10, 2, 6, 6, 4, 6, 15, 15, 15, 2]

circuit: 'QCNN' 
cost_fn: 'cross_entropy'
"""

# Constant declarations
EPOCHS = 250
LEARNING_RATE = 0.001
BATCH_SIZE = 16
SNR = 0

if __name__ == "__main__":
    # Choosing Unitary
    Unitaries= ['U_TTN', 'U_5', 'U_6', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4']#, 'U_SU4_no_pooling', 'U_SU4_1D', 'U_9_1D']
    U_num_params = {'U_TTN': 2, 'U_6': 10, 'U_5': 10, 'U_9': 2, 'U_13': 6, 'U_14': 6, 'U_15': 4, 'U_SO4': 6, 'U_SU4': 15}#, 15, 15, 2]{'U_6': 10, 'U_5': 10, 'U_9': 2, 'U_13': 6, 'U_14': 6, 'U_15': 4, 'U_SO4': 6, 'U_SU4': 15}
    U_learning = {'U_TTN': 0.001, 'U_6': 0.0005, 'U_5': 10, 'U_9': 2, 'U_13': 6, 'U_14': 6, 'U_15': 4, 'U_SO4': 6, 'U_SU4': 15}#, 15, 15, 2]{'U_6': 10, 'U_5': 10, 'U_9': 2, 'U_13': 6, 'U_14': 6, 'U_15': 4, 'U_SO4': 6, 'U_SU4': 15}

    # Set filename for saving results of training
    filename = "Result"+str(datetime.datetime.now().date()) + '_' + str(datetime.datetime.now().time()).replace(':', '.')
    #Input name of quantum data file to be trained on
    Qdata = "Qdata30__f1_20_21__f2_50_51__f3_100_101_10_11"
    fname = "C:\\Users\\Fuad K\\Desktop\\Physics\\5th Year\\My_QCNN\\Quantum_Data\\"+ Qdata + ".csv"
    testName = "SNR" + str(SNR) + "_LR" + str(LEARNING_RATE) + "_"
    
    
    labels = []
    waves = []

    with open(fname, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=',')
        for row in reader:
            wave = np.array(eval(row[0].strip(" ").replace("j", 'j,')))  # Extract wave state from row, excluding the label
            label = int(row[-1])  # Extract label from the last element of the row
            waves.append(wave)
            labels.append(label)

    dataset=[waves,labels]

    #snr? Frequences
    #was 0.1
    freqs=[1]
    print(' Data reading complete... \n')
    #print(freqs)
    #print("test",testName,"\n")
    #print("Running QCNN...")
    
    Unitaries= ['U_6', 'U_9', 'U_14', 'U_15', 'U_SO4', 'U_SU4']

    best_accuracy = 0.
    accuracies = {}   

    for U in Unitaries:
        print("\n Running QCNN with ", U, " and ", str(U_num_params[U]), " params.")
        accuracy = Benchmarking.Benchmarking(dataset, U, U_num_params[U], filename, testName + U, LEARNING_RATE, BATCH_SIZE, circuit='QCNN', steps = EPOCHS, snr=1)
        accuracies[U] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            Architecture = U
            print("\n Accuracy Improvement \n")

    print(accuracies)

    Unitaries= ['U_TTN', 'U_6', 'U_5', 'U_9', 'U_13', 'U_14', 'U_15', 'U_SO4', 'U_SU4']
    
    best_accuracy = 0.000001
    """
    for U in U_num_params:
        print("\n Running QCNN with ", U, " and ", str(U_num_params[U]), " params.")
        accuracy = Benchmarking.Benchmarking(dataset, U, U_num_params[U], filename, testName + U, LEARNING_RATE, BATCH_SIZE, circuit='QCNN', steps = EPOCHS, snr=1)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            Architecture = U
            print("\n Accuracy Improvement \n")
    """
    #print("Best circuit architecture used the ", Architecture, " ansatz, with a test set accuracy of ", str(best_accuracy * 100), "%")
    

    #accuracy = Benchmarking.Benchmarking(dataset, 'U_6', U_num_params['U_6'], filename, testName, LEARNING_RATE, BATCH_SIZE, circuit='QCNN', steps = EPOCHS , snr=SNR)
    """
    lr = LEARNING_RATE
    lr_comparisons = {}
    U_comparisons = {}
    
    for _ in range(5):
        lr /= 10
        U_comparisons = {}
        for U in U_num_params:
            print("\n Running QCNN with ", U, ", ", str(U_num_params[U]), " params, and a learning rate of ", str(lr))
            accuracy = Benchmarking.Benchmarking(dataset, U, U_num_params[U], filename, testName, lr, BATCH_SIZE, circuit='QCNN', steps = EPOCHS, snr=1)
            U_comparisons[U] = accuracy

        lr_comparisons[lr] = U_comparisons    
        print(lr_comparisons)
    
    f = open('C:\\Users\\Fuad K\\Desktop\\Physics\\5th Year\\My_QCNN\\Rewritten Code\\Hypertraining.txt', 'a')
    f.write(str(lr_comparisons))
    f.close()

    
    for _ in range(5):
        lr /= 10
        testName = "_SNR" + str(SNR) + "_LR" + str(lr)
        accuracy = Benchmarking.Benchmarking(dataset, "Large", U_num_params["Large"], filename, testName, lr, BATCH_SIZE, circuit='QCNN', steps = EPOCHS, snr=SNR)
        lr_comparisons[lr] = accuracy
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            best_lr = lr
    print("The best learning rate is " + str(best_lr) + " with an accuracy of " + str(best_accuracy))
    print(lr_comparisons)
    """
    #train pnoise network with gnoise
    
    
    