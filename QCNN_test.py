import pickle
import QCNN_circuit
import Benchmarking
import numpy as np
'''
Loads in a pickeled file and then unpickels it
Specifically does this for the paramaters
ARG: filename
Returns: list of paramaters
'''
#Neither of these accuracy are used
#we use the one from brenchmarking
def accuracy_test(predictions, labels, binary = True):
    '''
    This functions calculates the accuracy of the preedicitons
    args: predictions - is the label outputed by neural network (QCNN/CNN)
          labels - Y_test/Y_train datta
          binary True/Flase (not used)
    '''
    #QE add meaningful varaible names
    acc = 0
    for l,p in zip(labels, predictions):
        if p[0] > p[1]: 
            if p[0]>0.75:
                P = 0
            else:
                P = 1
        else:
            P = 1
        if P == l:
            acc = acc + 1
    return acc / len(labels)

def accuracy2(predictions,labels):
    acc = 0
    #data = open('Data2.txt','a')
    
    for lab,pred in zip(labels, predictions):
        #print(lab)
        #data.write(str((lab,pred)))
        if pred[0]>pred[1]:
            classification = 0
        else:
            if ((pred[1]-pred[0])>0.1):
                classification = 1
            else:
                classification=0

        if classification == lab:
            acc = acc + 1
    #data.close()
    return acc / len(labels)
        

def loadParams(filename):
    paramfile = open(filename, 'rb')     
    params = pickle.load(paramfile)
    paramfile.close()
    params = np.array(params)
    #params = np.array(eval(params.replace(" ", ',')))
    return params

def testParameters(params,filename):
    print("Paramaters loaded")
    datafile = open(filename, 'rb')     
    currentdata = pickle.load(datafile)
    X_train, X_val, X_test, Y_train, Y_val, Y_test = currentdata
    #X_train, X_test, Y_train, Y_test = currentdata
    #print(Y_test)
    datafile.close()
    U = 'U_6'
    U_params = 10
    predictions = [QCNN_circuit.QCNN(x, params, U, U_params) for x in X_test]
    #print(predictions)
    accuracy = Benchmarking.accuracy_test(predictions, Y_test)
    print("Accuracy for " + U + " Amplitude :" + str(accuracy))
    #accuracythreshold = accuracy2(predictions, Y_test)
    #print("Accuracy with thresold :" + str(accuracythreshold))

#This is what runs first when you run the File
#This is where you can test your QCNN model
if __name__ == "__main__":
    #This was the code that was used for if you just want to test a single model
    #name = input("Enter file name of the model you would like to load: ")
    #params =loadParams(name)
    #print(params)
    #oldData = input("Enter file name of data: ")

    #Put the model and the corresponding data
    params =["C:\\Users\\Fuad K\\Desktop\\Physics\\5th Year\\My_QCNN\\QCNN_Models\\modelSNR100_LR0.001_U_6_U_6_C1.2424575214461901.pkl"]
    oldData =["C:\\Users\\Fuad K\\Desktop\\Physics\\5th Year\\My_QCNN\\QCNN_Data\\QData2024-03-18SNR0_LR0.001_U_SU4\\dataSNR0_LR0.001_U_SU4.pkl"]
    for i in range(1):
        paramsLoaded =loadParams(params[i])
        paramsLoaded = [paramsLoaded[i] for i in range(paramsLoaded.size)]
        testParameters(paramsLoaded,oldData[i])