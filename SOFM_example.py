import numpy as np
from dataSetGen import DataSetGenerator
from sofm import SOFM
from keras.utils import to_categorical
from neural_net import train_model, predict

randSeed = 1
dataSetSize = 10
hiddenSOFMSize = 15

"""
Generate training set
"""
dataGenerator = DataSetGenerator()
trainData = dataGenerator.generateDataSet(dataSetSize, randSeed)
X_train_norm = dataGenerator.normalizeX(trainData['X'])
Y_train = trainData['Y']
Y_train_one_hot = to_categorical(Y_train).T

"""
Generate test set
"""
testData = dataGenerator.generateDataSet(dataSetSize, randSeed+5)
X_test_norm = dataGenerator.normalizeX(testData['X'])
Y_test = testData['Y']
Y_test_one_hot = to_categorical(Y_test).T


def countAccuracy(nnLables, trueLables):
    """
    Find the part of the test sample to which the model assigns correct labels
    
    Parameters
    ----------
    nnLables : numpy array
        lables generated by the model
        
    trueLables:
        correct lables
    
    Returns
    ----------
    Accuracy of the model, a number from 0 to 1
    """
    counter = 0
    l = len(trueLables)
    for i in range(l):
        if np.argmax(nnLables.T[i]) == trueLables[i]:
            counter += 1
    
    return counter/l


def train():
    """
    train the model and save the trained params to file "last_saved_params.npy"
    print the accuracy
    
    Parameters
    ----------
    trainSize: number
        number of samples to be taken from data set for training
    """
    # draw data
    print("Some data visualisation")
    for i in range(3):
        for j in range(i + 1, 3):
            dataGenerator.drawDataSet(X_train_norm, Y_train, i, j)
            
    print('-----------------------------------------')
    print('-----------------------------------------')
    
    """
    Step1: train the SOFM
    """
    sofm = SOFM()
    sofm.initParams(hiddenSOFMSize, 10)
    print("Initial node centers")
    sofm.drawNodeCenters(0, 1)

    sofm.applyKMeans(X_train_norm, 10000, 0.01, showProgress=True)
    print("Node centers after KNN")
    for i in range(3):
        for j in range(i+1, 3):
            sofm.drawNodeCenters(i, j)
        
    sofm.train(X_train_norm,
               mode='WTA',
               numIterations=10000,
               learningRate=0.1,
               showProgress = True)
    
    """
    Step 2: preprocess the training data with the SOFM
    """
    X_train_preprocessed = sofm.apply(X_train_norm)

    """
    Step 3: train a simple NN (in fact, it is a linear regression
    with MSE error)
    """
    layers_dims = [X_train_preprocessed.T.shape[0], 4]
    activations = ["none"]

    trained_params = train_model(X_train_preprocessed.T,
                             Y_train_one_hot,
                             layers_dims,
                             activations,
                             learning_rate = 0.1,
                             num_iterations = 3000,
                             print_cost=True,
                             cost_function="MSE")
    
    np.save('last_saved_params.npy', trained_params)
    
    trainPredict = predict(X_train_preprocessed.T, trained_params)
    X_test_preprocessed = sofm.apply(X_test_norm)
    testPredict = predict(X_test_preprocessed.T, trained_params)
    
    trainAcc = countAccuracy(trainPredict, Y_train)
    testAcc = countAccuracy(testPredict, Y_test)
    
    print("Accuracy on training set: ", str(trainAcc*100), "%")
    print("Accuracy on test set: ", str(testAcc*100), "%")
 

train()