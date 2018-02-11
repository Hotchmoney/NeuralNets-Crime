'''
Predicts upcoming crime type using last look_back crime types
'''
import numpy
import matplotlib.pyplot as plt
from keras.utils.visualize_util import plot
import pandas
import math
from keras.models import load_model
from keras.models import Sequential
from keras.layers.core import Dropout
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers import Merge
from keras.layers import Input
from keras.layers import Embedding
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

'''
THIS FILE RUNS THE NEURAL NETWORK BY
DEFINING THE ARCHITECTURE AND TRAINING IT.
IT WILL ALSO TEST USING THE EVALUATE METHOD.
WILL ALSO PRINT OUT A FILE WITH THE PREDICTED CLASSES
USING THE NEURAL NETWORK.
RUN PROCESS_DATA.PY FIRST, THEN RUN THIS.
'''


'''
Set the seed for the sake of being able to repeat the results
'''
numpy.random.seed(7)

dataframe = numpy.loadtxt("Kingston_Police_Formatted.csv",delimiter=",")
train_size = int(len(dataframe)*0.67) #Train dataset length of 2/3rds into the dataset
test_size = int(len(dataframe)-train_size) #Test dataset length 1/3rd of the dataset

#Retrieving crime type from dataframe
train_main_type = dataframe[0:train_size,:6]
test_main_type = dataframe[train_size:len(dataframe),:6]

#Retrieving hour Of Year from dataframe
train_DOY_main = dataframe[0:train_size,11:13]
test_DOY_main = dataframe[train_size:len(dataframe),11:13]

#Retrieving location from dataframe
train_loc_main = dataframe[0:train_size,19:]
test_loc_main = dataframe[train_size:len(dataframe),19:]


'''
create_database will create a numpy array with each
datapoint having x number of previous data points.
'''
def create_database(dataset,look_back=1):
    #3D arrays: len(dataset) X number of datapoints X features per datapoint
    dataX = [] #datapoints are arrays containing last look_back crime types
    dataY = [] #datapoints are the upcoming crime type

    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back),:])
        dataY.append(dataset[i+look_back,:])

    return numpy.array(dataX), numpy.array(dataY)

def create_database_1(dataset,look_back=1):
    dataX = []
    dataY = []

    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back)])
        dataY.append(dataset[i+look_back])

    return numpy.array(dataX), numpy.array(dataY)

'''
Using create_database to get the 4 previous values for each datapoint.
Creates a 3D numpy array.
'''
#X: input, Y: expected output
trainX_main, trainY_main = create_database(train_main_type,5)
testX_main, testY_main = create_database(test_main_type,5)

trainX_DOY, trainY_DOY = create_database(train_DOY_main,5)
testX_DOY, testY_DOY = create_database(test_DOY_main,5)

trainX_loc, trainY_loc = create_database_1(train_loc_main,5)
testX_loc, testY_loc = create_database_1(test_loc_main, 5)
'''
Properly reshaping the data so that network can take it.
'''
trainX_main = numpy.reshape(trainX_main,(trainX_main.shape[0],5,6))
testX_main = numpy.reshape(testX_main,(testX_main.shape[0],5,6))

print("Reshape Type",trainX_main.shape,testX_main.shape)

trainX_DOY = numpy.reshape(trainX_DOY,(trainX_DOY.shape[0],5,2))
testX_DOY = numpy.reshape(testX_DOY,(testX_DOY.shape[0],5,2))

print("Reshape HOY",trainX_DOY.shape,testX_DOY.shape)

trainX_loc = numpy.reshape(trainX_loc,(trainX_loc.shape[0],5,1))
testX_loc = numpy.reshape(testX_loc,(testX_loc.shape[0],5,1))

print("Reshape Location", trainX_loc.shape, testX_loc.shape)

'''
Make a neural network with LSTM and dense of 2 output
for the hour of year feature.
'''
DOY_Branch = Sequential()
DOY_Branch.add(LSTM(32,return_sequences=False,input_dim = 2))
DOY_Branch.add(Dense(2))

'''
Make a neural network with LSTM and dense of 6 output
for the Type of crime feature.
'''
Type_Branch = Sequential()
Type_Branch.add(LSTM(32,return_sequences=False,input_dim = 6))
Type_Branch.add(Dense(6))

'''
Make a neural network with LSTM and dense of 2 output
for the location feature.
'''
Loc_Branch = Sequential()
Loc_Branch.add(LSTM(32,return_sequences=False,input_dim=1))
Loc_Branch.add(Dense(1))

'''
The final model takes a merge of all three previous models
and concatenates and passes  to a dense layer which outputs the
6 possible categories.
'''
final_model = Sequential()
final_model.add(Merge([DOY_Branch, Type_Branch, Loc_Branch], mode='concat'))
final_model.add(Dense(6,activation='softmax'))

'''
Model is compiled with an ADAM optimizer and loss function is
mean squared error. The metrics are accuracy and categorical_accuracy.
'''
final_model.compile(optimizer='adam', loss='mean_squared_error',metrics=['accuracy','categorical_accuracy'])

'''
Train model using training dataset
'''
final_model.fit([trainX_DOY, trainX_main, trainX_loc], trainY_main, nb_epoch=10,batch_size=10, verbose=2)

'''
Evaluate the model using the testing dataset.
'''
try:
    print("Evaluating:")
    score = final_model.evaluate([testX_DOY, testX_main, testX_loc], testY_main, batch_size = 10)
    print ("Score is : "+score)
except:
    print(" Couldn't evaluate due to nonetype error")
'''
Save the model and the weights.
'''
final_model.save('FunctionMergeCrimeHOYClusterLoc.h5')

'''
Predict the output using all elements of the training dataset
and the test dataset.
'''
trainPredict = final_model.predict([trainX_DOY, trainX_main, trainX_loc])
testPredict = final_model.predict([testX_DOY, testX_main, testX_loc])

'''
Save the output of the predict_classes.
'''
numpy.savetxt("resultstrainpredict.csv", trainPredict, delimiter=",")
numpy.savetxt("resultstestpredict.csv",testPredict,delimiter=",")
