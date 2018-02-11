'''
Predicts upcoming crime type using last look_back crime types
'''
import numpy
import matplotlib.pyplot as plt
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
SHAPE THE DATASET
'''

numpy.random.seed(7)

dataframe = numpy.loadtxt("Kingston_Police_Formatted.csv",delimiter=",")
train_size = int(len(dataframe)*0.67) #length of 2/3rds into the dataset
test_size = int(len(dataframe)-train_size) #length 1/3rd of the dataset

train_main_type = dataframe[0:train_size,:6]
test_main_type = dataframe[train_size:len(dataframe)-1,:6]

train_DOY_main = dataframe[0:train_size,11:13]
test_DOY_main = dataframe[train_size:len(dataframe)-1,11:13]

train_loc_main = dataframe[0:train_size,9:11]
test_loc_main = dataframe[train_size:len(dataframe)-1,9:11]



def create_database(dataset,look_back=1):
    #3D arrays, len(dataset) X number of datapoints X features per datapoint
    dataX = [] #datapoints are arrays containing last look_back crime types
    dataY = [] #datapoints are the upcoming crime type

    for i in range(len(dataset)-look_back-1):
        dataX.append(dataset[i:(i+look_back),:])
        dataY.append(dataset[i+look_back,:])

    return numpy.array(dataX), numpy.array(dataY)

trainX_main, trainY_main = create_database(train_main_type,4)
testX_main, testY_main = create_database(test_main_type,4)

trainX_DOY, trainY_DOY = create_database(train_DOY_main,4)
testX_DOY, testY_DOY = create_database(test_DOY_main,4)

trainX_loc, trainY_loc = create_database(train_loc_main,4)
testX_loc, testY_loc = create_database(test_loc_main, 4)

trainX_main = numpy.reshape(trainX_main,(trainX_main.shape[0],4,6))
testX_main = numpy.reshape(testX_main,(testX_main.shape[0],4,6))

print("Reshape Type",trainX_main.shape,testX_main.shape)

trainX_DOY = numpy.reshape(trainX_DOY,(trainX_DOY.shape[0],4,2))
testX_DOY = numpy.reshape(testX_DOY,(testX_DOY.shape[0],4,2))

print("Reshape HOY",trainX_DOY.shape,testX_DOY.shape)

trainX_loc = numpy.reshape(trainX_loc,(trainX_loc.shape[0],4,2))
testX_loc = numpy.reshape(testX_loc,(testX_loc.shape[0],4,2))

print("Reshape Location", trainX_loc.shape, testX_loc.shape)

DOY_Branch = Sequential()
DOY_Branch.add(LSTM(32,return_sequences=False,input_dim = 2))
DOY_Branch.add(Dense(2))

Type_Branch = Sequential()
Type_Branch.add(LSTM(32,return_sequences=False,input_dim = 6))
Type_Branch.add(Dense(6))

Loc_Branch = Sequential()
Loc_Branch.add(LSTM(32,return_sequences=False,input_dim=2))

merged = Merge([DOY_Branch, Type_Branch, Loc_Branch], mode='concat')

final_model = Sequential()
final_model.add(merged)
final_model.add(Dense(6,activation='softmax'))

final_model.compile(optimizer='rmsprop', loss='mean_squared_error',metrics=['accuracy','categorical_accuracy'])
final_model.fit([trainX_DOY, trainX_main, trainX_loc], trainY_main, nb_epoch=1,batch_size=10, verbose=2)
try:
    print("Evaluating:)
    score = final_model.evaluate([testX_DOY, testX_main, testX_loc], testY_main, batch_size = 10)
    print ("Score is : "score)

except:
    print("Couldn't evaluate due to nonetype error")

final_model.save('FunctionMergeCrimeHOY.h5')
