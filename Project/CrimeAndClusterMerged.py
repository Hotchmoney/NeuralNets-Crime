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