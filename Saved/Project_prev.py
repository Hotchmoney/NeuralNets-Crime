import numpy
import matplotlib.pyplot as plt
import pandas
import math
from keras.models import load_model
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error

numpy.random.seed(7)

dataframe = numpy.loadtxt("Kingston_Police_Formatted.csv",delimiter=",")
train_size = int(len(dataframe)*0.67)
test_size = int(len(dataframe)-train_size)

train, test=dataframe[0:train_size,:], dataframe[train_size:len(dataframe),:]
print(len(train),len(test))

def create_database(dataset,look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a= dataset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(dataset[i+look_back,:])
    return numpy.array(dataX), numpy.array(dataY)

look_back = 1
trainX, trainY = create_database(train,look_back)
testX, testY = create_database(test,look_back)
trainX = numpy.reshape(trainX,(trainX.shape[0],1,9))
testX = numpy.reshape(testX,(testX.shape[0],1,9))

print(trainX[0])

'''
model=Sequential()
model.add(LSTM(16,input_dim=look_back*4+5))
model.add(Dense(9))
model.compile(loss='mean_squared_error',optimizer='adam')
model.fit(trainX, trainY, nb_epoch=100,batch_size=10, verbose=2)



model.save('my_model.h5')
'''
