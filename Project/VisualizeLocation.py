import numpy
from keras.models import load_model
from keras.utils.visualize_util import plot

model = load_model('my_modeltest.h5')

numpy.random.seed(7)


dataframe = numpy.loadtxt("Kingston_Police_Formatted.csv",delimiter=",")
ToUseX = dataframe[:,9:]


def create_database(dataset,look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a= dataset[i:(i+look_back),:]
        dataX.append(a)
        dataY.append(dataset[i+look_back,:])
    return numpy.array(dataX), numpy.array(dataY)


latmax = max(dataframe[:,7])
longmax = max(dataframe[:,8])
latmin = min(dataframe[:,7])
longmin = min(dataframe[:,8])

print(latmax)
print(longmax)
print(latmin)
print(longmin)

toUse = create_database(ToUseX,4)
toUse = numpy.reshape(toUse,(toUse.shape[0],4,2))

predict = model.predict(toUse)

for x in predict:
    print(x)
