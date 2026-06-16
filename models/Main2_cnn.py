import os
# os.environ["CUDA_VISIBLE_DEVICES"]="-1"

from keras.layers import Input, Dense, LSTM, merge ,Conv1D,Dropout,Bidirectional,Multiply,add,BatchNormalization
from keras.models import Model
import matplotlib.pyplot as plt

from attention_utils import get_activations
from keras.layers import merge
from keras.layers.core import *
from keras.layers.recurrent import LSTM
from keras.models import *
import keras
import  pandas as pd
import  numpy as np
from sklearn.metrics import mean_squared_error,r2_score,mean_absolute_error
from math import sqrt




SINGLE_ATTENTION_VECTOR = False
def attention_3d_block(inputs):
    # inputs.shape = (batch_size, time_steps, input_dim)
    input_dim = int(inputs.shape[2])
    a = inputs
    #a = Permute((2, 1))(inputs)
    #a = Reshape((input_dim, TIME_STEPS))(a) # this line is not useful. It's just to know which dimension is what.
    a = Dense(input_dim, activation='softmax')(a)
    if SINGLE_ATTENTION_VECTOR:
        a = Lambda(lambda x: K.mean(x, axis=1), name='dim_reduction')(a)
        a = RepeatVector(input_dim)(a)
    a_probs = Permute((1, 2), name='attention_vec')(a)

    output_attention_mul = merge([inputs, a_probs], name='attention_mul', mode='mul')
    return output_attention_mul

# Another way to write the attention mechanism Suitable for use with the above error reporting Source:https://blog.csdn.net/uhauha2929/article/details/80733255
def attention_3d_block2(inputs, single_attention_vector=False):
    #If the previous layer is an LSTM, return_sequences=True is required
    # inputs.shape = (batch_size, time_steps, input_dim)
    time_steps = K.int_shape(inputs)[1]
    input_dim = K.int_shape(inputs)[2]
    a = Permute((2, 1))(inputs)
    a = Dense(time_steps, activation='softmax')(a)
    if single_attention_vector:
        a = Lambda(lambda x: K.mean(x, axis=1))(a)
        a = RepeatVector(input_dim)(a)

    a_probs = Permute((2, 1))(a)
    # Multiplying the ATTENTION weights, but not summing them, doesn't seem to have much of an effect
    # If categorizing tasks, doing a Flatten unfolding will do the trick
    # element-wise
    output_attention_mul = Multiply()([inputs, a_probs])
    return output_attention_mul



def create_dataset(dataset, look_back):
    '''
    Processing of data
    '''
    dataX, dataY = [], []
    a = int(len(dataset)/153)
    for k in range(a):
        d=1
        for i in range(k*153+look_back,(k+1)*153+1):
            a = dataset[(i-look_back):i,:]
            dataX.append(a)
            dataY.append(dataset[i-1,6])
    TrainX = np.array(dataX)[:,:,:6]
    Train_Y = np.array(dataY)

    return TrainX, Train_Y

#Multi-dimensional normalization Returns data and maximum and minimum values
def NormalizeMult(data):
    #normalize for inverse normalization
    data = np.array(data)
    normalize = np.arange(2*data.shape[1],dtype='float64')

    normalize = normalize.reshape(data.shape[1],2)
    print(normalize.shape)
    for i in range(0,data.shape[1]):
        #Column i
        list = data[:,i]
        listlow,listhigh =  np.percentile(list, [0, 100])
        # print(i)
        normalize[i,0] = listlow
        normalize[i,1] = listhigh
        delta = listhigh - listlow
        if delta != 0:
            #Line j
            for j in range(0,data.shape[0]):
                data[j,i]  =  (data[j,i] - listlow)/delta
    #np.save("./normalize.npy",normalize)
    return  data,normalize

#multidimensional inverse normalization
def FNormalizeMult(data,normalize):
    data = np.array(data)
    for i in  range(0,data.shape[1]):
        listlow =  normalize[i,0]
        listhigh = normalize[i,1]
        delta = listhigh - listlow
        if delta != 0:
            #Line j
            for j in range(0,data.shape[0]):
                data[j,i]  =  data[j,i]*delta + listlow

    return data


def attention_model():
    inputs = Input(shape=(TIME_STEPS, INPUT_DIMS))
    x = Conv1D(filters = 64, kernel_size = 1, activation = 'relu')(inputs)  #, padding = 'same'
    attention_mul = Flatten()(x)
    output = Dense(1)(attention_mul)
    model = Model(inputs=[inputs], outputs=output)
    return model




#Load data

data = pd.read_excel(". /New Microsoft Excel Worksheet.xlsx")

data = data.drop(['Date (UTC)','Mean air temperature 2m(°C)','ET0','Water demand','Surface air pressure (hPa)'], axis = 1)

print(data.columns)
print(data.shape)


INPUT_DIMS = 6
TIME_STEPS = 20
lstm_units = 64

#normalize
data,normalize = NormalizeMult(data)
pollution_data = data[:,6].reshape(len(data),1)

# train_X, _ = create_dataset(data,TIME_STEPS)
# _ , train_Y = create_dataset(pollution_data,TIME_STEPS)
train_X,train_Y = create_dataset(data,TIME_STEPS)
print(train_X.shape,train_Y.shape)

m = attention_model()
m.summary()
m.compile(optimizer=keras.optimizers.Adam(lr=0.001), loss='mse')

history = m.fit(train_X[:21*134], train_Y[:21*134], epochs=500
                , batch_size=512)

result2 = m.predict(train_X[21*134:])
result1 = m.predict(train_X[:21*134])
train_mse = mean_squared_error(result1,train_Y[:21*134])
test_mse = mean_squared_error(result2,train_Y[21*134:])
train_rmse = sqrt(mean_squared_error(result1,train_Y[:21*134]))
test_rmse = sqrt(mean_squared_error(result2,train_Y[21*134:]))
train_r2 = r2_score(result1,train_Y[:21*134])
test_r2 = r2_score(result2,train_Y[21*134:])
train_mae = mean_absolute_error(result1,train_Y[:21*134])
test_mae = mean_absolute_error(result2,train_Y[21*134:])
#m.save("./model.h5")
#np.save("normalize.npy",normalize)
name = 'result_cnn'
os.makedirs(name,exist_ok=True)

with open(name+'/result.txt','w') as f:
    f.writelines('train_mse:'+str(train_mse)+'\n')
    f.writelines('test_mse:' + str(test_mse)+'\n')
    f.writelines('train_rmse:' + str(train_rmse)+'\n')
    f.writelines('test_rmse:' + str(test_rmse)+'\n')
    f.writelines('train_r2:' + str(train_r2)+'\n')
    f.writelines('test_r2:' + str(test_r2)+'\n')
    f.writelines('train_mae:' + str(train_mae)+'\n')
    f.writelines('test_mae:' + str(test_mae)+'\n')
# loss diagram
plt.figure(0)
plt.xlabel('eopchs')
plt.ylabel('train_loss')
plt.title('train_loss_pic')
plt.plot(range(1, len(history.history['loss']) + 1), history.history['loss'])
plt.savefig(name + '/train_loss.jpg')


plt.figure(1)
plt.xlabel('data')
plt.ylabel('result')
plt.title('All data')
plt.plot(range(1, len(result1.squeeze())+1), result1.squeeze(),color='green')
plt.plot(range(1, len(result1.squeeze())+1), train_Y[:21*134],color='red')
plt.savefig(name + '/21years.jpg')

plt.figure(2)
plt.xlabel('data')
plt.ylabel('result')
plt.title('Predicted data (n days)')
plt.plot(range(1, len(result2.squeeze())+1), result2.squeeze(),color='green', label = 'True value')
plt.plot(range(1, len(result2.squeeze())+1), train_Y[21*134:],color='red', label = 'Predicted')

plt.savefig(name + '/2years.jpg')






