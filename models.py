from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Dropout, merge
from keras.layers.advanced_activations import LeakyReLU, PReLU, ELU
from keras.regularizers import l2
from keras.optimizers import *
import os

def fc_block1(x, n=3000, d=0.5):
    x = Dense(n)(x)
    x = BatchNormalization()(x)
    x = LeakyReLU()(x)
    x = Dropout(d)(x)
    return x

def fc_identity(input_tensor, n=3000, d=0.5):
    x = fc_block1(input_tensor, n, d)
    x = Dense(int(input_tensor.shape[1]))(x)
    x = merge([x, input_tensor], mode='sum', concat_axis=1)
    x = LeakyReLU()(x)
    return x

def model9d(feature_size,opt=nadam()):
    n = int(4 * 1024)

    in2 = Input((feature_size,), name='x2')
    x2 = fc_block1(in2, n, d=0.5)
    x2 = fc_identity(x2, n, d=0.5)
    x2 = fc_identity(x2, n, d=0.5)

    x = fc_identity(x2, n, d=0.5)
    x = fc_identity(x, n, d=0.5)
    x = fc_block1(x, n)

    out = Dense(1, activation='relu', name='output')(x)

    model = Model(input=in2, output=out)
    model.compile(optimizer=opt, loss='mean_squared_error')
    return model
