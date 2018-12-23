import numpy as np
import matplotlib.pyplot as plt
import keras
from keras.layers import *
from keras.models import *
from keras import backend as K
from copy import deepcopy
from prepare_data import prepare_training_data, lag, consumption_train
from tqdm import tqdm
import pandas as pd

seed = 0
plot = True
innerstepsize = 0.02 # stepsize in inner SGD
innerepochs = 3 # number of epochs of each inner SGD
outerstepsize0 = 0.1 # stepsize of outer optimization, i.e., meta-optimization
niterations = len(consumption_train)//672 # number of outer updates; each iteration we sample one task and update on it

rng = np.random.RandomState(seed)
ntrain = 1 # Size of training minibatches

# model parameters
num_neurons = 24
batch_size = 1  # this forces the lstm to step through each time-step one at a time
batch_input_shape=(batch_size, 1, lag)

# Define model.
# instantiate a sequential model
model = Sequential()

# add LSTM layer - stateful MUST be true here in 
# order to learn the patterns within a series
model.add(LSTM(units=num_neurons, 
              batch_input_shape=batch_input_shape, 
              stateful=True))

# followed by a dense layer with a single output for regression
model.add(Dense(1))
# compile
model.compile(loss='mean_absolute_error', optimizer='Adam')

def train_on_batch(x,y):
    model.train_on_batch(x,y)

# Reptile training loop
i=0
for iteration in tqdm(range(niterations-606), total = niterations-606, desc='Training'):
    weights_before = model.weights
    # Generate task
    X, y, _ = prepare_training_data(consumption_train.iloc[i:i+672]
                ['consumption'], lag)
    i+=672
    # Do SGD on this task
    inds = rng.permutation(len(X))
    for _ in range(innerepochs):
        for start in range(0, len(X), ntrain):
            mbinds = inds[start:start+ntrain]
            train_on_batch(X[mbinds], y[mbinds])

    weights_after = model.weights
    model.reset_states()

    outerstepsize = outerstepsize0 * (1 - iteration / niterations) # linear schedule
    for i in range(len(weights_after)):
        model.weights[i]  = (weights_before[i]+
                                (weights_after[i]-weights_before[i])*outerstepsize)