# -*- coding: utf-8 -*-
"""
Created on Thu Oct  4 19:23:52 2018

@author: Mohamed Hammad
"""
# math and data manipulation
import numpy as np
import pandas as pd

# set random seeds 
from numpy.random import seed
from tensorflow import set_random_seed
from sklearn.preprocessing import MinMaxScaler

# modeling
from keras.models import Sequential
from keras.layers import LSTM, Dense

# progress bar
from tqdm import tqdm

RANDOM_SEED = 2018
seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)

consumption_train = pd.read_csv('consumption_train.csv', 
                                index_col=0, parse_dates=['timestamp'])
#choose subset of series for training
frac_series_to_use = 1
ntrain = 1 # Size of training minibatches
rng = np.random.RandomState(seed=RANDOM_SEED)
series_ids = consumption_train.series_id.unique()
series_mask = rng.binomial(1,
                           frac_series_to_use,
                           size=series_ids.shape).astype(bool)

training_series = series_ids[series_mask]

#reduce training data to series subset
consumption_train = consumption_train.loc[consumption_train.series_id.isin(training_series)]
cold_start_test = pd.read_csv('cold_start_test.csv', 
                              index_col=0, parse_dates=['timestamp'])
submission_format = pd.read_csv('submission_format.csv',
                                index_col='pred_id',
                                parse_dates=['timestamp'])

pred_windows = submission_format[['series_id', 'prediction_window']].drop_duplicates()
cold_start_test = cold_start_test.merge(pred_windows, on='series_id')

num_cold_start_days_provided = (cold_start_test.groupby('series_id')
                                               .prediction_window
                                               .value_counts()
                                               .divide(24))
def create_lagged_features(df, lag=1):
    if not type(df) == pd.DataFrame:
        df = pd.DataFrame(df, columns=['consumption'])
    
    def _rename_lag(ser, j):
        ser.name = ser.name + f'_{j}'
        return ser
        
    # add a column lagged by `i` steps
    for i in range(1, lag + 1):
        df = df.join(df.consumption.shift(i).pipe(_rename_lag, i))

    df.dropna(inplace=True)
    return df

# example series
#test_series = consumption_train[consumption_train.series_id == 100283]

def prepare_training_data(consumption_series, lag):
    """ Converts a series of consumption data into a
        lagged, scaled sample.
    """
    # scale training data
    scaler = MinMaxScaler(feature_range=(-1, 1))
    consumption_vals = scaler.fit_transform(consumption_series.values.reshape(-1, 1))
    
    # convert consumption series to lagged features
    consumption_lagged = create_lagged_features(consumption_vals, lag=lag)

    # X, y format taking the first column (original time series) to be the y
    X = consumption_lagged.drop('consumption', axis=1).values
    y = consumption_lagged.consumption.values
    
    # keras expects 3 dimensional X
    X = X.reshape(X.shape[0], 1, X.shape[1])
    
    return X, y, scaler

#_X, _y, scaler = prepare_training_data(test_series.consumption, 5)

# lag of 24 to simulate smallest cold start window. Our series
# will be converted to a num_timesteps x lag size matrix
lag =  24

# model parameters
num_neurons = 24
batch_size = 1  # this forces the lstm to step through each time-step one at a time
batch_input_shape=(batch_size, 1, lag)

# instantiate a sequential model
model1 = Sequential()

# add LSTM layer - stateful MUST be true here in 
# order to learn the patterns within a series
model1.add(LSTM(units=num_neurons, 
              batch_input_shape=batch_input_shape, 
              stateful=True))

# followed by a dense layer with a single output for regression
model1.add(Dense(1))

# compile
model1.compile(loss='mean_absolute_error', optimizer='adam')

num_training_series = consumption_train.series_id.nunique()
num_passes_through_data = 1

for i in range(num_passes_through_data):
    count = 0
    # reset the LSTM state for training on each series
    for ser_id, ser_data in tqdm(consumption_train.groupby('series_id'),
                                 total =152,#len(consumption_train.series_id.unique()), 
                                 desc='Training'):
        count += 1
        # prepare the data
        X, y, scaler = prepare_training_data(ser_data.consumption, lag)

        # fit the model: note that we don't shuffle batches (it would ruin the sequence)
        # and that we reset states only after an entire X has been fit, instead of after
        # each (size 1) batch, as is the case when stateful=False
        model1.fit(X, y, epochs=1, batch_size=batch_size, verbose=0, shuffle=False)
        model1.reset_states()
        if count == 152:
            break
def generate_hourly_forecast(num_pred_hours, consumption, model, scaler, lag):
    """ Uses last hour's prediction to generate next for num_pred_hours, 
        initialized by most recent cold start prediction. Inverts scale of 
        predictions before return.
    """
    # allocate prediction frame
    preds_scaled = np.zeros(num_pred_hours)
    
    # initial X is last lag values from the cold start
    X = scaler.transform(consumption.values.reshape(-1, 1))[-lag:]
    
    # forecast
    for i in range(num_pred_hours):
        # predict scaled value for next time step
        yhat = model1.predict(X.reshape(1, 1, lag), batch_size=1)[0][0]
        preds_scaled[i] = yhat
        
        # update X to be latest data plus prediction
        X = pd.Series(X.ravel()).shift(-1).fillna(yhat).values

    # revert scale back to original range
    hourly_preds = scaler.inverse_transform(preds_scaled.reshape(-1, 1)).ravel()
    return hourly_preds

pred_window_to_num_preds = {'hourly': 24, 'daily': 7, 'weekly': 2}
pred_window_to_num_pred_hours = {'hourly': 24, 'daily': 7 * 24, 'weekly': 2 * 7 * 24}

my_submission = submission_format.copy()

num_test_series = my_submission.series_id.nunique()

model1.reset_states()


'''for ser_id, pred_df in tqdm(my_submission.groupby('series_id'), 
                            total=num_test_series, 
                            desc="Forecasting from Cold Start Data"):
        
    # get info about this series' prediction window
    pred_window = pred_df.prediction_window.unique()[0]
    num_preds = pred_window_to_num_preds[pred_window]
    num_pred_hours = pred_window_to_num_pred_hours[pred_window]
    
    # prepare cold start data
    series_data = cold_start_test[cold_start_test.series_id == ser_id].consumption
    cold_X, cold_y, scaler = prepare_training_data(series_data, lag)
    
    # fine tune our lstm model to this site using cold start data    
    model1.fit(cold_X, cold_y, epochs=2, batch_size=batch_size, verbose=0, shuffle=False)
    
    # make hourly forecasts for duration of pred window
    preds = generate_hourly_forecast(num_pred_hours, series_data, model1, scaler, lag)
    
    # reduce by taking sum over each sub window in pred window
    reduced_preds = [pred.sum() for pred in np.split(preds, num_preds)]
    
    # store result in submission DataFrame
    ser_id_mask = my_submission.series_id == ser_id
    my_submission.loc[ser_id_mask, 'consumption'] = reduced_preds'''
    