# math and data manipulation
import numpy as np
import pandas as pd

# set random seeds 
from numpy.random import seed
from tensorflow import set_random_seed
from sklearn.preprocessing import MinMaxScaler

# progress bar
from tqdm import tqdm

RANDOM_SEED = 2018
seed(RANDOM_SEED)
set_random_seed(RANDOM_SEED)

consumption_train = pd.read_csv('consumption_train.csv', 
                                index_col=0, parse_dates=['timestamp'])
#choose subset of series for training
frac_series_to_use = 1

rng = np.random.RandomState(seed=RANDOM_SEED)
series_ids = consumption_train.series_id.unique()

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
    for i in range(1, lag):
        df = df.join(df.consumption.shift(i).pipe(_rename_lag, i))

    df.dropna(inplace=True)
    return df

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
    
    return X, y, scaler

# lag of 24 to simulate smallest cold start window. Our series
# will be converted to a num_timesteps x lag size matrix
lag =  24