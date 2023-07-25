import pandas as pd
import numpy as np
from tqdm import tqdm
import os
import re


def preprocess_data(dataframe, shares, sequence_length):
    dataframe['date'] = pd.to_datetime(dataframe['date'])
    dataframe = dataframe.sort_values('date')
    # Assuming the dataframe is sorted by date
    dataframe['target'] = dataframe['close'].rolling(window=5).mean().shift(-5)
    dataframe['target'] = (dataframe['target'] - dataframe['close']) / dataframe['close']
    # Drop rows with NaN values
    dataframe = dataframe.dropna()

    cols_to_update = dataframe.columns.drop(['date', 'target', 'stock_id', 'close'])

    for col in cols_to_update:
        dataframe[col] = dataframe[col] / shares

    # Convert to numpy arrays
    trading_data = dataframe.drop(columns=['date', 'target']).values
    # Prepare input and output data for Transformer
    output_sequence_length = 1  # choose your output sequence length

    X = []
    Y = []
    for i in range(len(trading_data) - sequence_length + output_sequence_length):
        try:
            X.append(trading_data[i:i+sequence_length])
            Y.append(dataframe['target'].values[i+sequence_length-1])
        except:
            X.append(trading_data[i:])
            Y.append(dataframe['target'].values[-1])

    X = np.array(X)
    Y = np.array(Y)
    return X, Y


def split_and_shuffle(X, Y, test_size=0.2):
    # Create an array of indices and shuffle them
    indices = np.arange(X.shape[0])
    np.random.shuffle(indices)

    # Apply the shuffled indices to X and Y
    X = X[indices]
    Y = Y[indices]

    # Determine the index at which to split the data
    split_idx = int(X.shape[0] * (1 - test_size))

    # Split X and Y into training and test sets
    X_train = X[:split_idx]
    Y_train = Y[:split_idx]
    X_test = X[split_idx:]
    Y_test = Y[split_idx:]

    return X_train, Y_train, X_test, Y_test


#get training or test data
def get_training_data(csv_files, look_back, share_data_path, test=False):
    X_list = []
    Y_list = []
    df_shares = pd.read_csv(share_data_path)
    for file in tqdm(csv_files[:10]):
        stock_id = os.path.splitext(os.path.basename(file))[0]
        try:
            shares_str = df_shares[df_shares['股票代號'] == int(stock_id)]['發行股數'].iloc[0]
            shares_str = shares_str.split(' ')[0]  # This will extract "5,895,645,647"
            shares = int(shares_str.replace(',', ''))
        except:
            print(f"No shares data found for stock_id: {stock_id}")

        df = pd.read_csv(file)
        if len(df) < 10:
            continue
        # Preprocess data and add to list
        X, Y = preprocess_data(df, shares, look_back)
        if len(X) == 0 or len(Y) == 0:
            continue
        X_list.append(X)
        Y_list.append(Y)

    # Concatenate all X and Y
    X_all = np.concatenate(X_list, axis=0)
    Y_all = np.concatenate(Y_list, axis=0)
    if test:
        return X_all, Y_all
    X_train, Y_train, X_test, Y_test = split_and_shuffle(X_all, Y_all)
    return X_train, Y_train, X_test, Y_test
