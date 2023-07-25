from setting import *
from utils.preprocessing import get_training_data
from utils.transformer import TransformerModel
from utils.export_charts import plot_roc, plot_acc_roc
import os
import glob
import pandas as pd


def arrange_data():
    # Get a list of all the csv files
    csv_files = glob.glob('test_data/*.csv')

    # Loop over the list of files
    for file in csv_files:
        df = pd.read_csv(file)

        if 'Unnamed: 0' in df.columns:
            df = df.drop(['Unnamed: 0', 'change'], axis=1)

        df['date'] = pd.to_datetime(df['date'])
        df = df.sort_values('date')
        #df = df[df['date'] < '2023-01-01']
        # Saving the DataFrame to CSV without index
        df.to_csv(file, index=False)


def training_transformer(look_back=10):
    # Join the path with the pattern for csv files
    pattern = os.path.join(TRAINING_DATA_PATH, '*.csv')

    # Use glob to get all csv files
    csv_files = glob.glob(pattern)

    X_train, Y_train, X_test, Y_test = get_training_data(csv_files, look_back, SHARE_DATA_PATH)

    # Hyperparameters
    feature_size = 902
    hidden_size = 512
    nhead = 2
    num_layers = 2
    output_size = 1

    num_epochs = 1
    learning_rate = 0.001
    model = TransformerModel(feature_size, hidden_size, output_size, nhead, num_layers)
    model.training_step(X_train, Y_train, X_test, Y_test, learning_rate, num_epochs, TRANSFORMER_PATH)

    #do model test
    pattern = os.path.join(TEST_DATA_PATH, '*.csv')
    # Use glob to get all csv files
    csv_files = glob.glob(pattern)
    X_test, Y_test = get_training_data(csv_files, look_back, SHARE_DATA_PATH, test=True)
    test_output, test_y = model.test_model(X_test, Y_test)
    #plot_roc(test_output, test_y)
    plot_acc_roc(test_output, test_y)


training_transformer()