"""
An example of using multivariate time series data in a single-layer Long Short Term Memory (LSTM) RNN network.
Used on a regression problem for pollution prediction on hourly basis, given as input the data known in previous hour
(Part 1), or data known in the two previous hours (Part 2).

References:
Dataset: https://archive.ics.uci.edu/ml/datasets/Beijing+PM2.5+Data (dataset)
Starting point: https://machinelearningmastery.com/multivariate-time-series-forecasting-lstms-keras/

"""

import numpy as np
import pandas as pd
from datetime import datetime
from matplotlib import pyplot as plt


def parse(x):
    return datetime.strptime(x, "%Y %m %d %H")


def preprocess_data():
    # Load and pre-process data
    data = pd.read_csv("data/pollution_data.csv", parse_dates=[['year', 'month', 'day', 'hour']], index_col=0, date_parser=parse)
    data.drop('No', axis=1, inplace=True)
    data.columns = ['pollution', 'dew', 'temp', 'press', 'wnd_dir', 'wnd_spd', 'snow', 'rain']
    data.index.name = 'date'
    data['pollution'].fillna(0, inplace=True)
    data = data[24:]
    print(data.head())
    data.to_csv('data/pollution_data_processed.csv')


def descriptive_stats():
    # Visualize the data in time
    data = pd.read_csv('data/pollution_data_processed.csv', header=0, index_col=0)
    print(data.head())
    data.plot(subplots=True, use_index=True)
    plt.legend(loc=2)
    plt.show()


def data_prepare_lstm():
    """
    This method prepares the time-series data for the LSTM, converting time-series into a supervised ML problem.
    The goal of our model would be to predict the pollution in the next hour, based on conditions of prior time step.


    :return: numpy matrices for: trainX, trainY, testX, testY; the StandardScaler object
    """

    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler
    from sklearn.preprocessing import LabelEncoder

    data = pd.read_csv('data/pollution_data_processed.csv', header=0, index_col=0)

    # Use LabelEndocoder to convert categorical number into numeric
    le = LabelEncoder()
    data.wnd_dir = le.fit_transform(data.wnd_dir)

    # OneHot-encode the wnd_dir column
    wnd_dir = pd.get_dummies(data.wnd_dir, prefix='wnddir')
    # Exclude one one-hot encoded column, as the first 3 already have the needed information
    wnd_dir.drop('wnddir_3', axis=1, inplace=True)
    # Merge wnd_dir data back to our datafame and remove old `wnd_dir` column
    data = pd.concat([data, wnd_dir], axis=1, join='outer')
    data.drop('wnd_dir', axis=1, inplace=True)

    # Now we need to create an `output` column which is a shifted up version of the `pollution` column
    # This is because for a given data row, we need predict the output of NEXT row
    data['output'] = data.pollution.shift(-1, axis=0)
    # Remove last row, since there's no output value for it
    data.drop(data.tail(1).index, inplace=True)

    # Normalize data, the StandardScaler will actually return a numpy matrix
    scaler = StandardScaler()
    data = scaler.fit_transform(data)
    # data_scaled = pd.DataFrame(data_scaled, columns=data.columns, index=data.index)

    # Split into training and test sets
    train, test = train_test_split(data, test_size=0.2, shuffle=False)

    # Put input and output columns in separate variables
    trainX = train[:, :-1]
    trainY = train[:, -1]
    testX = test[:, :-1]
    testY = test[:, -1]

    return trainX, trainY, testX, testY, scaler


def data_prepare_lstm_2(trainX, trainY, testX, testY):
    """
    This method uses the previous `data_prepare_lstm` method to build a more complex data structure as input.
    Instead of passing a single read of the input variables at a given LSTM cell, we now pass 2: the most recent reads,
    and the second most recent reads (given by two rows in our dataset). Let's see if the performance improves..

    :return: numpy matrices for: trainX, trainY, testX, testY
    """

    # First, I'll put the data in a pandas DataFrame for easier manipulation
    trainx_df = pd.DataFrame(trainX)
    testx_df = pd.DataFrame(testX)
    trainy_df = pd.DataFrame(trainY)
    testy_df = pd.DataFrame(testY)

    # Stack the input dataframes horizontally (twice) and shift the second piece one position up
    trainx_df = pd.concat([trainx_df, trainx_df], axis=1)
    trainx_df.columns = list(np.arange(trainx_df.shape[1]))
    for i in range(10,20):
        trainx_df[i] = trainx_df[i].shift(-1, axis=0)

    testx_df = pd.concat([testx_df, testx_df], axis=1)
    testx_df.columns = list(np.arange(testx_df.shape[1]))
    for i in range(10, 20):
        testx_df[i] = testx_df[i].shift(-1, axis=0)

    # Shift the outputs one position up as well
    trainy_df = trainy_df.iloc[1:]
    testy_df = testy_df.iloc[1:]

    # Since we've shifted part of the input matrices one position up, we need to delete the bottom row
    trainx_df = trainx_df.iloc[:-1]
    testx_df = testx_df.iloc[:-1]

    # print(trainx_df.shape, trainy_df.shape, testx_df.shape, testy_df.shape)
    # print(trainx_df.head())
    return np.array(trainx_df), np.array(trainy_df), np.array(testx_df), np.array(testy_df)


def model(trainX, trainY, testX, testY):

    from keras.models import Sequential
    from keras.layers import LSTM, Dense

    model = Sequential()

    # You give the input shape of a single sample in the LSTM node
    model.add(LSTM(50, input_shape=(trainX.shape[1], trainX.shape[2])))

    # Add a single output layer
    model.add(Dense(1))

    # Note that this is a Regression problem we're solving, predicting a continuous value of the `pollution` variable
    model.compile(loss='mse', optimizer='adam')

    # Fit the model
    fit_ = model.fit(trainX, trainY, epochs=50, batch_size=64, validation_data=(testX, testY), verbose=2, shuffle=False)

    # Plot the loss function
    plt.plot(fit_.history['loss'], label='train')
    plt.plot(fit_.history['val_loss'], label='test')
    plt.legend()
    plt.show()

    return model, fit_


def model_eval(model, scaler, testX, testY):
    """
    Evaluate the model.

    :return: /
    """

    from sklearn.metrics import mean_squared_error

    y_pred = model.predict(testX)

    # Reshape back to starting shape
    testX = testX.reshape((testX.shape[0], testX.shape[2]))

    # Because `scaler` was used on the whole matrix, we need to recreate it again, and then apply the inverse function
    inv_ypred = np.concatenate((y_pred, testX[:, 1:]), axis=1)
    inv_ypred = scaler.inverse_transform(inv_ypred)
    inv_ypred = inv_ypred[:, 0]

    # Do the same steps for the real outputs
    testY = testY.reshape((len(testY), 1))
    inv_y = np.concatenate((testY, testX[:, 1:]), axis=1)
    inv_y = scaler.inverse_transform(inv_y)
    inv_y = inv_y[:, 0]

    # Calculate Root Mean Squared Error
    rmse = np.sqrt(mean_squared_error(inv_y, inv_ypred))
    print('Test RMSE: %.3f' % rmse)


if __name__ == '__main__':

    # --------------- Data prep ---------------

    # Call pre-processing method only the first time
    # preprocess_data()

    # Run some descriptive stats on data
    # descriptive_stats()

    # Prepare data for LSTM
    trainX, trainY, testX, testY, scaler = data_prepare_lstm()

    # --------------- Part 1 ---------------

    # Add a third dimension to the matrices, as needed for the input of LSTM
    # Dimensions now are [samples, timestamps, features]
        # trainX = trainX.reshape((trainX.shape[0], 1, trainX.shape[1]))
        # testX = testX.reshape((testX.shape[0], 1, testX.shape[1]))

    # Create the neural net model
    # model, fit_ = model(trainX, trainY, testX, testY)
    # My output: Epoch 50 / 50 | loss: 0.0864 - val_loss: 0.0736

    # Get the Root Mean Squared error on the test set
    # model_eval(model, scaler, testX, testY)

    # --------------- Part 2 ---------------

    trainX, trainY, testX, testY, scaler = data_prepare_lstm()

    trainX, trainY, testX, testY = data_prepare_lstm_2(trainX, trainY, testX, testY)
    print(trainX.shape, trainY.shape, testX.shape, testY.shape)

    # Reshape similar as earlier
    trainX2 = trainX.reshape((trainX.shape[0], 2, 10))
    testX2 = testX.reshape((testX.shape[0], 2, 10))

    # Assert our shapes are done right
    assert trainX[10, 12] == trainX2[10, 1, 2]
    assert trainX[10, 3] == trainX2[10, 0, 3]
    assert testX[10, 12] == testX2[10, 1, 2]
    assert testX[10, 3] == testX2[10, 0, 3]

    # Run the model
    model(trainX2, trainY, testX2, testY)
    # My output: Epoch 50 / 50 | loss: 0.0734 - val_loss: 0.0665
    # Result is better compared to previous try






