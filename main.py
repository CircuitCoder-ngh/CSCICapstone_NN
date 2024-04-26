import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import customCallbacks
import requests
import datetime
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.constraints import max_norm


def getAvailableIndicators():
    url = "https://twelve-data1.p.rapidapi.com/technical_indicators"

    headers = {
        "X-RapidAPI-Key": "9778901d6amsh21cb41746d38e0cp18a259jsn700a0307ed38",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers)

    print(response.status_code)
    return response.json()


def getTimeSeries(symbol, interval, outputsize):
    url = "https://twelve-data1.p.rapidapi.com/time_series"

    querystring = {"symbol": symbol, "interval": interval, "outputsize": outputsize, "format": "json"}

    headers = {
        "X-RapidAPI-Key": "9778901d6amsh21cb41746d38e0cp18a259jsn700a0307ed38",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.status_code)
    return response.json()


def getATR(interval, symbol, time_period, outputsize):
    url = "https://twelve-data1.p.rapidapi.com/atr"

    querystring = {"interval": interval, "symbol": symbol, "time_period": time_period, "outputsize": outputsize,
                   "format": "json"}

    headers = {
        "X-RapidAPI-Key": "9778901d6amsh21cb41746d38e0cp18a259jsn700a0307ed38",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.status_code)
    return response.json()


def getOBV(symbol, interval, outputsize):
    url = "https://twelve-data1.p.rapidapi.com/obv"

    querystring = {"symbol": symbol, "interval": interval, "format": "json", "outputsize": outputsize,
                   "series_type": "close"}

    headers = {
        "X-RapidAPI-Key": "9778901d6amsh21cb41746d38e0cp18a259jsn700a0307ed38",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.status_code)
    return response.json()


def getRSI(symbol, interval, time_period, outputsize):
    url = "https://twelve-data1.p.rapidapi.com/rsi"

    querystring = {"interval": interval, "symbol": symbol, "format": "json", "time_period": time_period,
                   "series_type": "close",
                   "outputsize": outputsize}

    headers = {
        "X-RapidAPI-Key": "9778901d6amsh21cb41746d38e0cp18a259jsn700a0307ed38",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.status_code)
    return response.json()


def getMACD(symbol, interval, signal_period, outputsize, fast_period, slow_period):
    url = "https://twelve-data1.p.rapidapi.com/macd"

    querystring = {"interval": interval, "symbol": symbol, "signal_period": signal_period, "outputsize": outputsize,
                   "series_type": "close", "fast_period": fast_period, "slow_period": slow_period, "format": "json"}

    headers = {
        "X-RapidAPI-Key": "9778901d6amsh21cb41746d38e0cp18a259jsn700a0307ed38",
        "X-RapidAPI-Host": "twelve-data1.p.rapidapi.com"
    }

    response = requests.get(url, headers=headers, params=querystring)

    print(response.status_code)
    return response.json()


def combineDataToCSV(symbol, interval, outputsize, time_period):
    """Performs API call to retrieve price and indicator values, combines
    all the data into a list and writes it out to a new CSV"""
    data_price = getTimeSeries(symbol=symbol, interval=interval, outputsize=outputsize)
    data_rsi = getRSI(symbol=symbol, interval=interval, time_period=time_period, outputsize=outputsize)
    data_obv = getOBV(symbol=symbol, interval=interval, outputsize=outputsize)
    data_atr = getATR(symbol=symbol, interval=interval, time_period=time_period, outputsize=outputsize)
    data_macd = getMACD(symbol=symbol, interval=interval, outputsize=outputsize,
                        signal_period=9, slow_period=26, fast_period=12)

    # Initialize the combined list
    combined_list = []

    # Merge indicator values
    for sample_rsi, sample_obv, sample_atr, sample_macd, sample_price \
            in zip(data_rsi['values'], data_obv['values'], data_atr['values'],
                   data_macd['values'], data_price['values']):
        combined_sample = {
            'datetime': sample_rsi['datetime'],  # other fn's depend on this staying loc[0]
            'close': sample_price['close'],  # other fn's depend on this staying loc[1]
            'high': sample_price['high'],
            'low': sample_price['low'],
            'open': sample_price['open'],
            'vol': sample_price['volume'],
            'obv': sample_obv['obv'],
            'rsi': sample_rsi['rsi'],
            'atr': sample_atr['atr'],
            'macd': sample_macd['macd']
        }
        combined_list.append(combined_sample)

    # puts list in chronological order (loc[0] is oldest, loc[i] is most recent
    combined_list = reverse_list(combined_list)

    # Create a DataFrame from the combined list
    df = pd.DataFrame(combined_list)

    # Specify the CSV file path (adjust as needed)
    csv_file_path = f'{symbol}{interval}{time_period}.csv'

    # Write the DataFrame to the CSV file
    df.to_csv(csv_file_path, index=False)

    print(f"Data saved to {csv_file_path}")


def csvToList(filename):
    """Reads in CSV data, removes header and converts it to a list.
    Returns the newly created list"""
    # dataset = np.loadtxt(filename)
    # dataset = np.genfromtxt(fname=filename, delimiter=',', dtype=float)
    df = pd.read_csv(filename, header=None)
    dataset = df.values.tolist()
    dataset.pop(0)
    # numeric_data = np.array(dataset)[:, 1:].astype(float)

    return dataset


def reverse_list(lst):
    """Reverses the order of a list and returns it"""
    new_lst = lst[::-1]
    return new_lst


def createTrainingLabelCSV(filename, profit_target):  # profit_target inputted as a %
    """Reads in a CSV, checks if desired profit target is reached within window size (currently
    set to be 12 candles), writes out a new CSV of binary values"""
    # note: len(training_labels) == len(dataset) - 12
    df = pd.read_csv(filename, header=None)
    dataset = df.values.tolist()
    dataset.pop(0)  # removes header
    training_labels = []
    window_size = 12  # TODO: set as param and test dif sizes
    # loop through each sample in dataset
    for i in range(0, len(dataset) - window_size):
        current_close = float(dataset[i][1])
        # check ahead 12 candles, see if profit_target is reached
        for j in range(1, window_size + 1):
            other_close = float(dataset[i + j][1])  # could make this 'other_high' and change 1 to 2
            if ((other_close - current_close) / current_close) * 100 >= profit_target:
                training_labels.append(1)
                break
            if j == window_size:
                training_labels.append(0)
        # add 12 "2"s to end of training_labels for testing
        if i == len(dataset) - (window_size + 1):
            for x in range(window_size):
                training_labels.append(2)

    df_labels = pd.DataFrame(training_labels)
    csv_file_path = f'{filename[:-4]}_TrainingLabels{profit_target}.csv'
    df_labels.to_csv(csv_file_path, index=False)

    print(f"Training labels saved to {csv_file_path}")


def normalizeListToCSV(filename):
    """Reads in CSV data and normalizes it w/ assumption that col[0] is datetime.
    Writes normalized data out to a new CSV file"""
    dataset = csvToList(filename)

    # convert date into timestamp then into 'time of day' indicator
    for item in dataset:
        day = 24 * 60 * 60
        item[0] = datetime.datetime.strptime(item[0], '%Y-%m-%d %H:%M:%S').timestamp()
        item[0] = np.sin(item[0] * (2 * np.pi / day))

    # Scales all columns of the data into [0,1] range
    scaler = preprocessing.MinMaxScaler()
    scaled_data = scaler.fit_transform(dataset)
    df_scaled_data = pd.DataFrame(scaled_data)

    csv_file_path = f'{filename[:-4]}_Normalized.csv'
    df_scaled_data.to_csv(csv_file_path, index=False)

    print(f"Normalized data saved to {csv_file_path}")


def csvToArray(filename):
    """Reads data from a CSV file and converts it to a Numpy array"""
    dataset = csvToList(filename)
    return np.array(dataset)


def createConfusionMatrix(model, model_name, t_data, t_labels, threshold):
    """Creates and saves a confusion matrix
    Parameters: model, a chosen name for the model, and test data appropriate for the model"""
    predictions = model.predict(t_data)

    # Round to 0 or 1 based on the threshold
    binary_predictions = np.where(predictions >= threshold, 1, 0)

    cm = confusion_matrix(y_true=t_labels, y_pred=binary_predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f'models/groupBcm/{model_name}CM.png')
    plt.close()
    # plt.show()


def create3dDataset(dataset, data_labels, look_back):  # look_back must be 1 w/ current setup
    """Converts input tensor from 2d to 3d
    (intended to format data for LSTM layer)"""
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back - 1):
        a = dataset[i:(i + look_back), :]  # [...:i, 0] would only add the val from col[0]
        dataX.append(a)
        dataY.append(data_labels[i + look_back])

    # x = []
    # for i in range(look_back, len(dataset)):
    #     x.append(dataset[i - look_back:i, :])  # [...:i, 0] would only add the val from col[0]
    # x = np.array(x)

    dataX = np.array(dataX)
    dataY = np.array(dataY)
    print(dataX.shape)
    print(dataY.shape)
    # reshape input to be [samples, time steps, features]
    # x = np.reshape(x, (x.shape[0], look_back, x.shape[2]))  # this line is redundant
    return dataX, dataY



# maybe create a fn that creates models based off the params I am changing
# def create_new_model(train_data, train_labels,

# combineDataToCSV(symbol="SPY", outputsize="5000", interval="5min", time_period="14")
# createTrainingLabelCSV('SPY5min14.csv', 0.2)
# normalizeListToCSV('SPY5min14.csv')
training_data = csvToArray('SPY5min14_Normalized.csv')[:-12]
training_data = np.delete(training_data, 4, axis=1)  # removes 'open'
training_data = np.delete(training_data, 3, axis=1)  # removes 'low'
training_data = np.delete(training_data, 2, axis=1)  # removes 'high'
training_labels = csvToArray('SPY5min14.csv_TrainingLabels0.2.csv')[:-12]

full_t_data = training_data
full_l_data = training_labels
test_size = int(len(training_data) * 0.8)
test_data = training_data[test_size:]
test_labels = training_labels[test_size:]
training_data = training_data[:test_size]  # added after model8 created
training_labels = training_labels[:test_size]  # added after model8 created
# test_data, test_labels = create3dDataset(test_data, test_labels, 2)


def createNewModel(train_data, train_labels, test_data, test_labels,
                   ts, llstm, ulstm, lblstm, do, kc, ld, ud, ne, sb):
    """Assumes input shape of 7"""

    model = Sequential()

    if llstm > 0:
        train_data, train_labels = create3dDataset(train_data, train_labels, lblstm)
        test_data, test_labels = create3dDataset(test_data, test_labels, lblstm)
        for i in range(llstm - 1):
            model.add(LSTM(ulstm, activation='relu', input_shape=(train_data.shape[1], train_data.shape[2]),
                           return_sequences=True, kernel_constraint=max_norm(kc)))
        model.add(LSTM(ulstm, activation='relu', input_shape=(train_data.shape[1], train_data.shape[2]),
                       return_sequences=False, kernel_constraint=max_norm(kc)))

    if do > 0:
        model.add(Dropout(do))

    if llstm == 0:
        model.add(Dense(ud, input_shape=(7,), activation='relu'))  # TODO: replace 7 w/ train_x.shape[_]
        ld -= 1
    for i in range(ld - 1):
        model.add(Dense(ud, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])
    model.fit(train_data, train_labels, epochs=ne, batch_size=sb)
    model_name = f'{ts}_{llstm}_{ulstm}_{lblstm}_{do}_{kc}_{ld}_{ud}_{ne}_{sb}'
    model.save(f'models/groupB/{model_name}.keras')

    model.evaluate(test_data, test_labels)
    createConfusionMatrix(model, model_name, test_data, test_labels, ts)
    createConfusionMatrix(model, f'{model_name}_full', train_data, train_labels, ts)


thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
layers_LSTM = [3]  # [1, 2, 3]  # [0, 1, 2, 3]
units_LSTM = [64]  # [8, 16, 32, 64]  # [4, 8, 16, 32, 64]
lookback_LSTM = [3]  # [1, 2, 3]
dropout = [0, 0.1, 0.2, 0.4, 0.8]
kernel_constraints = [None, 5, 2.5, 1]
layers_Dense = [2, 3, 4]  # [1, 2, 3, 4]
units_Dense = [8, 16, 32, 64]  # [4, 8, 16, 32, 64]
num_epoch = [50, 100, 150, 200]
size_batch = [3, 6, 12, 24]

# ts = 0.7, do = 0, kc = 5, ne = 100, sb = 6
ts, do, kc, ne, sb = 0.7, 0, 5, 150, 6
for llstm in layers_LSTM:
    for ulstm in units_LSTM:
        for lblstm in lookback_LSTM:
            for ld in layers_Dense:
                for ud in units_Dense:
                    createNewModel(training_data, training_labels, test_data, test_labels,
                                   ts, llstm, ulstm, lblstm, do, kc, ld, ud, ne, sb)


# for ts, llstm, ulstm, lblstm, do, kc, ld, ud, ne, sb in zip(thresholds, layers_LSTM, units_LSTM, lookback_LSTM, dropout,
#                                                             kernel_constraints, layers_Dense, units_Dense, num_epoch,
#                                                             size_batch):
#     createNewModel(training_data, training_labels, test_data, test_labels,
#                    ts, llstm, ulstm, lblstm, do, kc, ld, ud, ne, sb)

# model = keras.models.load_model('models/3DenselayersPrecision150e10b.keras')
# model.evaluate(training_data, training_labels)

# model = keras.models.load_model('models/1LSTM_RSF_2Dense_150e10b.keras')
# model.evaluate(test_data, test_labels)

# model = keras.models.load_model('models/2LSTM_RST_2Dense_150e10b.keras')
# model.evaluate(test_data, test_labels)

# model = model4()
# model = keras.models.load_model('models/1LSTM_RSF_Drop_2Dense_150e10b.keras')
# model.evaluate(test_data, test_labels)
# createConfusionMatrix(model, 'model4b', test_data, test_labels)

# model = model5()
# model = keras.models.load_model('models/1LSTM_RSF_2Dense_250e10b.keras')
# model.evaluate(test_data, test_labels)
# createConfusionMatrix(model, 'model5', test_data)
#
# model = model6()
# # model = keras.models.load_model('models/2LSTM_RST_Drop_2Dense_250e10b.keras')
# model.evaluate(test_data, test_labels)
# createConfusionMatrix(model, 'model6', test_data)

# model = model7()
# model = keras.models.load_model('models/2LSTM_RST_Drop_3Dense_250e10bV2.keras')
# model.evaluate(test_data, test_labels)
# createConfusionMatrix(model, 'model7b', test_data, test_labels)

# model = model8()
# model = keras.models.load_model('models/1LSTM_RSF_3Dense_250e10b.keras')
# model.evaluate(test_data, test_labels)
# training_data, training_labels = create3dDataset(training_data, training_labels, 2)
# createConfusionMatrix(model, 'model8b', test_data, test_labels)

# model = model9()
# model = keras.models.load_model('models/3LSTM_RSF_2Dense_200e10b.keras')
# model.evaluate(test_data, test_labels)
# createConfusionMatrix(model, 'model9t09', test_data, test_labels)

# model = model10()
# model = keras.models.load_model('models/3_32LSTM_RST_2Dense_200e10b.keras')
# model.evaluate(test_data, test_labels)
# createConfusionMatrix(model, 'model10t09', test_data, test_labels)

# model = model11()
# model.evaluate(test_data, test_labels)
# createConfusionMatrix(model, 'model11t08', test_data, test_labels, 0.8)

# model = model12(training_data, training_labels)
# model = keras.models.load_model('models/1LSTM_KC3_RSF_2Dense_150e6b_lb1.keras')
# model.evaluate(test_data, test_labels)
# createConfusionMatrix(model, 'model12t09_lb1', test_data, test_labels, 0.9)

# model = model13(training_data, training_labels)
# model.evaluate(test_data, test_labels)
# model = keras.models.load_model('models/2LSTM_KC3_RST_2Dense_150e6b_lb1.keras')
# createConfusionMatrix(model, 'model13t095_lb1', test_data, test_labels, 0.95)

# model = model14(training_data, training_labels)
# createConfusionMatrix(model, 'model14t095_lb1', test_data, test_labels, 0.95)

# model = model15(training_data, training_labels)
# model = keras.models.load_model('models/1LSTM_KC3_RSF_3_16Dense_150e3b_lb2.keras')
# createConfusionMatrix(model, 'model15t01_lb2', test_data, test_labels, 1)
# for layer in model.layers:
#     print(layer.get_weights())

# model = model16(training_data, training_labels)
# createConfusionMatrix(model, 'model16t095', test_data, test_labels, 0.95)
