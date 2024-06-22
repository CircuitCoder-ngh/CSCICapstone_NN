import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import plotly.graph_objects as go
import matplotlib.pyplot as plt
import customCallbacks
import requests
import datetime
import os
from sklearn import preprocessing
from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from keras.constraints import max_norm
from apiCalls import *


def combineDataToCSV(symbol, interval, outputsize, time_period, optional=None):
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
            'open': sample_price['open'],
            'high': sample_price['high'],
            'low': sample_price['low'],
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
    # csv_file_path = f'historical_data/{symbol}{interval}{time_period}{optional}.csv'
    csv_file_path = f'historical_data/current.csv'

    # Write the DataFrame to the CSV file
    df.to_csv(csv_file_path, index=False)

    print(f"Data saved to {csv_file_path}")


def combineDataToCSV_AV(symbol, interval, month, time_period, optional=None):
    """Performs API call to retrieve price and indicator values, combines
    all the data into a list and writes it out to a new CSV"""
    data_price = getTimeSeries_AV(symbol=symbol, interval=interval, month=month)
    data_rsi = getRSI_AV(symbol=symbol, interval=interval, time_period=time_period, month=month)
    data_obv = getOBV_AV(symbol=symbol, interval=interval, month=month)
    data_atr = getATR_AV(symbol=symbol, interval=interval, time_period=time_period, month=month)
    data_macd = getMACD_AV(symbol=symbol, interval=interval, month=month,
                           signal_period=9, slow_period=26, fast_period=12)

    # Initialize the combined list
    combined_list = []

    # Merge indicator values
    for sample_rsi, sample_obv, sample_atr, sample_macd, sample_price \
            in zip(data_rsi['Technical Analysis: RSI'], data_obv['Technical Analysis: OBV'],
                   data_atr['Technical Analysis: ATR'], data_macd['Technical Analysis: MACDEXT'],
                   data_price[f'Time Series ({interval})']):
        combined_sample = {
            'datetime': sample_price,  # other fn's depend on datetime at loc[0], and close at loc[1]
            'close': data_price[f'Time Series ({interval})'][sample_price]['4. close'],
            'open': data_price[f'Time Series ({interval})'][sample_price]['1. open'],
            'high': data_price[f'Time Series ({interval})'][sample_price]['2. high'],
            'low': data_price[f'Time Series ({interval})'][sample_price]['3. low'],
            'vol': data_price[f'Time Series ({interval})'][sample_price]['5. volume'],
            'obv': data_obv['Technical Analysis: OBV'][sample_obv]['OBV'],
            'rsi': data_rsi['Technical Analysis: RSI'][sample_rsi]['RSI'],
            'atr': data_atr['Technical Analysis: ATR'][sample_atr]['ATR'],
            'macd': data_macd['Technical Analysis: MACDEXT'][sample_macd]['MACD']
        }
        combined_list.append(combined_sample)
        # print(f'{sample_price}, {sample_obv}, {sample_rsi}, {sample_atr}, {sample_macd}')

    # puts list in chronological order (loc[0] is oldest, loc[i] is most recent
    combined_list = reverse_list(combined_list)

    # Create a DataFrame from the combined list
    df = pd.DataFrame(combined_list)

    # Specify the CSV file path (adjust as needed) # removed {optional} from end of filepath
    if optional is not None:
        csv_file_path = f'historical_data/current.csv'
    else:
        csv_file_path = f'historical_data/{symbol}{interval}_raw/{symbol}{interval}{time_period}_{month}.csv'

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
    # csv_file_path = f'historical_data/{filename[:-4]}_TrainingLabels{profit_target}.csv'
    csv_file_path = f'historical_data/SPY5min_20to23_TrainingLabels{profit_target}.csv'
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

    # csv_file_path = f'historical_data/{filename[:-4]}_Normalized.csv'
    csv_file_path = f'historical_data/SPY5min_20to23_Normalized.csv'
    df_scaled_data.to_csv(csv_file_path, index=False)

    print(f"Normalized data saved to {csv_file_path}")


def csvToArray(filename):
    """Reads data from a CSV file and converts it to a Numpy array"""
    dataset = csvToList(filename)
    return np.array(dataset)


def listToCSV(array, file_path):
    df = pd.DataFrame(array)
    df.to_csv(file_path, index=False)
    print(f"Data saved to {file_path}")


def createConfusionMatrix(model, model_name, group_name, t_data, t_labels, threshold):
    """Creates and saves a confusion matrix
    Parameters: model, a chosen name for the model, the group name, and test data appropriate for the model"""
    predictions = model.predict(t_data)

    # Round to 0 or 1 based on the threshold
    binary_predictions = np.where(predictions >= threshold, 1, 0)

    cm = confusion_matrix(y_true=t_labels, y_pred=binary_predictions)

    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.savefig(f'models/{group_name}cm/{model_name}CM.png')
    plt.close()
    # plt.show()


def create3dDataset(dataset, data_labels, look_back):  # look_back must be 1 w/ current setup
    """Converts input tensor from 2d to 3d
    (intended to format data for LSTM layer)"""
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):  # removed -1 after tradeModel7
        a = dataset[i:(i + look_back), :]  # [...:i, 0] would only add the val from col[0]
        dataX.append(a)
        if data_labels is not None:
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


def createNewModel(group_name, train_data, train_labels, test_data, test_labels,
                   ts, llstm, ulstm, lblstm, do, kc, ld, ud, ne, sb):
    """Creates a new model based on the inputted parameters
    error(fixed after groupB and B2): ld label is ld-=1 if llstm == 0"""

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

    if llstm == 0:  # creates an input dense layer if no there is no lstm input
        model.add(Dense(ud, input_shape=(train_data.shape[1],), activation='relu'))
        ld -= 1
    for i in range(ld - 1):
        model.add(Dense(ud, activation='relu'))
    if llstm == 0:
        ld += 1

    model.add(Dense(1, activation='sigmoid'))

    # TODO: test out accuracy vs precision
    # model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Accuracy()])
    model.fit(train_data, train_labels, epochs=ne, batch_size=sb)
    model_name = f'{ts}_{llstm}_{ulstm}_{lblstm}_{do}_{kc}_{ld}_{ud}_{ne}_{sb}_acc'
    model.save(f'models/{group_name}/{model_name}.keras')

    print(model_name)
    model.evaluate(test_data, test_labels)
    print('------------------')
    createConfusionMatrix(model, model_name, group_name, test_data, test_labels, ts)
    createConfusionMatrix(model, f'{model_name}_full', group_name, train_data, train_labels, ts)


def createNewModel2(group_name, train_data, train_labels, test_data, test_labels,
                   ts, llstm, ulstm, lblstm, do, kc, ld, ud, ne, sb):
    """created for groupE, dense layer has 'n' neurons.
    n = (train_data.shape[1] * train_data.shape[2] * ulstm)"""

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

    # if llstm == 0:  # creates an input dense layer if no there is no lstm input
    #     model.add(Dense(ud, input_shape=(train_data.shape[1],), activation='relu'))
    #     ld -= 1
    # for i in range(ld - 2):
    #     model.add(Dense(ud, activation='relu'))
    # if llstm == 0:
    #     ld += 1
    #  adding fully connected dense layer w/ units == n
    # n = ulstm * llstm * ld * ud
    n = train_data.shape[1] * train_data.shape[2] * ulstm
    model.add(Dense(n, activation='relu'))

    model.add(Dense(1, activation='sigmoid'))

    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])
    model.fit(train_data, train_labels, epochs=ne, batch_size=sb)
    model_name = f'{ts}_{llstm}_{ulstm}_{lblstm}_{do}_{kc}_{ld}_fd_{ne}_{sb}'
    model.save(f'models/{group_name}/{model_name}.keras')

    print(model_name)
    model.evaluate(test_data, test_labels)
    print('------------------')
    createConfusionMatrix(model, model_name, group_name, test_data, test_labels, ts)
    createConfusionMatrix(model, f'{model_name}_full', group_name, train_data, train_labels, ts)


def testResultsCSV(group, training_data, training_labels, test_data, test_labels, optional=None):
    """reads in all models from models/{group} and evaluates them on training and test data
    saves results in a csv"""
    path = f"models/{group}"
    dirs = os.listdir(path)
    groupresults = []

    for file in dirs:
        if file.endswith(".keras"):
            print(file)
            model = keras.models.load_model(f'models/{group}/{file}')
            config = model.get_config()  # Returns pretty much every information about your model
            in_shape = config["layers"][0]["config"]["batch_input_shape"]
            print(in_shape)  # returns a tuple of width, height and channels
            train_data, train_labels = training_data, training_labels
            tst_data, tst_labels = test_data, test_labels
            if len(in_shape) > 2:
                train_data, train_labels = create3dDataset(training_data, training_labels, in_shape[1])
                tst_data, tst_labels = create3dDataset(test_data, test_labels, in_shape[1])

            # evaluate model and save results to array
            # train_prec = model.evaluate(train_data, train_labels)
            # test_prec = model.evaluate(tst_data, tst_labels)
            # groupresults.append([file, train_prec, test_prec])
            createConfusionMatrix(model, f'{file[:-6]}{optional}', train_data, train_labels, 0.7)  # TODO: fix this

    # Create a DataFrame from the combined list
    df = pd.DataFrame(groupresults)

    # Specify the CSV file path (adjust as needed)
    csv_file_path = f'{group}{optional}results.csv'

    # Write the DataFrame to the CSV file
    # df.to_csv(csv_file_path, index=False)

    print(f"Data saved to {csv_file_path}")


def plotGroupResults(filename):
    """get results from csv and plot in graph (WIP: currently outputs blank png)"""
    x = []
    tstp_y = []
    tstl_y = []
    trp_y = []
    trl_y = []
    # over_60 = []
    # over_80 = []
    list = csvToList(filename)
    for item in list:
        # item[1][0] == train_loss, item[1][1] == train_prec, item[2][0] == test_loss, item[2][1] == test_prec
        trn = eval(item[1])  # np.array(item[1])
        tst = eval(item[2])  # np.array(item[2])
        tst_l = tst[0]  # float(tst.item(0)[1:15])
        tst_p = tst[1]  # float(tst.item(0)[21:-1])
        trn_l = trn[0]  # float(trn.item(0)[1:15])
        trn_p = trn[1]  # float(trn.item(0)[21:-1])
        if tst_p > 1: tst_p = 0
        if trn_p > 1: trn_p = 0
        # if tst_p > 0.6: over_60.append(item)
        # if tst_p > 0.8: over_80.append(item)
        x.append(item[0])
        tstp_y.append(tst_p)
        tstl_y.append(tst_l)
        trp_y.append(trn_p)
        trl_y.append(trn_l)

    # save over_60 and over_80 results for groupB
    # listToCSV(over_60)
    # listToCSV(over_80)

    plt.plot(x, tstp_y, label="GroupBTestDataPrecisionResults")
    plt.xlabel('Model Name')
    plt.ylabel('Precision')
    plt.show()
    # plt.savefig(f'models/groupBcm/GroupBTestDataPrecisionResults.png')  # fix this
    # plt.close()


# def testRetrainModels(group_name, training_data, training_labels, prec, r_period, optional=None):
#     """tests out models w/ precision > 'prec' within 'group_name' by retraining every 'r_period' samples"""
#     path = f"models/{group_name}"
#     dirs = os.listdir(path)
#     groupresults = []
#
#     for file in dirs:
#         if file.endswith(".keras"):
#             print(file)
#             model = keras.models.load_model(f'models/{group_name}/{file}')
#             if model prec > prec:
#                 config = model.get_config()  # Returns pretty much every information about your model
#                 in_shape = config["layers"][0]["config"]["batch_input_shape"]
#                 print(in_shape)  # returns a tuple of width, height and channels
#                 if len(in_shape) > 2:
#                     train_data, train_labels = create3dDataset(training_data, training_labels, in_shape[1])




# ------------- data prep for groupB -------------- #
# combineDataToCSV(symbol="SPY", outputsize="2000", interval="5min", time_period="14", optional="_april")
# createTrainingLabelCSV('historical_data/SPY5min14_april.csv', 0.2)
# normalizeListToCSV('historical_data/SPY5min14_april.csv')
# training_data = csvToArray('historical_data/SPY5min14_april_Normalized.csv')[:-12]
# training_data = np.delete(training_data, 4, axis=1)  # removes 'open'
# training_data = np.delete(training_data, 3, axis=1)  # removes 'low'
# training_data = np.delete(training_data, 2, axis=1)  # removes 'high'
# training_labels = csvToArray('historical_data/SPY5min14_april_TrainingLabels0.2.csv')[:-12]

# testResultsCSV('groupB2', training_data, training_labels, training_data, training_labels, optional='_april')

# full_t_data = training_data
# full_l_data = training_labels
# test_size = int(len(training_data) * 0.8)
# test_data = training_data[test_size:]
# test_labels = training_labels[test_size:]
# training_data = training_data[:test_size]  # added after model8 created
# training_labels = training_labels[:test_size]  # added after model8 created
## test_data, test_labels = create3dDataset(test_data, test_labels, 2)
# ----------------------------------------------------- #


# ------ created groupB1 to compare to groupB2 -------- #
# for item in groupB_testResultsOver60.csv, load model, test it on april data,
# data = csvToList('groupB_testResultsOver60.csv')
# groupresults = []
# for item in data:
#
#     model = keras.models.load_model(f'models/groupB/{item[0]}')
#     config = model.get_config()  # Returns pretty much every information about your model
#     in_shape = config["layers"][0]["config"]["batch_input_shape"]
#     # print(in_shape)  # returns a tuple of width, height and channels
#     train_data, train_labels = training_data, training_labels
#     if len(in_shape) > 2:
#         train_data, train_labels = create3dDataset(training_data, training_labels, in_shape[1])
#     createConfusionMatrix(model, f'{item[0][:-6]}_april', train_data, train_labels, 0.7)
#     # evaluate model and save results to array
#     train_prec = model.evaluate(train_data, train_labels)
#     groupresults.append([item[0], train_prec])
# # Create a DataFrame from the combined list
# df = pd.DataFrame(groupresults)
#
# # Specify the CSV file path (adjust as needed)
# csv_file_path = f'groupB1_aprilresults.csv'
#
# # Write the DataFrame to the CSV file
# df.to_csv(csv_file_path, index=False)
#
# print(f"Data saved to {csv_file_path}")
# --------------------------------------------------------------- #


# ------------ Creation of groupB2 using best from groupB ------- #
# list60 = csvToList('groupB_testResultsOver60.csv')
# list60.pop(0)
# for item in list60:
#     # load model, train it on data, save it in groupB2, save confusion matrix
#     model = keras.models.load_model(f'models/groupB/{item[0]}')
#     mname = f'{item[0][:-11]}250_3'
#     # get model config, reshape data if necessary
#     config = model.get_config()  # Returns pretty much every information about your model
#     in_shape = config["layers"][0]["config"]["batch_input_shape"]
#     print(in_shape)  # returns a tuple of width, height and channels
#     train_data, train_labels = training_data, training_labels
#     tst_data, tst_labels = test_data, test_labels
#     if len(in_shape) > 2:
#         train_data, train_labels = create3dDataset(training_data, training_labels, in_shape[1])
#         tst_data, tst_labels = create3dDataset(test_data, test_labels, in_shape[1])
#     model.fit(train_data, train_labels, epochs=250, batch_size=3)
#
#     # save model and cm to groupB2
#     model.save(f'models/groupB2/{mname}.keras')
#
#     model.evaluate(tst_data, tst_labels)
#     createConfusionMatrix(model, mname, tst_data, tst_labels, 0.7)
#     createConfusionMatrix(model, f'{mname}_full', train_data, train_labels, 0.7)
# ------------------------------------------------ #


# -------------- creation of groupC -------------- #
# data = csvToArray('historical_data/SPY5min_20to23_Normalized.csv')[:-12]
# data = np.delete(data, 4, axis=1)  # removes 'open'
# data = np.delete(data, 3, axis=1)  # removes 'low'
# data = np.delete(data, 2, axis=1)  # removes 'high'
# data_labels = csvToArray('historical_data/SPY5min_20to23_TrainingLabels0.2.csv')[:-12]
# test_size = int(len(data) * 0.8)
# test_data = data[test_size:]
# test_labels = data_labels[test_size:]
# training_data = data[:test_size]
# training_labels = data_labels[:test_size]
#
# # thresholds = [0.5, 0.6, 0.7, 0.8, 0.9, 1]
# layers_LSTM = [1, 2, 3]  # [0, 1, 2, 3]
# units_LSTM = [8, 16, 32, 64]  # [4, 8, 16, 32, 64]
# lookback_LSTM = [1, 2, 3]
# dropout = [0, 0.1, 0.2, 0.4, 0.8]
# # kernel_constraints = [None, 5, 2.5, 1]
# layers_Dense = [2, 3, 4]  # [1, 2, 3, 4]
# units_Dense = [8, 16, 32, 64]  # [4, 8, 16, 32, 64]
# # num_epoch = [50, 100, 150, 200]
# # size_batch = [6, 12, 24]
# i = 0  # 25
# j = 0
# createNewModel('groupC', training_data, training_labels, test_data, test_labels,
#                                            0.7, 3, 8, 2, 0, 5, 4, 64, 150, 6)
# createNewModel('groupC', training_data, training_labels, test_data, test_labels,
#                                            0.7, 1, 8, 3, 0, 10, 2, 8, 150, 6)
# createNewModel('groupC', training_data, training_labels, test_data, test_labels,
#                                            0.7, 3, 8, 6, 0, 10, 3, 64, 150, 3)
# createNewModel2('groupC', training_data, training_labels, test_data, test_labels,
#                                            0.7, 1, 8, 3, 0, 10, 2, 8, 150, 6)
# createNewModel('groupC', training_data, training_labels, test_data, test_labels,
#                                            0.7, 1, 8, 3, 0, 10, 2, 8, 150, 6)
# ts, kc, ne, sb = 0.7, 10, 150, 12
# ld, ud = 1, 64
# for do in dropout:
#     for llstm in layers_LSTM:
#         for ulstm in units_LSTM:
#             for lblstm in lookback_LSTM:
#                 # for ld in layers_Dense:
#                 #     for ud in units_Dense:
#                         # for sb in size_batch:
#                         # if j < i:
#                         #     j += 1
#                         # else:
#                             createNewModel2('groupE', training_data, training_labels, test_data, test_labels,
#                                            ts, llstm, ulstm, lblstm, do, kc, ld, ud, ne, sb)
# print('GroupE has completed training!')
# ------------------------------------------------------ #
"""
- make final 'fully connected layer' have neurons equal to ud *ld * ulstm * llstm
- test out kc = 100, or removing altogether
- try out CNN -> LSTM (identified features are fed into recurrent layer)
- test accuracy over precision
- test sigmoid training labels
- test training labels for upside and downside (2 sigmoids instead of 1)
- test model trained on 0.2 training labels on 0.1 training labels
"""

# ------------- data prep for groupC ------------------- #
# for loop creating raw csv data for each month 2021-2023
# years = [2021, 2022, 2023]
# months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
# for yr in years:
#     for m in months:
#         combineDataToCSV_AV(symbol='SPY', interval='1min', month=f'{yr}-{m}', time_period=14)
#
# # combine csv files for dif months into single csv
# df_csv_append = pd.DataFrame()
# path = f"historical_data/SPY1min_raw"
# dirs = os.listdir(path)
# for file in dirs:
#     if file.endswith(".csv"):
#         df = pd.read_csv(f'historical_data/SPY1min_raw/{file}')
#         df_csv_append = pd.concat([df_csv_append, df], ignore_index=True)
# df_csv_append.to_csv(f'historical_data/SPY1min_rawCombined.csv', index=False)
#
# # filter out extended hours data
# df = pd.read_csv('historical_data/SPY1min_rawCombined.csv')
# # Convert the timestamp column to datetime
# df['datetime'] = pd.to_datetime(df['datetime'])
# # Filter rows between 09:30:00 and 16:00:00
# filtered_df = df[(df['datetime'].dt.time >= pd.to_datetime('09:00:00').time()) &
#                  (df['datetime'].dt.time <= pd.to_datetime('16:25:00').time())]
# # Print the filtered DataFrame
# filtered_df.to_csv(f'historical_data/SPY1min_rawCombinedFiltered.csv', index=False)

# create training labels (save csv) -> normalize data (save csv) -> train groupB models w/ this
# createTrainingLabelCSV('historical_data/SPY5min_rawCombinedFiltered.csv', 0.2)
# normalizeListToCSV('historical_data/SPY5min_rawCombinedFiltered.csv')
# --------------------------------------------------- #
