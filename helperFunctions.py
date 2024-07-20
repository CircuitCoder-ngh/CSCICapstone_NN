from apiCalls import *
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
import matplotlib.pyplot as plt


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