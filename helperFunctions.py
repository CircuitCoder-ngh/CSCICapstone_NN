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


def combineDailyATRToCSV_AV(symbol, interval, time_period, optional=None):

    data_atr = getATR_AV(symbol=symbol, interval=interval, time_period=time_period, month=None)

    # Initialize the combined list
    combined_list = []

    # Merge indicator values
    for sample_atr in data_atr['Technical Analysis: ATR']:
        combined_sample = {
            'datetime': sample_atr,
            'atr': data_atr['Technical Analysis: ATR'][sample_atr]['ATR']
        }
        combined_list.append(combined_sample)

    # puts list in chronological order (loc[0] is oldest, loc[i] is most recent
    combined_list = reverse_list(combined_list)

    # Create a DataFrame from the combined list
    df = pd.DataFrame(combined_list)

    filtered_df = df[df['datetime'] >= '2021-01-01']

    csv_file_path = f'historical_data/{symbol}{interval}_ATR.csv'

    # Write the DataFrame to the CSV file
    filtered_df.to_csv(csv_file_path, index=False)

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
    """Converts input tensor from 2d to 3d"""
    dataX, dataY = [], []
    for i in range(len(dataset) - look_back):  # removed -1 after tradeModel7
        a = dataset[i:(i + look_back), :]  # [...:i, 0] would only add the val from col[0]
        dataX.append(a)
        if data_labels is not None:
            dataY.append(data_labels[i + look_back])

    dataX = np.array(dataX)
    dataY = np.array(dataY)
    print(dataX.shape)
    print(dataY.shape)
    return dataX, dataY


def addDailyATRLines(d):
    # TODO: add ATR lines to data before saving to 'current.csv' in refresh live data fn
    # TODO: add distance from ATR lines as input
    atr_vals = pd.DataFrame(csvToList('historical_data/SPYdaily_ATR.csv'))
    data = pd.DataFrame(d)
    """
    data; (datetime,close,open,high,low,vol,obv,rsi,atr,macd)
    atr_vals; (datetime,atr)
    for item in data:
        if new day -> save prev_day_close, get atr w/ matching date, and calculate new atr lines
        calculate item distance from each line
        item.append(distance from atr lines)
    """
    # Convert the 'datetime' columns to datetime objects
    data['datetime'] = pd.to_datetime(data['datetime'])
    atr_vals['datetime'] = pd.to_datetime(atr_vals['datetime'])

    # Extract the date part from the 'datetime' column in data
    data['date'] = data['datetime'].dt.date

    # Merge ATR values into the main data on the 'date' column
    data = data.merge(atr_vals, left_on='date', right_on='datetime', how='left', suffixes=('', '_atr'))

    # Initialize new columns for distances from upper and lower lines
    for i in range(1, 7):
        data[f'distance_from_upperLine{i}'] = None
        data[f'distance_from_lowerLine{i}'] = None

    prev_day_close = None
    upper_lines = None
    lower_lines = None

    for i in range(len(data)):
        current_row = data.iloc[i]

        # Check if a new day starts
        if i == 0 or current_row['date'] != data.iloc[i - 1]['date']:
            # TODO: this calcs atr lines from 4:30 close instead of 4:00 (aka 16:00)
            prev_day_close = data.iloc[i - 1]['close']
            atr = current_row['atr']

            # Calculate upper and lower lines based on ATR multiples
            upper_lines = [prev_day_close + (multiplier * atr) for multiplier in [0.236, 0.382, 0.5, 0.618, 0.786, 1]]
            lower_lines = [prev_day_close - (multiplier * atr) for multiplier in [0.236, 0.382, 0.5, 0.618, 0.786, 1]]

        # Calculate distances from upper and lower lines for each row
        for j in range(1, 7):
            data.at[i, f'distance_from_upperLine{j}'] = abs(current_row['close'] - upper_lines[j])
            data.at[i, f'distance_from_lowerLine{j}'] = abs(current_row['close'] - lower_lines[j])

    # Drop the extra 'datetime_atr' and 'date' columns
    data = data.drop(columns=['datetime_atr', 'date'])

    return data


# TODO: fix this
# def combineAndFilter(data):
# # ------------- data prep for groupC ------------------- #
# #     for loop creating raw csv data for each month 2021-2023
#     years = [2021, 2022, 2023]
#     months = ['01', '02', '03', '04', '05', '06', '07', '08', '09', '10', '11', '12']
#     for yr in years:
#         for m in months:
#             combineDataToCSV_AV(symbol='SPY', interval='1min', month=f'{yr}-{m}', time_period=14)
#
#     # combine csv files for dif months into single csv
#     df_csv_append = pd.DataFrame()
#     path = f"historical_data/SPY1min_raw"
#     dirs = os.listdir(path)
#     for file in dirs:
#         if file.endswith(".csv"):
#             df = pd.read_csv(f'historical_data/SPY1min_raw/{file}')
#             df_csv_append = pd.concat([df_csv_append, df], ignore_index=True)
#     df_csv_append.to_csv(f'historical_data/SPY1min_rawCombined.csv', index=False)
#
#     # filter out extended hours data
#     df = pd.read_csv('historical_data/SPY1min_rawCombined.csv')
#     # Convert the timestamp column to datetime
#     df['datetime'] = pd.to_datetime(df['datetime'])
#     # Filter rows between 09:30:00 and 16:00:00
#     filtered_df = df[(df['datetime'].dt.time >= pd.to_datetime('09:00:00').time()) &
#                      (df['datetime'].dt.time <= pd.to_datetime('16:25:00').time())]
#     # Print the filtered DataFrame
#     filtered_df.to_csv(f'historical_data/SPY1min_rawCombinedFiltered.csv', index=False)
#
#     # create training labels (save csv) -> normalize data (save csv) -> train groupB models w/ this
#     createTrainingLabelCSV('historical_data/SPY5min_rawCombinedFiltered.csv', 0.2)
#     normalizeListToCSV('historical_data/SPY5min_rawCombinedFiltered.csv')
