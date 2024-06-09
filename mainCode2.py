import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Flatten, Dropout
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from mainCode import csvToArray, createConfusionMatrix, listToCSV, csvToList
from tensorflow import keras
from keras.callbacks import EarlyStopping
import datetime

"""Creates a CNN autoencoder (32/3, 2, 64/3, 2) to learn patterns in the data,
then feeds predicted features output along w/ the data into a sequential model of 
fully connected dense layers (320, 160, 80, 2) that predicts max upward and downward 
changes that will occur in the upcoming window """

encoder_name = 'autoencoder1'
delta_model_name = 'deltaModel1e'
trade_model_name = 'tradeModel1b_upOnly'
group_name = 'groupCNNa'
threshold = 0.7
trade_window = 12  # distance to predict price delta and trade opportunity
window_size = 20  # window size (for CNN lookback)
ae_epochs = 200  # for autoencoder
ae_batch_size = 6
d_epochs = 200  # for delta model
d_batch_size = 12
t_epochs = 200  # for trade model
t_batch_size = 12
desired_delta = 1
patience = 20
retrain_encoder = False
retrain_delta_model = False
retrain_trade_model = True


def normalizeData():
    # retrieve historical data : [datetime,close,open,high,low,vol,obv,rsi,atr,macd]
    data = csvToList('historical_data/SPY5min_rawCombinedFiltered.csv')  # [:-trade_window]

    # drop vals to not use
    np.delete(data, 5, axis=1)  # removes vol
    np.delete(data, 4, axis=1)   # removes low
    np.delete(data, 3, axis=1)   # removes high
    np.delete(data, 2, axis=1)   # removes open

    # convert date into timestamp then into 'time of day' indicator
    for item in data:
        day = 24 * 60 * 60
        item[0] = datetime.datetime.strptime(item[0], '%Y-%m-%d %H:%M:%S').timestamp()
        item[0] = np.sin(item[0] * (2 * np.pi / day))

    # Convert to pd dataFrame to compute delta values
    df = pd.DataFrame(data)
    print(df)
    df.drop(columns=[0], inplace=True)
    print(df)
    df = df.astype(float)
    unscaled_data = df.copy()
    delta_df = df.diff().fillna(0)  # Fill NaN values with 0 for the first row

    # Convert delta values back to NumPy array
    delta_data = delta_df.to_numpy()

    # Drop delta_data: time; Drop data: close
    np.delete(delta_data, 0, axis=1)
    np.delete(data, 1, axis=1)

    # Concatenate original data and delta values along the feature axis
    combined_data = np.concatenate((data, delta_data), axis=1)
    cd_df = pd.DataFrame(combined_data)
    cd_df.dropna()
    combined_data = cd_df.to_numpy()
    print(combined_data)
    # listToCSV(combined_data, f'historical_data/{group_name}_data/combined_data.csv')
    combined_data = csvToList('historical_data/groupCNNa_data/combined_data.csv')

    # Normalize the data
    # ex. if x has domain [-5,5], scaling [0,1] will turn -2 into a 0.3 (negative vals under 0.5 if even distribution)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_data)

    return unscaled_data, scaled_data


# Create dataset function (preps data for CNN autoencoder and labels for delta_model)
def create_dataset(data, window_size, unscaled_data, trade_window, desired_delta):
    X = []
    y = []
    z = []
    max_upward, max_downward = calculate_max_changes(unscaled_data, trade_window)
    trade_labels = create_trade_labels(unscaled_data, trade_window, desired_delta)
    # currently contains unscaled delta vals
    # y = np.vstack((max_upward, max_downward)).T
    # print('y: ')
    # print(y)
    for i in range(len(data) - window_size - trade_window):
        X.append(data[i:i + window_size])
        y.append([max_upward[i + window_size], max_downward[i + window_size]])
        z.append(trade_labels[i + window_size])
    dataX = np.array(X)
    dataY = np.array(y)
    dataZ = np.array(z)
    print(f'X shape: {dataX.shape}')
    print(f'y shape: {dataY.shape}')
    print(f'z shape: {dataZ.shape}')

    return dataX, dataY, dataZ


# calculates the max upward and max downward changes
def calculate_max_changes(data, trade_window):
    """looks into future 'trade_window' and identifies max up/down price change"""
    max_upward_changes = []
    max_downward_changes = []
    for i in range(len(data) - trade_window):
        # window = data[i:i + trade_window, 0]  # Assuming close price is the 1st feature
        window = data.iloc[i:i + trade_window, 0]
        # print(f'window: {window}')
        # print(f'window.iloc[0] = {window.iloc[0]}')
        max_upward_changes.append(np.max(window) - window.iloc[0])
        max_downward_changes.append(np.min(window) - window.iloc[0])
    return np.array(max_upward_changes), np.array(max_downward_changes)


def create_trade_labels(data, trade_window, desired_delta):
    """looks into the future 'trade_window' and says whether good trade available"""
    up_labels = []
    down_labels = []
    labels = []
    for i in range(len(data) - trade_window):
        initial_price = data.iloc[i, 0]  # Assuming the price is the first feature
        for j in range(1, trade_window + 1):
            future_price = data.iloc[i + j, 0]
            if future_price >= initial_price + desired_delta:
                up_labels.append(1)
                down_labels.append(0)
                labels.append(1)
                break
            # elif future_price <= initial_price - desired_delta:
            #     up_labels.append(0)
            #     down_labels.append(1)
            #     labels.append(1)
            #     break
            elif j == trade_window:
                up_labels.append(0)
                down_labels.append(0)
                labels.append(0)

    up_labels = np.array(up_labels)
    down_labels = np.array(down_labels)
    labels = np.array(labels)
    combined_labels = np.column_stack((up_labels, down_labels))

    return labels  # return 'combined_samples' for 2 outputs, or 'labels' for 1 output


def display_test_results1(model, test_data, test_labels, trade_model_name):
    """for displaying results of trade opportunity model"""
    # Evaluate the model on the test data
    model.evaluate(test_data, test_labels, verbose=0)

    # Create and save confusion matrix
    createConfusionMatrix(model, model_name=trade_model_name, group_name=group_name, t_data=test_data,
                          t_labels=test_labels, threshold=threshold)


def createConfusionMatrices(model, model_name, group_name, t_data, t_labels, threshold):
    """Creates and saves a confusion matrix
    Parameters: model, a chosen name for the model, the group name, and test data appropriate for the model"""
    predictions = model.predict(t_data)
    up_predictions = predictions[:, 0]
    down_predictions = predictions[:, 1]

    # Round to 0 or 1 based on the threshold
    binary_predictions = (predictions > threshold).astype(int)
    binary_up_predictions = np.where(up_predictions >= threshold, 1, 0)
    binary_down_predictions = np.where(down_predictions >= threshold, 1, 0)

    cm = confusion_matrix(y_true=t_labels[:, 0], y_pred=binary_up_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix for Upside Trades")
    plt.savefig(f'models/{group_name}cm/{model_name}_upCM.png')
    plt.close()

    cm = confusion_matrix(y_true=t_labels[:, 1], y_pred=binary_down_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix for Downside Trades")
    plt.savefig(f'models/{group_name}cm/{model_name}_downCM.png')
    plt.close()
    # plt.show()

# Example usage:
# Assuming trade_model, test_combined, and y_test are defined as in the previous code
# display_test_results1(trade_model, test_combined, y_test)


# Evaluate the model and display test results
def display_test_results2(model, test_data, test_labels):
    """for displaying results of price delta model"""
    # Evaluate the model on the test data
    loss, mae = model.evaluate(test_data, test_labels, verbose=0)

    # Generate predictions
    predictions = model.predict(test_data)

    # Display metrics
    print(f"Test MAE: {mae:.4f}")
    print(f"Test Loss: {loss:.4f}")

    # Plot predictions vs actual values
    plt.figure(figsize=(14, 7))
    plt.plot(test_labels[:, 0], label='Actual Max Upward Change', alpha=0.7)
    plt.plot(test_labels[:, 1], label='Actual Max Downward Change', alpha=0.7)
    plt.plot(predictions[:, 0], label='Predicted Max Upward Change', alpha=0.7)
    plt.plot(predictions[:, 1], label='Predicted Max Downward Change', alpha=0.7)
    plt.legend()
    # plt.savefig(f'models/{group_name}/')  # TODO: fix this
    plt.show()

# Example usage of display_test_results
# Assuming test_combined and y_test are defined as in the previous code
# display_test_results2(model, X_test, y_test)


def get_encoder(retrain, num_inputs, X_train, X_test):
    if retrain:
        # Define the CNN Autoencoder model
        input_layer = Input(shape=(window_size, num_inputs))  # window_size * # of inputs
        print(f'input_layer = {input_layer}')

        # Encoder
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
        x = MaxPooling1D(pool_size=2, padding='same')(x)
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
        encoded = MaxPooling1D(pool_size=2, padding='same', name='encoded_layer')(x)

        # Decoder
        x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(encoded)
        x = UpSampling1D(size=2)(x)
        x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
        x = UpSampling1D(size=2)(x)
        decoded = Conv1D(filters=num_inputs, kernel_size=3, activation='sigmoid', padding='same')(x)

        # Compile the model
        autoencoder = Model(input_layer, decoded)
        autoencoder.compile(optimizer='adam', loss='mse')

        # Train the autoencoder
        autoencoder.fit(X_train, X_train, epochs=ae_epochs, batch_size=ae_batch_size, validation_data=(X_test, X_test))
        autoencoder.save(f'models/{group_name}/{encoder_name}.keras')
        ae_model = autoencoder

        # Print evaluation of the autoencoder
        train_loss = autoencoder.evaluate(X_train, X_train)
        print(f"Training Loss: {train_loss}")
        test_loss = autoencoder.evaluate(X_test, X_test)
        print(f"Test Loss: {test_loss}")

    else:
        # # Create/Load the encoder model
        ae_model = keras.models.load_model(f'models/{group_name}/{encoder_name}.keras')

    print('ae model summary: ')
    ae_model.summary()
    encoder = Model(ae_model.input, ae_model.get_layer('max_pooling1d_1').output)
    # encoder = Model(input_layer, encoded)

    return encoder


def get_delta_model(retrain, train_combined, y_train, test_combined, y_test):
    # # Define the price delta prediction model
    # # [samples, window_size / 4, # filters in last CNN]
    # #          -flatten & concat-> [samples, (20 / 4 * 64 = 320) + # of features] -> [samples, 320 + # of features]
    if retrain:
        delta_model = Sequential([
            Dense(320, activation='relu', input_shape=(train_combined.shape[1],)),
            Dropout(0.5),
            Dense(160, activation='relu'),
            Dropout(0.5),
            Dense(80, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='linear')
        ])

        # Compile the model
        delta_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Train the model
        delta_model.fit(train_combined, y_train, epochs=d_epochs, batch_size=d_batch_size,
                        validation_data=(test_combined, y_test), callbacks=[early_stopping])
        delta_model.save(f'models/{group_name}/{delta_model_name}e.keras')

    else:  # Load the model
        delta_model = keras.models.load_model(f'models/{group_name}/{delta_model_name}.keras')

    return delta_model


def get_trade_model(retrain, train_features, trade_labels_train, test_features, trade_labels_test):
    if retrain:
        trade_model = Sequential([
            Dense(340, activation='relu', input_shape=(train_features.shape[1],)),
            Dropout(0.5),
            Dense(170, activation='relu'),
            Dropout(0.5),
            Dense(85, activation='relu'),
            Dropout(0.5),
            Dense(1, activation='sigmoid')  # Assuming binary classification for trade opportunity
        ])

        trade_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.Precision()])

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Train the trade model
        history = trade_model.fit(
            train_features, trade_labels_train,
            epochs=t_epochs,
            batch_size=t_batch_size,
            validation_data=(test_features, trade_labels_test),
            callbacks=[early_stopping]
        )

        trade_model.save(f'models/{group_name}/{trade_model_name}.keras')

    else:  # Load desired model
        trade_model = keras.models.load_model(f'models/{group_name}/{trade_model_name}.keras')

    return trade_model


def run_pipeline():
    # ----create training/test data (X) and training/test labels (y)----
    unscaled_data, scaled_data = normalizeData()
    X, y, z = create_dataset(scaled_data, window_size, unscaled_data, trade_window, desired_delta)

    # Split the data sequentially
    split_index = int(len(X) * 0.8)
    X_train, X_test = X[:split_index], X[split_index:]
    y_train, y_test = y[:split_index], y[split_index:]
    num_inputs = X_train.shape[2]
    print(f'X_train shape: {X_train.shape} ...is [2] == num of inputs?')
    print(f'y_train shape: {y_train.shape} ...is [0] == X_train[0]')
    print(f'X_test shape: {X_test.shape}')
    print(f'y_test shape: {y_test.shape}')
    print(f'num inputs: {num_inputs}')

    encoder = get_encoder(retrain_encoder, num_inputs, X_train, X_test)
    # Encode the train and test data
    encoded_train = encoder.predict(X_train)
    encoded_test = encoder.predict(X_test)

    # Flatten encoded features
    encoded_train_flat = encoded_train.reshape(encoded_train.shape[0], -1)
    encoded_test_flat = encoded_test.reshape(encoded_test.shape[0], -1)

    # Combine encoded features with raw data
    train_combined = np.hstack((encoded_train_flat, X_train[:, -1, :]))
    test_combined = np.hstack((encoded_test_flat, X_test[:, -1, :]))

    print(f'train_combined shape: {train_combined.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'test_combined shape: {test_combined.shape}')
    print(f'y_test shape: {y_test.shape}')

    # Get delta model
    delta_model = get_delta_model(retrain_delta_model, train_combined, y_train, test_combined, y_test)
    display_test_results2(delta_model, test_combined, y_test)

    # Get delta model predictions
    predicted_deltas_train = delta_model.predict(train_combined)
    predicted_deltas_test = delta_model.predict(test_combined)

    # Combine (encoded features + X data) with predicted deltas for the trade model
    train_features = np.hstack(
        (train_combined, predicted_deltas_train))  # changed from encoded_train_flat in tradeModel1
    test_features = np.hstack((test_combined, predicted_deltas_test))  # changed from encoded_test_flat for tradeModel1b

    # Split trading labels into train and test set
    trade_labels_train, trade_labels_test = z[:split_index], z[split_index:]

    print(f'train_features shape: {train_features.shape}')
    print(f'test_features shape: {test_features.shape}')
    print(f'trade_labels_train shape: {trade_labels_train.shape}')
    print(f'trade_labels_test shape: {trade_labels_test.shape}')

    # Get trade model
    trade_model = get_trade_model(retrain_trade_model, train_features, trade_labels_train, test_features, trade_labels_test)
    display_test_results1(trade_model, test_features, trade_labels_test, trade_model_name)
    # createConfusionMatrices(trade_model, trade_model_name, group_name, test_features, trade_labels_test, threshold)

    # Testing model
    t1, t2, z = create_dataset(scaled_data, window_size, unscaled_data, trade_window, desired_delta / 2)
    trade_labels_test = z[split_index:]
    td_name = trade_model_name + '_halvedLabels'
    display_test_results1(trade_model, test_features, trade_labels_test, td_name)
    # createConfusionMatrices(trade_model, trade_model_name, group_name, test_features, trade_labels_test, threshold)


run_pipeline()


"""
inputs for delta model:
- relative time of day (drop delta time of day)
- close price delta (drop close price)
- vol (maybe get rid of this)
- RSI (to indicate overbought/oversold)
- ?RSI delta 
- ATR (to indicate volatility)
- ?ATR delta (to indicate how volatility is changing)
- ?OBV (to indicate current volume flow)
- OBV delta (to indicate how volume flow is changing)
- MACD (to indicate whether we are in uptrend or downtrend)
- ?MACD delta (to indicate how momentum is changing)
- EMA delta (to indicate current momentum)

- close price delta patterns will be identified with CNN
- LSTM might be used to... ???
- LSTM returns output of shape [batch_size, number of units]

output for delta model
- prediction of max up/down price delta for desired window


question: is CNN getting data[i:i+window_size] and being used to predict delta price for i+trade_window
or is it being used to predict delta price for (i+window_size)+trade_window ? 
"""



"""
to use live:
    get latest data (X)
    CNN autoencoder predict,
    delta_model predict,
    trade_model predict (-1)

** recreate live data backtesting by providing limited data at each point (is this already accomplished?)
- put everything into a fn
"""

# # Predict trade opportunities using the trade model
# predictions = (trade_model.predict(test_features) > 0.5).astype(int).flatten()
#
# # Plot the close price along with trade markers
# unscaled_data_test = unscaled_data[split_index:]
# close_prices = unscaled_data_test.iloc[:, 0]  # Assuming the close price is the first feature in the last timestep
#
# plt.figure(figsize=(15, 7))
# plt.plot(close_prices, label='Close Price', color='blue')
# plt.scatter(np.arange(len(close_prices))[predictions == 1], close_prices[predictions == 1],
#             color='red', marker='o', label='Trade Signal')
# plt.xlabel('Time')
# plt.ylabel('Price')
# plt.title('Close Price with Trade Signals')
# plt.legend()
# plt.show()

