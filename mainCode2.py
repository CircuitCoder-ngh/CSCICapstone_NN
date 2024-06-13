import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Flatten, Dropout, LSTM
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from mainCode import csvToArray, createConfusionMatrix, listToCSV, csvToList, create3dDataset
from tensorflow import keras
from keras.callbacks import EarlyStopping
import datetime

"""Creates a CNN autoencoder (32/3, 2, 64/3, 2) to learn patterns in the data,
then feeds predicted features output along w/ the data into a sequential model of 
fully connected dense layers (320, 160, 80, 2) that predicts max upward and downward 
changes that will occur in the upcoming window """

encoder_name = 'autoencoder1'
delta_model_name = 'deltaModel3'  # 2=lb3, 3=lb12+u340
trade_model_name = 'tradeModel7'
# trade models: 3=lb3,4=lb12,5a=lb12+RSTRSF,5=RSTdoRSF+lessDenselayers, 6=moreDenseunits, 7=deltaModel3
one_output = False
group_name = 'groupCNNa'
d_lstm_units = 340
t_lstm_units = 340
d_look_back = 12  # must match deltaModel's lookback
t_look_back = 12   # must match tradeModel's lookback
threshold = 0.3  # default 0.7
up_threshold = 0.7
down_threshold = 0.7
trade_window = 12  # distance to predict price delta and trade opportunity
window_size = 20  # window size (for CNN lookback)
ae_epochs = 200  # for autoencoder
ae_batch_size = 6
d_epochs = 200  # for delta model
d_batch_size = 12
t_epochs = 200  # for trade model
t_batch_size = 12
desired_delta = 0.5
patience = 3
d_num_of_lstm = 1
t_num_of_lstm = 1
retrain_encoder = False
retrain_delta_model = False
retrain_trade_model = False
testing_mode1 = True


def normalizeData(data):
    # drop vals to not use
    # print('-----------')
    # print(data)
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
    # print(df)
    df.drop(columns=[0], inplace=True)
    # print(df)
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
    # print(combined_data)
    # listToCSV(combined_data, f'historical_data/{group_name}_data/combined_data.csv')
    # combined_data = csvToList('historical_data/groupCNNa_data/combined_data.csv')

    # Normalize the data
    # ex. if x has domain [-5,5], scaling [0,1] will turn -2 into a 0.3 (negative vals under 0.5 if even distribution)
    scaler = MinMaxScaler()
    scaled_data = scaler.fit_transform(combined_data)

    return unscaled_data, scaled_data


# Create dataset function (preps data for CNN autoencoder and labels for delta_model)
def create_dataset(data, window_size, unscaled_data, trade_window, desired_delta, future_unscaled_data):
    X = []
    y = []
    z = []
    if not future_unscaled_data is None:
        max_upward, max_downward = calculate_max_changes(future_unscaled_data, trade_window)
        trade_labels = create_trade_labels(future_unscaled_data, trade_window, desired_delta, one_output)
    else:
        max_upward, max_downward = calculate_max_changes(unscaled_data, trade_window)
        trade_labels = create_trade_labels(unscaled_data, trade_window, desired_delta, one_output)
    # currently contains unscaled delta vals
    # y = np.vstack((max_upward, max_downward)).T
    # print('y: ')
    # print(y)
    for i in range(len(data) - window_size):  # '- trade_window' before testMode1
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


def create_trade_labels(data, trade_window, desired_delta, one_output):
    """looks into the future 'trade_window' and says whether good trade available"""
    up_labels = []
    down_labels = []
    labels = []
    for i in range(len(data) - trade_window):
        initial_price = data.iloc[i, 0]  # Assuming the price is the first feature
        for j in range(1, trade_window + 1):
            future_price = data.iloc[i + j, 0]
            if future_price >= initial_price + desired_delta:  # identifies winning up trades
                up_labels.append(1)
                down_labels.append(0)
                labels.append(1)
                break
            elif future_price <= initial_price - desired_delta:  # identifies winning down trades
                up_labels.append(0)
                down_labels.append(1)
                labels.append(1)
                break
            elif j == trade_window:
                up_labels.append(0)
                down_labels.append(0)
                labels.append(0)

    up_labels = np.array(up_labels)
    down_labels = np.array(down_labels)
    labels = np.array(labels)
    combined_labels = np.column_stack((up_labels, down_labels))

    # return 'combined_labels' for 2 outputs, or 'labels' for 1 output
    if one_output:
        return labels
    else:
        return combined_labels


def display_test_results1(model, test_data, test_labels, trade_model_name):
    """for displaying results of trade opportunity model w/ 1 output"""
    # Evaluate the model on the test data
    # model.evaluate(test_data, test_labels, verbose=0)

    t_name = f'{trade_model_name}_{threshold}'
    # Create and save confusion matrix
    createConfusionMatrix(model, model_name=t_name, group_name=group_name, t_data=test_data,
                          t_labels=test_labels, threshold=threshold)


def createConfusionMatrices(model, model_name, group_name, t_data, t_labels, threshold):
    """Creates and saves two confusion matrices to display results of trade opp model w/ 2 outputs
    Parameters: model, a chosen name for the model, the group name, and test data appropriate for the model"""
    predictions = model.predict(t_data)  # .reshape(-1)
    up_predictions = predictions[:, 0]
    down_predictions = predictions[:, 1]
    print(f'predictions: {predictions}')
    print(f'up_pred: {up_predictions}')
    print(f'down_pred: {down_predictions}')

    # loss, prec = model.evaluate(t_data, t_labels, verbose=0)

    # debugging - realized threshold is too high so predictions aren't getting rounded to 1
    # for p, t in zip(predictions, t_labels):
    #     print(f'predicted: {p}')
    #     print(f'actual___: {t}')

    scaler = MinMaxScaler()
    up_predictions = up_predictions.reshape(-1, 1)
    down_predictions = down_predictions.reshape(-1, 1)
    print(f'up_pred: {up_predictions}')
    print(f'down_pred: {down_predictions}')

    scaled_up_predictions = scaler.fit_transform(up_predictions)
    # scaled_up_predictions = np.array(scaled_up_predictions)

    scaled_down_predictions = scaler.fit_transform(down_predictions)

    # Round to 0 or 1 based on the threshold
    # binary_predictions = (predictions > threshold).astype(int)
    binary_up_predictions = np.where(scaled_up_predictions >= up_threshold, 1, 0)
    binary_down_predictions = np.where(scaled_down_predictions >= down_threshold, 1, 0)

    cm = confusion_matrix(y_true=t_labels[:, 0], y_pred=binary_up_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix for Longs")
    plt.savefig(f'models/{group_name}cm/{model_name}_upCM_s{up_threshold}.png')  # 's' to rep scaled labels
    plt.close()

    cm = confusion_matrix(y_true=t_labels[:, 1], y_pred=binary_down_predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot()
    plt.title("Confusion Matrix for Shorts")
    plt.savefig(f'models/{group_name}cm/{model_name}_downCM_s{down_threshold}.png')
    plt.close()
    # plt.show()

# Example usage:
# Assuming trade_model, test_combined, and y_test are defined as in the previous code
# display_test_results1(trade_model, test_combined, y_test)


def plot_signals_chart(trade_model, unscaled_data, features, split_index):
    # Plot the closing prices
    plt.figure(figsize=(10, 6))
    closing_prices = unscaled_data.iloc[:, 0]
    plt.plot(closing_prices, label='Closing Price')

    # Get indices where labels are 1
    labels = trade_model.predict(features)
    up_labels = labels[:, 0]
    down_labels = labels[:, 1]
    up_marker_indices = []
    down_marker_indices = []
    for i in range(len(up_labels)):
        if up_labels[i] == 1:
            up_marker_indices.append(i)
    for i in range(len(down_labels)):
        if down_labels[i] == 1:
            down_marker_indices.append(i)

    scaler = MinMaxScaler()
    up_predictions = up_labels.reshape(-1, 1)
    down_predictions = down_labels.reshape(-1, 1)
    print(f'up_pred: {up_predictions}')
    print(f'down_pred: {down_predictions}')

    scaled_up_predictions = scaler.fit_transform(up_predictions)
    # scaled_up_predictions = np.array(scaled_up_predictions)

    scaled_down_predictions = scaler.fit_transform(down_predictions)

    # Round to 0 or 1 based on the threshold
    # binary_predictions = (predictions > threshold).astype(int)
    binary_up_predictions = np.where(scaled_up_predictions >= up_threshold, 1, 0)
    binary_down_predictions = np.where(scaled_down_predictions >= down_threshold, 1, 0)
    up_marker_indices = np.where(binary_up_predictions == 1)[0]
    down_marker_indices = np.where(binary_down_predictions == 1)[0]
    print(f'up_marker_indices: {up_marker_indices}')
    print(f'down_marker_indices: {down_marker_indices}')
    print(f'closing_prices.iloc[up_marker_indices: {closing_prices.iloc[up_marker_indices]}')
    print(f'closing_prices.iloc[down_marker_indices: {closing_prices.iloc[down_marker_indices]}')

    # Plot markers on the same graph
    plt.scatter(up_marker_indices + split_index, closing_prices.iloc[up_marker_indices + split_index],
                color='green', label='Long', marker='o')
    plt.scatter(down_marker_indices + split_index, closing_prices.iloc[down_marker_indices + split_index],
                color='red', label='Short', marker='o')

    # Add labels and title
    plt.title('Closing Prices with Markers')
    plt.xlabel('Time')
    plt.ylabel('Closing Price')
    plt.legend()
    plt.show()


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
        # delta_model = Sequential([  # used for deltaModel1
        #     Dense(320, activation='relu', input_shape=(train_combined.shape[1],)),
        #     Dropout(0.5),
        #     Dense(160, activation='relu'),
        #     Dropout(0.5),
        #     Dense(80, activation='relu'),
        #     Dropout(0.5),
        #     Dense(2, activation='linear')
        # ])
        if d_num_of_lstm > 0:
            train_combined, y_train = create3dDataset(train_combined, y_train, d_look_back)
            test_combined, y_test = create3dDataset(test_combined, y_test, d_look_back)
        delta_model = Sequential([  # used for deltaModel2
            LSTM(d_lstm_units, activation='relu', input_shape=(train_combined.shape[1], train_combined.shape[2]),
                 return_sequences=False),  # , kernel_constraint=max_norm(kc)
            # output shape of LSTM: (batch_size, lstm_units), or for RST: (batch_size, timesteps, lstm_units)
            # timesteps == train_combined.shape[1]
            Dense(4080, activation='relu'),
            Dropout(0.5),
            # Dense(160, activation='relu'),
            # Dropout(0.5),
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
        delta_model.save(f'models/{group_name}/{delta_model_name}.keras')

    else:  # Load the model
        delta_model = keras.models.load_model(f'models/{group_name}/{delta_model_name}.keras')

    return delta_model


def get_trade_model(retrain, train_features, trade_labels_train, test_features, trade_labels_test):
    if retrain:
        # trade_model = Sequential([  # used for tradeModel1
        #     Dense(340, activation='relu', input_shape=(train_features.shape[1],)),
        #     Dropout(0.5),
        #     Dense(170, activation='relu'),
        #     Dropout(0.5),
        #     Dense(85, activation='relu'),
        #     Dropout(0.5),
        #     Dense(1, activation='sigmoid')  # Assuming binary classification for trade opportunity
        # ])
        if t_num_of_lstm > 0:
            train_features, trade_labels_train = create3dDataset(train_features, trade_labels_train, t_look_back)
            test_features, trade_labels_test = create3dDataset(test_features, trade_labels_test, t_look_back)
        # trade_model = Sequential([
        #     # used for tradeModel2,3,4; tradeModel3 had lookback 3, tradeModel4 had lookback 12
        #     LSTM(t_lstm_units, activation='relu', input_shape=(train_features.shape[1], train_features.shape[2]),
        #          return_sequences=False),  # , kernel_constraint=max_norm(kc)
        #     Dense(340, activation='relu'),
        #     Dropout(0.5),
        #     Dense(170, activation='relu'),
        #     Dropout(0.5),
        #     Dense(85, activation='relu'),
        #     Dropout(0.5),
        #     Dense(2, activation='sigmoid')
        # ])
        trade_model = Sequential([  # made for tradeModel5,6,7
            LSTM(t_lstm_units, activation='relu', input_shape=(train_features.shape[1], train_features.shape[2]),
                 return_sequences=False),  # , kernel_constraint=max_norm(kc)
            # Dropout(0.5),
            # LSTM(int(t_lstm_units/2), activation='relu', return_sequences=False),
            # output shape of LSTM: (batch_size, lstm_units), or for RST: (batch_size, timesteps, lstm_units)
            Dense(4080, activation='relu'),
            Dropout(0.5),
            # Dense(1020, activation='relu'),
            # Dropout(0.5),
            # Dense(510, activation='relu'),
            # Dropout(0.5),
            # Dense(255, activation='relu'),
            # Dropout(0.5),
            # Dense(128, activation='relu'),
            # Dropout(0.5),
            Dense(64, activation='relu'),
            Dropout(0.5),
            Dense(2, activation='sigmoid')
        ])
        # trade_model = Sequential([  # made for tradeModel8
        #     Input(shape=(window_size, 7),
        #     Conv1D(filters=32, kernel_size=3, activation='relu', padding='same'),
        #     # Dropout(0.5),
        #     Dense(4080, activation='relu'),
        #     Dropout(0.5),
        #     Dense(64, activation='relu'),
        #     Dropout(0.5),
        #     Dense(2, activation='sigmoid')
        # ])

        trade_model.compile(optimizer='adam', loss='binary_crossentropy', metrics=[keras.metrics.Precision()])

        # Define early stopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Train the trade model
        trade_model.fit(
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


def run_pipeline(data):
    # ----create training/test data (X) and training/test labels (y)----
    unscaled_data, scaled_data = normalizeData(data)
    X, y, z = create_dataset(scaled_data, window_size, unscaled_data, trade_window, desired_delta, None)

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
    # train_combined_cpy = train_combined.copy()
    # test_combined_cpy = test_combined.copy()

    print(f'train_combined shape: {train_combined.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'test_combined shape: {test_combined.shape}')
    print(f'y_test shape: {y_test.shape}')

    # Get delta model
    delta_model = get_delta_model(retrain_delta_model, train_combined, y_train, test_combined, y_test)
    if d_num_of_lstm > 0:
        train_combined, y_train = create3dDataset(train_combined, y_train, d_look_back)
        test_combined, y_test = create3dDataset(test_combined, y_test, d_look_back)
    # display_test_results2(delta_model, test_combined, y_test)

    # Get delta model predictions
    predicted_deltas_train = delta_model.predict(train_combined)
    predicted_deltas_test = delta_model.predict(test_combined)

    # Combine (encoded features + X data) with predicted deltas for the trade model
    if d_num_of_lstm > 0:
        train_features = np.hstack(
            (train_combined[:, -1, :], predicted_deltas_train))
        test_features = np.hstack((test_combined[:, -1, :], predicted_deltas_test))
    else:
        train_features = np.hstack(
            (train_combined, predicted_deltas_train))  # changed from encoded_train_flat in tradeModel1
        test_features = np.hstack((test_combined, predicted_deltas_test))
        # changed from encoded_test_flat for tradeModel1b

    # Split trading labels into train and test set
    trade_labels_train, trade_labels_test = z[:split_index], z[split_index:]
    test_features2 = test_features.copy()

    print(f'train_features shape: {train_features.shape}')
    print(f'test_features shape: {test_features.shape}')
    print(f'trade_labels_train shape: {trade_labels_train.shape}')
    print(f'trade_labels_test shape: {trade_labels_test.shape}')

    # Get trade model and test it
    trade_model = get_trade_model(retrain_trade_model, train_features, trade_labels_train, test_features, trade_labels_test)
    if t_num_of_lstm > 0:
        # train_features, trade_labels_train = create3dDataset(train_features, trade_labels_train, t_look_back)
        test_features, trade_labels_test = create3dDataset(test_features, trade_labels_test, t_look_back)
    # display_test_results1(trade_model, test_features, trade_labels_test, trade_model_name)
    # createConfusionMatrices(trade_model, trade_model_name, group_name, test_features, trade_labels_test, threshold)
    plot_signals_chart(trade_model, unscaled_data, test_features, split_index)

    # Testing model w/ labels for half the desired delta
    t1, t2, z = create_dataset(scaled_data, window_size, unscaled_data, trade_window, desired_delta / 2)
    trade_labels_test2 = z[split_index:]
    if t_num_of_lstm > 0:
        # train_features, trade_labels_train = create3dDataset(train_features, trade_labels_train, t_look_back)
        test_features2, trade_labels_test2 = create3dDataset(test_features2, trade_labels_test2, t_look_back)
    td_name = trade_model_name + '_halvedLabels'
    # display_test_results1(trade_model, test_features, trade_labels_test, td_name)
    # createConfusionMatrices(trade_model, td_name, group_name, test_features2, trade_labels_test2, threshold)


def run_pipeline2(data, future_data):  # all retrain must be False
    # ----create training/test data (X) and training/test labels (y)----
    unscaled_data, scaled_data = normalizeData(data)
    future_unscaled_data, future_scaled_data = normalizeData(future_data)
    X, y, z = create_dataset(scaled_data, window_size, unscaled_data, trade_window, desired_delta, future_unscaled_data)

    # Split the data sequentially
    # split_index = int(len(X) * 0.8)
    # X_train, X_test = X[:split_index], X[split_index:]
    # y_train, y_test = y[:split_index], y[split_index:]
    num_inputs = X.shape[2]
    encoder = get_encoder(retrain_encoder, num_inputs, X, X)
    # Encode the train and test data
    encoded_train = encoder.predict(X)

    # Flatten encoded features
    encoded_train_flat = encoded_train.reshape(encoded_train.shape[0], -1)

    # Combine encoded features with raw data
    train_combined = np.hstack((encoded_train_flat, X[:, -1, :]))

    # Get delta model
    delta_model = get_delta_model(retrain_delta_model, train_combined, y, train_combined, y)
    if d_num_of_lstm > 0:
        train_combined, y_train = create3dDataset(train_combined, y, d_look_back)
    # display_test_results2(delta_model, test_combined, y_test)

    # Get delta model predictions
    predicted_deltas_train = delta_model.predict(train_combined)

    # Combine (encoded features + X data) with predicted deltas for the trade model
    if d_num_of_lstm > 0:
        train_features = np.hstack(
            (train_combined[:, -1, :], predicted_deltas_train))
    else:
        train_features = np.hstack(
            (train_combined, predicted_deltas_train))  # changed from encoded_train_flat in tradeModel1
        # changed from encoded_test_flat for tradeModel1b

    # Split trading labels into train and test set
    trade_labels_train = z

    # Get trade model and test it
    trade_model = get_trade_model(retrain_trade_model, train_features, trade_labels_train,
                                  train_features, trade_labels_train)
    if t_num_of_lstm > 0:
        train_features, trade_labels_train = create3dDataset(train_features, trade_labels_train, t_look_back)

    return trade_model.predict(train_features), trade_labels_train[-1]


# retrieve historical data : [datetime,close,open,high,low,vol,obv,rsi,atr,macd]
raw_list = csvToList('historical_data/SPY5min_rawCombinedFiltered.csv')  # [:-trade_window]
split_index = int(len(raw_list) * 0.8)
raw_list = raw_list[split_index:]
# raw_list2 = csvToList('historical_data/SPY5min_rawCombinedFiltered.csv')  # [:-trade_window]
# run_pipeline(data)

# set test_data (last 5000 samples), set trade_history = [], feed into model.predict
# if not holding_position and model.predict(-1)[0] == 1 then long, if model.predict(-1)[1] == 1 then short
# position_start_time = i and position_start_price = closing_price[i]
# if pst > 12 then close position; if closing_price[i] - position_start_price >== desired_delta then close position
# if closing position then append to trade_history
data_window_size = 300
starting_balance = 1000
nopen_up_positions = 0
nclosed_up_positions = 0
nopen_down_positions = 0
nclosed_down_positions = 0
trade_history = []
winning_up_trades = 0
losing_up_trades = 0
winning_down_trades = 0
losing_down_trades = 0
max_nopen = 2
all_predictions = []
all_labels = []

for i in range(len(raw_list) - data_window_size - trade_window):
    print(f'list iteration {i}')
    # splice data to only have 'data_window_size' samples
    raw_data1 = csvToList('historical_data/SPY5min_rawCombinedFiltered.csv')[split_index:]  # [:-trade_window]
    raw_data2 = csvToList('historical_data/SPY5min_rawCombinedFiltered.csv')[split_index:]  # [:-trade_window]
    data_window = raw_data1[i:i+data_window_size]
    future_data_window = raw_data2[i:i+data_window_size+trade_window]

    # get data, labels, and predictions for current timestep (i+data_window_size)
    current_predictions, trade_labels_test = run_pipeline2(data_window, future_data_window)

    up_predictions = current_predictions[:, 0]
    down_predictions = current_predictions[:, 1]


    # loss, prec = model.evaluate(t_data, t_labels, verbose=0)

    # debugging - realized threshold is too high so predictions aren't getting rounded to 1
    # for p, t in zip(predictions, t_labels):
    #     print(f'predicted: {p}')
    #     print(f'actual___: {t}')

    scaler = MinMaxScaler()  # TODO: make run_pipeline2 return full predictions so it can be scaled here easier
    up_predictions = up_predictions.reshape(-1, 1)
    down_predictions = down_predictions.reshape(-1, 1)

    scaled_up_predictions = scaler.fit_transform(up_predictions)
    # scaled_up_predictions = np.array(scaled_up_predictions)

    scaled_down_predictions = scaler.fit_transform(down_predictions)

    # Round to 0 or 1 based on the threshold
    binary_up_predictions = np.where(up_predictions >= up_threshold, 1, 0)
    binary_down_predictions = np.where(down_predictions >= down_threshold, 1, 0)

    current_predictions = [binary_up_predictions[-1, 0], binary_down_predictions[-1, 0]]
    print(f'predictions: {current_predictions}')
    # print(f'binup_pred: {binary_up_predictions}')
    # print(f'bindown_pred: {binary_down_predictions}')
    print(f'trade_label: {trade_labels_test}')

    # Store predictions and labels for evaluation
    all_predictions.append(current_predictions)
    all_labels.append(trade_labels_test)

    if current_predictions[0] == 1 and trade_labels_test[0] == 1:
        winning_up_trades += 1
        starting_balance = int(starting_balance * 1.002)
    elif current_predictions[0] == 1 and trade_labels_test[0] == 0:
        losing_up_trades += 1
        starting_balance = int(starting_balance * .998)
    if current_predictions[1] == 1 and trade_labels_test[1] == 1:
        winning_down_trades += 1
        starting_balance = int(starting_balance * 1.002)
    elif current_predictions[1] == 1 and trade_labels_test[1] == 0:
        losing_down_trades += 1
        starting_balance = int(starting_balance * .998)

    accuracy = accuracy_score(all_labels, all_predictions)
    print(f'Model Accuracy: {accuracy:.2f}')
    print(f'up winners: {winning_up_trades}')
    print(f'up losers: {losing_up_trades}')
    print(f'down winners: {winning_down_trades}')
    print(f'down losers: {losing_down_trades}')
    print(f'starting balance: 1000')
    print(f'ending balance: {starting_balance}')

    # saves closing price, the up/down trading labels, the current prediction
    # trade_history.append([unscaled_data.iloc[-1][0], trade_labels_test[-1], current_predictions[-1]])

# After the loop, you can evaluate the model's performance
# For example, using accuracy, precision, recall, F1-score, etc.

accuracy = accuracy_score(all_labels, all_predictions)
print(f'Model Accuracy: {accuracy:.2f}')
print(f'up winners: {winning_up_trades}')
print(f'up losers: {losing_up_trades}')
print(f'down winners: {winning_down_trades}')
print(f'down losers: {losing_down_trades}')
print(f'starting balance: 1000')
print(f'ending balance: {starting_balance}')
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
- change CM to save w/ threshold in name, retest model2 and model2b
- create and test tradeModel1b_downOnly
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

