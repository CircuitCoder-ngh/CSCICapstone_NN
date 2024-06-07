import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split
from keras.models import Model, Sequential
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Flatten
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, accuracy_score
from mainCode import csvToArray, createConfusionMatrix

"""Creates a CNN autoencoder (32/3, 2, 64/3, 2) to learn patterns in the data,
then feeds predicted features output along w/ the data into a sequential model of 
fully connected dense layers (320, 160, 80, 2) that predicts max upward and downward 
changes that will occur in the upcoming window """

delta_model_name = 'deltaModel1'
encoder_name = 'autoencoder1'
trade_model_name = 'tradeModel1'
group_name = 'groupCNNa'
threshold = 0.7
trade_window = 12  # distance to predict price delta and trade opportunity
window_size = 20  # window size (for CNN lookback)
epochs = 200
batch_size = 6
d_epochs = 100
d_batch_size = 12

# retrieve historical data : [datetime,close,open,high,low,vol,obv,rsi,atr,macd]
data = csvToArray('historical_data/SPY5min_rawCombinedFiltered.csv')[:-trade_window]

# TODO: make function to create training labels and save them

# drop vals to not use
np.delete(data, 5, axis=1)  # removes vol
np.delete(data, 4, axis=1)   # removes low
np.delete(data, 3, axis=1)   # removes high
np.delete(data, 2, axis=1)   # removes open

# Transform data to delta values # TODO: pick which indicators to diff
# TODO: get all delta vals then combine w norm vals
# copy data, drop vals to not diff (datetime, etc), get diff vals, combine delta w/ data
# delta_data = data[['close', 'OBV', 'EMA']].diff().dropna()

# -------------
# Compute delta values
delta_df = data.diff().fillna(0)  # Fill NaN values with 0 for the first row

# Convert delta values back to NumPy array
delta_data = delta_df.to_numpy()

# Drop delta_data: time; Drop data: close
np.delete(delta_data, 0, axis=1)
np.delete(data, 1, axis=1)

# Concatenate original data and delta values along the feature axis
combined_data = np.concatenate((data, delta_data), axis=1)
# --------------

# Normalize the data # TODO: pick which features to scale
# ex. if x has domain [-5,5], scaling [0,1] will turn -2 into a 0.3 (negative vals under 0.5 if even distribution)
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(combined_data)


# Create dataset function (preps data for CNN autoencoder)
def create_dataset(data, window_size):
    X = []
    for i in range(len(data) - window_size):
        X.append(data[i:i + window_size])
    return np.array(X)


# calculates the max upward and max downward changes
def calculate_max_changes(data, trade_window):
    max_upward_changes = []
    max_downward_changes = []
    for i in range(len(data) - trade_window):
        window = data[i:i + trade_window, 0]  # Assuming stock_price is the first feature # TODO: fix this
        max_upward_changes.append(np.max(window) - window[0])
        max_downward_changes.append(np.min(window) - window[0])
    return np.array(max_upward_changes), np.array(max_downward_changes)


def display_test_results1(model, test_data, test_labels):
    """for displaying results of trade opportunity model"""
    # Evaluate the model on the test data
    loss, accuracy = model.evaluate(test_data, test_labels, verbose=0)

    # Display accuracy and loss
    print(f"Test Accuracy: {accuracy * 100:.2f}%")
    print(f"Test Loss: {loss:.4f}")

    # Create and save confusion matrix
    createConfusionMatrix(model, model_name=trade_model_name, group_name=group_name, t_data=test_data,
                          t_labels=test_labels, threshold=threshold)


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
    plt.show()

# Example usage of display_test_results
# Assuming test_combined and y_test are defined as in the previous code
# display_test_results2(model, X_test, y_test)


# TODO: put everything below into a function

# create training/test data (X) and training/test labels (y)
X = create_dataset(scaled_data, window_size)
max_upward, max_downward = calculate_max_changes(data, window_size)  # currently contains unscaled delta vals
y = np.vstack((max_upward, max_downward)).T  # TODO: confirm X and y are same length

# Split the data sequentially
split_index = int(len(X) * 0.8)
X_train, X_test = X[:split_index], X[split_index:]
y_train, y_test = y[:split_index], y[split_index:]
num_inputs = X_train.shape[2]  # TODO: confirm this is correct, should equal num of inputs
print(f'X_train shape: {X_train.shape} ...is [2] == num of inputs?')
print(f'y_train shape: {y_train.shape} ...is [0] == X_train[0]')

# Define the CNN Autoencoder model
input_layer = Input(shape=(window_size, num_inputs))  # window_size * # of inputs

# Encoder
x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(input_layer)
x = MaxPooling1D(pool_size=2, padding='same')(x)
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(x)
encoded = MaxPooling1D(pool_size=2, padding='same')(x)

# Decoder
x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(encoded)
x = UpSampling1D(size=2)(x)
x = Conv1D(filters=32, kernel_size=3, activation='relu', padding='same')(x)
x = UpSampling1D(size=2)(x)
decoded = Conv1D(filters=3, kernel_size=3, activation='sigmoid', padding='same')(x)

# Compile the model
autoencoder = Model(input_layer, decoded)
autoencoder.compile(optimizer='adam', loss='mse')

# Train the autoencoder
autoencoder.fit(X_train, X_train, epochs=epochs, batch_size=batch_size, validation_data=(X_test, X_test))
autoencoder.save(f'models/{group_name}/{encoder_name}.keras')

# Print evaluation of the autoencoder
train_loss = autoencoder.evaluate(X_train, X_train)
print(f"Training Loss: {train_loss}")
test_loss = autoencoder.evaluate(X_test, X_test)
print(f"Test Loss: {test_loss}")

# Create the encoder model
encoder = Model(input_layer, encoded)

# Encode the train and test data
encoded_train = encoder.predict(X_train)
encoded_test = encoder.predict(X_test)

# Flatten encoded features
encoded_train_flat = encoded_train.reshape(encoded_train.shape[0], -1)
encoded_test_flat = encoded_test.reshape(encoded_test.shape[0], -1)

# Combine encoded features with raw data
train_combined = np.hstack((encoded_train_flat, X_train[:, -1, :]))
test_combined = np.hstack((encoded_test_flat, X_test[:, -1, :]))


# Define the price delta prediction model
# [samples, window_size / 4, # filters in last CNN]
#          -flatten & concat-> [samples, (20 / 4 * 64 = 320) + # of features] -> [samples, 320 + # of features]
delta_model = Sequential([
    Dense(320, activation='relu', input_shape=(train_combined.shape[1],)),
    Dense(160, activation='relu'),
    Dense(80, activation='relu'),
    Dense(2, activation='linear')
])

# Compile the model
delta_model.compile(optimizer='adam', loss='mse', metrics=['mae'])

# Train the model
delta_model.fit(train_combined, y_train, epochs=d_epochs, batch_size=d_batch_size, validation_data=(test_combined, y_test))
delta_model.save(f'models/groupCNNa/{delta_model_name}.keras')

display_test_results2(delta_model, test_combined, y_test)

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
"""