import datetime
import time

# import discord
from keras.callbacks import EarlyStopping
from keras.layers import Input, Conv1D, MaxPooling1D, UpSampling1D, Dense, Dropout, LSTM, Flatten, AveragePooling1D, Layer
from keras.models import Model, Sequential
from sklearn.preprocessing import MinMaxScaler
from tensorflow import keras
import tensorflow as tf
from helperFunctions import *


"""Changes from mainCode2...
1. adjusted timestamp input to accurately represent relative time of day (math was wrong)
2. adjusted create_trade_labels, still needs rework
- created ae2b + dm4b + tm8b -> 8bb=mainCode2.ae8+fixedTimeInput, 8b=mainCode2.ae8+fixedTimeInput+aebatchsize12
3. adjust creation of training data ( ) to include FVG and distance from daily satyATR lines
"""

encoder_name = 'autoencoder2bb'
delta_model_name = 'deltaModel1'
trade_model_name = 'tradeModel1'
group_name = 'groupTransformer'
up_threshold = 0.9
down_threshold = 0.9
trade_window = 12    # distance to predict price delta and trade opportunity
window_size = 40  # window size (for CNN lookback)
ae_epochs = 200  # for autoencoder
ae_batch_size = 6
d_epochs = 200  # for delta model
d_batch_size = 6
delta_lookback = 20
t_epochs = 200  # for trade model
t_batch_size = 6
t_lookback = 20
desired_delta = 1  # 1 for training, 0.5 for testing
patience = 10
retrain_encoder = True
retrain_delta_model = True
retrain_trade_model = True

# deprecated variables
one_output = False  # used to indicate use of LSTM
d_lstm_units = 340
t_lstm_units = 340
d_look_back = 12  # must match deltaModel's lookback
t_look_back = 12  # must match tradeModel's lookback
threshold = 0.3  # default 0.7
d_num_of_lstm = 1
t_num_of_lstm = 1


def relative_time_of_day(datetime_str):
    # Parse the datetime string
    dt = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")

    # Calculate the number of seconds since the start of the day
    seconds_since_start_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second

    return seconds_since_start_of_day


def normalizeData(data):
    # drop vals to not use
    np.delete(data, 5, axis=1)  # removes vol
    np.delete(data, 4, axis=1)  # removes low
    np.delete(data, 3, axis=1)  # removes high
    np.delete(data, 2, axis=1)  # removes open

    # convert date into timestamp into 'time of day' indicator
    for item in data:
        day = 24 * 60 * 60
        # tp = item[0]
        item[0] = relative_time_of_day(item[0])
        item[0] = np.sin(item[0] * (2 * np.pi / day))
        # print(f'{tp} == {item[0]}')

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

    # Drop delta_data (anything that we aren't interested in RoC of): time
    np.delete(delta_data, 0, axis=1)

    # Drop data (anything not roughly normalized that might trend over time): close
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

    listToCSV(unscaled_data, f'historical_data/{group_name}_unscaled_data.csv')
    listToCSV(scaled_data, f'historical_data/{group_name}_scaled_data.csv')
    return unscaled_data, scaled_data


# Create dataset function (preps data for CNN autoencoder and labels for delta_model)
def create_dataset(data, window_size, unscaled_data, trade_window, desired_delta):
    X = []
    y = []
    z = []
    print('XXXXXXXXXXXXXXXXXX')
    print(f'data shape == {data.shape}')
    print(f'unscaled data shape == {unscaled_data}')
    print('XXXXXXXXXXXXXXXXXXX')
    max_upward, max_downward = calculate_max_changes(unscaled_data, trade_window)
    trade_labels = create_trade_labels(unscaled_data, trade_window, desired_delta, one_output)
    for i in range(len(data) - window_size - trade_window):
        X.append(data[i:i + window_size])
        y.append([max_upward[i + window_size], max_downward[i + window_size]])
        z.append(trade_labels[i + window_size])
    # currently contains unscaled delta vals
    # y = np.vstack((max_upward, max_downward)).T
    # print('y: ')
    # print(y)

    dataX = np.array(X)
    dataY = np.array(y)
    dataZ = np.array(z)
    print(f'X shape: {dataX.shape}')
    print(f'y shape: {dataY.shape}')
    print(f'z shape: {dataZ.shape}')

    return dataX, dataY, dataZ


# Create dataset function (preps data for CNN autoencoder and labels for delta_model)
def create_X_dataset(data, window_size, unscaled_data, trade_window, desired_delta):
    X = []
    # max_upward, max_downward = calculate_max_changes(unscaled_data, trade_window)
    # trade_labels = create_trade_labels(unscaled_data, trade_window, desired_delta, one_output)
    for i in range(len(data) - window_size):  # excludes '-trade_window' b/c that is only needed when making labels
        X.append(data[i:i + window_size])
    return np.array(X)


# calculates the max upward and max downward changes
def calculate_max_changes(data, trade_window):
    """looks into future 'trade_window' and identifies max up/down price change,
    assumes close is 1st feature aka data[0]"""
    max_upward_changes = []
    max_downward_changes = []
    for i in range(len(data) - trade_window):
        # window = data[i:i + trade_window, 0]  # Assuming close price is the 1st feature
        window = data.iloc[i:i + trade_window, 0]
        # print(f'window: {window}')
        # print(f'window.iloc[0] = {window.iloc[0]}')
        max_up = np.max(window) - window.iloc[0]
        max_down = np.min(window) - window.iloc[0]
        if max_up > 0:
            max_upward_changes.append(max_up)
        else:
            max_upward_changes.append(0)
        if max_down < 0:
            max_downward_changes.append(max_down)
        else:
            max_downward_changes.append(0)
    return np.array(max_upward_changes), np.array(max_downward_changes)


def create_trade_labels(data, trade_window, desired_delta, one_output, deltaModel=None):
    """looks into the future 'trade_window' and says whether good trade available,
    assumes 'close' is data[0]"""
    up_labels = []
    down_labels = []
    for i in range(len(data) - trade_window):
        """if (np.max(future_price) - initial_price) > desired_delta) && (desired_delta > min_desired_delta)"""
        # TODO: train desired delta based on deltaModel predictions
        window = data.iloc[i:i + trade_window, 0]
        initial_price = window.iloc[0]
        if np.max(window) - initial_price > desired_delta:
            up_labels.append(1)
        else:
            up_labels.append(0)
        if (np.min(window) - initial_price) * -1 > desired_delta:
            down_labels.append(1)
        else:
            down_labels.append(0)

    up_labels = np.array(up_labels)
    down_labels = np.array(down_labels)
    combined_labels = np.column_stack((up_labels, down_labels))

    return combined_labels


def createConfusionMatrices(model, model_name, group_name, t_data, t_labels):
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


def plot_signals_chart(trade_model, unscaled_data, features, split_index, t_labels):
    # Initialize the plot and closing prices
    plt.figure(figsize=(10, 6))
    print(f'features shape == {features.shape}')
    print(f'unscaled data shape == {unscaled_data.shape}')
    labels = trade_model.predict(features)
    dif = len(unscaled_data) - len(labels)
    closing_prices = unscaled_data.iloc[dif:, 0].reset_index(drop=True)
    print(f'labels shape == {labels.shape}')
    print(f't_labels shape == {t_labels.shape}')
    print(f'closing_prices shape == {closing_prices.shape}')

    # Get indices where labels are 1
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

    scaled_up_predictions = scaler.fit_transform(up_predictions)
    # scaled_up_predictions = np.array(scaled_up_predictions)

    scaled_down_predictions = scaler.fit_transform(down_predictions)

    # Round to 0 or 1 based on the threshold
    # binary_predictions = (predictions > threshold).astype(int)
    binary_up_predictions = np.where(scaled_up_predictions >= up_threshold, 1, 0)
    binary_down_predictions = np.where(scaled_down_predictions >= down_threshold, 1, 0)
    up_marker_indices = np.where(binary_up_predictions == 1)[0]
    down_marker_indices = np.where(binary_down_predictions == 1)[0]
    tup_marker_indices = np.where(t_labels[:, 0] == 1)[0]
    tdown_marker_indices = np.where(t_labels[:, 1] == 1)[0]
    # print(f'up_marker_indices: {up_marker_indices}')
    # print(f'down_marker_indices: {down_marker_indices}')
    # print(f'closing_prices.iloc[up_marker_indices: {closing_prices.iloc[up_marker_indices]}')
    # print(f'closing_prices.iloc[down_marker_indices: {closing_prices.iloc[down_marker_indices]}')
    # print(f'len closing_prices: {len(closing_prices)}')
    # print(f'len binup predictions: {len(binary_up_predictions)}')
    # print(f'len features: {len(features)}')

    # Plot markers on the same graph
    # plt.plot(closing_prices, label='Closing Price')
    # plt.scatter(tup_marker_indices, closing_prices.iloc[tup_marker_indices],  # '+ dif'
    #             color='green', label='Long', marker='o')
    # plt.scatter(tdown_marker_indices, closing_prices.iloc[tdown_marker_indices],
    #             color='red', label='Short', marker='o')
    #
    # # Add lines based on the conditions
    # for idx in tup_marker_indices:
    #     x_values = [idx, idx + trade_window]
    #     y_values = [closing_prices.iloc[idx], closing_prices.iloc[idx] + desired_delta]
    #     # if t_labels[idx, 0] == 1:
    #     if up_labels[idx] == 1:
    #         plt.plot(x_values, y_values, color='green')
    #     else:
    #         plt.plot(x_values, y_values, color='black')
    #
    # for idx in tdown_marker_indices:
    #     x_values = [idx, idx + trade_window]
    #     y_values = [closing_prices.iloc[idx], closing_prices.iloc[idx] - desired_delta]
    #     # if t_labels[idx, 1] == 1:
    #     if down_labels[idx] == 1:
    #         plt.plot(x_values, y_values, color='red')
    #     else:
    #         plt.plot(x_values, y_values, color='black')
    #
    # wu = 0
    # lu = 0
    # wd = 0
    # ld = 0
    # # up_labels
    # print(f'up labels shape == {up_labels.shape}')
    # print(f'up labels len == {len(up_labels)}')
    # for i in range(len(up_labels)):
    #     if up_labels[i] == 1:
    #         if up_labels[i] == t_labels[i, 0]:
    #             wu += 1
    #         else:
    #             lu += 1
    # for i in range(len(down_labels)):
    #     if down_labels[i] == 1:
    #         if down_labels[i] == t_labels[i, 1]:
    #             wd += 1
    #         else:
    #             ld += 1
    #
    # print(f'wu == {wu}, lu == {lu}')
    # print(f'wd == {wd}, ld == {ld}')
    # acc = (wu + wd) / (wu + wd + lu + ld)
    # print(f'accuracy == {acc}')
    #
    # # Add labels and title
    # plt.title('Closing Prices with Markers')
    # plt.xlabel('Time')
    # plt.ylabel('Closing Price')
    # plt.legend()
    # plt.show()


def plot_predictions(trade_model, unscaled_data, features, t_labels):
    """
    Plots a line chart from the data and adds markers at points where predictions are 1.
    Aligns predictions with the end of the data if they are different lengths.
    Also plots actual values if provided.

    Parameters:
    data (list or array-like): The data to plot.
    predictions (list or array-like): The predictions from the Keras model with two channels (up and down).
    actual_vals (list or array-like, optional): The actual values to plot (up and down).
    """
    # (data, predictions, actual_vals)
    print(f'unscaled_data.shape == {unscaled_data.shape}')
    unscaled_data = unscaled_data[:-trade_window]
    predictions = trade_model.predict(features)
    dif = unscaled_data.shape[0] - features.shape[0]
    data = unscaled_data.iloc[dif:, 0].reset_index(drop=True)

    plt.figure(figsize=(10, 6))

    # Plot the data as a line chart
    plt.plot(data, label='Data', color='blue')

    print(f'unscaled data shape == {unscaled_data.shape}')
    print(f'features shape == {features.shape}')
    print(f'dif == {dif}')
    print(f'data shape == {data.shape}')
    print(f'predictions shape == {predictions.shape}')

    up_labels = predictions[:, 0]
    down_labels = predictions[:, 1]
    scaler = MinMaxScaler()
    up_predictions = up_labels.reshape(-1, 1)
    down_predictions = down_labels.reshape(-1, 1)
    scaled_up_predictions = scaler.fit_transform(up_predictions)
    scaled_down_predictions = scaler.fit_transform(down_predictions)

    # Round to 0 or 1 based on the threshold
    binary_up_predictions = np.where(scaled_up_predictions >= up_threshold, 1, 0)
    binary_down_predictions = np.where(scaled_down_predictions >= down_threshold, 1, 0)
    combined_predictions = np.column_stack((binary_up_predictions, binary_down_predictions))

    print(f'combined preds shape == {combined_predictions.shape}')
    print(f't_labels shape == {t_labels.shape}')

    # Calculate the starting index to align predictions with the end of the data
    start_index = len(data) - len(t_labels)
    print(f'start index == {start_index}')
    # Plot the actual values if provided
    if t_labels is not None:
        for i, (up_pred, down_pred) in enumerate(t_labels):
            if up_pred == 1:
                x1 = start_index + i
                y1 = data[start_index + i]
                # plt.plot(x1, y1, color='black', marker='o')
                x_vals = [x1, x1 + trade_window]
                y_vals = [y1, y1 + desired_delta]
                plt.plot(x_vals, y_vals, color="green")
            if down_pred == 1:
                x1 = start_index + i
                y1 = data[start_index + i]
                # plt.plot(x1, y1, color='black', marker='o')
                x_vals = [x1, x1 + trade_window]
                y_vals = [y1, y1 - desired_delta]
                plt.plot(x_vals, y_vals, color="red")

    # Calculate the starting index to align predictions with the end of the data
    start_index = len(data) - len(predictions)
    wu = 0
    lu = 0
    wd = 0
    ld = 0
    # Add markers for up and down predictions
    for i, (up_pred, down_pred) in enumerate(combined_predictions):
        if up_pred == 1:
            # TODO: figure out why '-12' made predictions line up w/ actual values
            if t_labels[i, 0] == 1:
                plt.plot(start_index + i, data[start_index + i], color='green', marker='o')
                wu += 1
            else:
                plt.plot(start_index + i, data[start_index + i], color='k', marker='o')
                lu += 1
        if down_pred == 1:
            if t_labels[i, 1] == 1:
                wd += 1
                plt.plot(start_index + i, data[start_index + i], color='red', marker='o')
            else:
                ld += 1
                plt.plot(start_index + i, data[start_index + i], color='k', marker='o')

    print(f'wu == {wu}, lu == {lu}')
    print(f'wd == {wd}, ld == {ld}')
    acc = (wu + wd) / (wu + wd + lu + ld)
    print(f'accuracy == {acc}')

    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Line Chart with Predictions')
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

        # Define the EarlyStopping callback
        early_stopping = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True)

        # Train the autoencoder
        autoencoder.fit(X_train, X_train, epochs=ae_epochs, batch_size=ae_batch_size, validation_data=(X_test, X_test),
                        callbacks=[early_stopping])
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
    encoder = Model(ae_model.input, ae_model.get_layer('encoded_layer').output)

    # encoder = Model(input_layer, encoded)

    return encoder


class MultiHeadAttention(Layer):
    def __init__(self, d_model, num_heads):
        super(MultiHeadAttention, self).__init__()
        self.num_heads = num_heads
        self.d_model = d_model

        assert d_model % self.num_heads == 0, "d_model must be divisible by num_heads"

        self.depth = d_model // self.num_heads

        self.wq = Dense(d_model)
        self.wk = Dense(d_model)
        self.wv = Dense(d_model)
        self.dense = Dense(d_model)

    def split_heads(self, x, batch_size):
        x = tf.reshape(x, (batch_size, -1, self.num_heads, self.depth))
        return tf.transpose(x, perm=[0, 2, 1, 3])

    def call(self, v, k, q):
        batch_size = tf.shape(q)[0]

        q = self.wq(q)
        k = self.wk(k)
        v = self.wv(v)

        q = self.split_heads(q, batch_size)
        k = self.split_heads(k, batch_size)
        v = self.split_heads(v, batch_size)

        scaled_attention, attention_weights = self.scaled_dot_product_attention(q, k, v)
        scaled_attention = tf.transpose(scaled_attention, perm=[0, 2, 1, 3])
        concat_attention = tf.reshape(scaled_attention, (batch_size, -1, self.d_model))

        output = self.dense(concat_attention)

        return output, attention_weights

    def scaled_dot_product_attention(self, q, k, v):
        matmul_qk = tf.matmul(q, k, transpose_b=True)
        dk = tf.cast(tf.shape(k)[-1], tf.float32)
        scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

        attention_weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
        output = tf.matmul(attention_weights, v)

        return output, attention_weights

def create_cnn_with_attention_for_delta_model(input_shape, d_model, num_heads):
    inputs = Input(shape=input_shape)

    # CNN layers with average pooling
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = AveragePooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = AveragePooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = AveragePooling1D(pool_size=2, padding='same')(x)

    # Multi-Head Attention layer
    attention, _ = MultiHeadAttention(d_model=d_model, num_heads=num_heads)(x, x, x)

    # Flatten and fully connected layers
    x = Flatten()(attention)
    x = Dense(4080, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(80, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation='linear')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_delta_model(retrain, train_combined, y_train, test_combined, y_test):
    # # Define the price delta prediction model
    # # [samples, window_size / 4, # filters in last CNN]
    # #          -flatten & concat-> [samples, (20 / 4 * 64 = 320) + # of features] -> [samples, 320 + # of features]
    if retrain:
        # Ensure the input shape is correct for the CNN model
        train_combined, y_train = create3dDataset(train_combined, y_train, delta_lookback)
        test_combined, y_test = create3dDataset(test_combined, y_test, delta_lookback)
        input_shape = (train_combined.shape[1], train_combined.shape[2])

        # Create the CNN model with multi-head attention and average pooling
        d_model = 256  # Adjust as needed
        num_heads = 4  # Adjust as needed
        delta_model = create_cnn_with_attention_for_delta_model(input_shape, d_model, num_heads)
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

def create_cnn_with_attention_for_trade_model(input_shape, d_model, num_heads):
    inputs = Input(shape=input_shape)

    # CNN layers with average pooling
    x = Conv1D(filters=64, kernel_size=3, activation='relu', padding='same')(inputs)
    x = AveragePooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(filters=128, kernel_size=3, activation='relu', padding='same')(x)
    x = AveragePooling1D(pool_size=2, padding='same')(x)
    x = Conv1D(filters=256, kernel_size=3, activation='relu', padding='same')(x)
    x = AveragePooling1D(pool_size=2, padding='same')(x)

    # Multi-Head Attention layer
    attention, _ = MultiHeadAttention(d_model=d_model, num_heads=num_heads)(x, x, x)

    # Flatten and fully connected layers
    x = Flatten()(attention)
    x = Dense(4080, activation='relu')(x)
    x = Dropout(0.5)(x)
    x = Dense(64, activation='relu')(x)
    x = Dropout(0.5)(x)
    outputs = Dense(2, activation='sigmoid')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model


def get_trade_model(retrain, train_features, trade_labels_train, test_features, trade_labels_test):
    if retrain:
        # Ensure the input shape is correct for the CNN model
        train_features, trade_labels_train = create3dDataset(train_features, trade_labels_train, t_lookback)
        test_features, trade_labels_test = create3dDataset(test_features, trade_labels_test, t_lookback)
        input_shape = (train_features.shape[1], train_features.shape[2])

        # Create the CNN model with multi-head attention and average pooling
        d_model = 256  # Adjust as needed
        num_heads = 4  # Adjust as needed
        trade_model = create_cnn_with_attention_for_trade_model(input_shape, d_model, num_heads)

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


def refresh_live_data(interval):
    combineDataToCSV_AV(symbol='SPY', interval=interval, time_period=14, month=None, optional=True)
    # filter out extended hours data
    df = pd.read_csv('historical_data/current.csv')
    # Convert the timestamp column to datetime
    df['datetime'] = pd.to_datetime(df['datetime'])
    # Filter rows between 09:30:00 and 16:00:00
    filtered_df = df[(df['datetime'].dt.time >= pd.to_datetime('09:00:00').time()) &
                     (df['datetime'].dt.time <= pd.to_datetime('16:25:00').time())]
    # Print the filtered DataFrame
    filtered_df.to_csv(f'historical_data/current.csv', index=False)


def run_pipeline(data):
    # ----create training/test data (X) and training/test labels (y)----
    unscaled_data, scaled_data = normalizeData(data)
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
    # train_combined_cpy = train_combined.copy()
    # test_combined_cpy = test_combined.copy()

    print(f'train_combined shape: {train_combined.shape}')
    print(f'y_train shape: {y_train.shape}')
    print(f'test_combined shape: {test_combined.shape}')
    print(f'y_test shape: {y_test.shape}')

    # Get delta model
    delta_model = get_delta_model(retrain_delta_model, train_combined, y_train, test_combined, y_test)
    train_combined, y_train = create3dDataset(train_combined, y_train, delta_lookback)
    test_combined, y_test = create3dDataset(test_combined, y_test, delta_lookback)
    # display_test_results2(delta_model, test_combined, y_test)

    # Get delta model predictions
    predicted_deltas_train = delta_model.predict(train_combined)
    predicted_deltas_test = delta_model.predict(test_combined)

    # Combine (encoded features + X data) with predicted deltas for the trade model
    train_features = np.hstack(
        (train_combined[:, -1, :], predicted_deltas_train))
    test_features = np.hstack((test_combined[:, -1, :], predicted_deltas_test))

    # Split trading labels into train and test set
    trade_labels_train, trade_labels_test = z[:split_index], z[split_index:]
    test_features2 = test_features.copy()

    print(f'train_features shape: {train_features.shape}')
    print(f'test_features shape: {test_features.shape}')
    print(f'trade_labels_train shape: {trade_labels_train.shape}')
    print(f'trade_labels_test shape: {trade_labels_test.shape}')

    # Get trade model and test it
    trade_model = get_trade_model(retrain_trade_model, train_features, trade_labels_train, test_features,
                                  trade_labels_test)
    trade_labels_test = trade_labels_test[t_lookback:]
    print(f'trade_labels_test shape: {trade_labels_test.shape}')

    # train_features, trade_labels_train = create3dDataset(train_features, trade_labels_train, t_lookback)
    test_features, trade_labels_test = create3dDataset(test_features, trade_labels_test, t_lookback)
    createConfusionMatrices(trade_model, trade_model_name, group_name, test_features, trade_labels_test)
    plot_predictions(trade_model, unscaled_data, test_features, trade_labels_test)

    # Testing model w/ labels for half the desired delta
    t1, t2, z = create_dataset(scaled_data, window_size, unscaled_data, trade_window, desired_delta / 2)
    trade_labels_test2 = z[split_index:]
    # train_features, trade_labels_train = create3dDataset(train_features, trade_labels_train, t_lookback)
    test_features2, trade_labels_test2 = create3dDataset(test_features2, trade_labels_test2, t_lookback)
    td_name = trade_model_name + '_halvedLabels'
    createConfusionMatrices(trade_model, td_name, group_name, test_features2, trade_labels_test2)
    plot_predictions(trade_model, unscaled_data, test_features2, trade_labels_test2)


def run_pipeline_deprecated(data):
    # ----create training/test data (X) and training/test labels (y)----
    unscaled_data, scaled_data = normalizeData(data)
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
    trade_model = get_trade_model(retrain_trade_model, train_features, trade_labels_train, test_features,
                                  trade_labels_test)
    trade_labels_test = trade_labels_test[t_look_back:]
    print(f'trade_labels_test shape: {trade_labels_test.shape}')

    if t_num_of_lstm > 0:
        # train_features, trade_labels_train = create3dDataset(train_features, trade_labels_train, t_look_back)
        test_features, trade_labels_test = create3dDataset(test_features, trade_labels_test, t_look_back)
    # createConfusionMatrices(trade_model, trade_model_name, group_name, test_features, trade_labels_test, threshold)
    # plot_signals_chart(trade_model, unscaled_data, test_features, split_index, trade_labels_test)
    plot_predictions(trade_model, unscaled_data, test_features, trade_labels_test)

    # Testing model w/ labels for half the desired delta
    t1, t2, z = create_dataset(scaled_data, window_size, unscaled_data, trade_window, desired_delta / 2)
    trade_labels_test2 = z[split_index:]
    if t_num_of_lstm > 0:
        # train_features, trade_labels_train = create3dDataset(train_features, trade_labels_train, t_look_back)
        test_features2, trade_labels_test2 = create3dDataset(test_features2, trade_labels_test2, t_look_back)
    td_name = trade_model_name + '_halvedLabels'
    # createConfusionMatrices(trade_model, td_name, group_name, test_features2, trade_labels_test2, threshold)
    plot_predictions(trade_model, unscaled_data, test_features2, trade_labels_test2)


def run_live_pipeline(data):  # all retrain must be False
    # ----create training/test data (X) and training/test labels (y)----
    unscaled_data, scaled_data = normalizeData(data)
    X = create_X_dataset(scaled_data, window_size, unscaled_data, trade_window, desired_delta)
    y = None
    z = None

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

    return trade_model.predict(train_features), predicted_deltas_train, unscaled_data


def message_post(token, channel_id, message):
    # url = f"https://discord.com/api/v9/channels/{channel_id}/messages"
    url = 'https://discord.com/api/webhooks/1253402890526134295/kR3NR9FPNrPPq3Ws62Y1DVouLomTU5XCCFw82vdkubm5DG3g87LXoVYuK-8C80B-hlZq'

    headers = {
        "Authorization": f"{token}",
    }

    data = {
        "content": message
    }

    response = requests.post(url, headers=headers, json=data)

    if response.status_code < 400:
        print("Message sent successfully.")
    else:
        print(f'{response.status_code}: Failed to send the message.')


channel_id = 1112148552039288892
token = 'MTExMjE0NzAxNjg2OTQ5NDg4Ng.Gh6UbG.54FbJ_JwPTjg7dfubsEF5GZ8xxwNN3TJe_z5Dc'


"""refresh live data -> run pipeline -> print"""
live = False
while live:
    current_minute = time.localtime().tm_min

    if current_minute % 5 == 0:
        encoder_name = 'autoencoder1'  # 1=window20, 2=window40, 3=window80+batchsize12, 4=1min+w80, 5=1min+w40
        delta_model_name = 'deltaModel3'  # 2=lb3, 3=lb12+u340, 4=tradewindow6+ae2, 5=ae3, 6=ae4+tw12, 7=ae5
        trade_model_name = 'tradeModel7'
        # trade models: 3=lb3,4=lb12,5a=lb12+RSTRSF,5=RSTdoRSF+lessDenselayers, 6=moreDenseunits, 7=deltaModel3, 8=tw6+ae2+dm4,
        # 9=ae3+dm5, 10=ae4+dm6+tw12+dd.5, 11=ae4+dm6+dd.3, 12=ae5+dm7+dd.3
        up_threshold = 0.9
        down_threshold = 0.9
        desired_delta = 1

        refresh_live_data('5min')
        data = csvToList('historical_data/current.csv')
        predictions, deltas, unscaled_data = run_live_pipeline(data)
        up_predictions = predictions[:, 0]
        down_predictions = predictions[:, 1]

        scaler = MinMaxScaler()
        up_predictions = up_predictions.reshape(-1, 1)
        down_predictions = down_predictions.reshape(-1, 1)
        scaled_up_predictions = scaler.fit_transform(up_predictions)
        scaled_down_predictions = scaler.fit_transform(down_predictions)

        # Round to 0 or 1 based on the threshold
        # binary_predictions = (predictions > threshold).astype(int)
        binary_up_predictions = np.where(scaled_up_predictions >= up_threshold, 1, 0)
        binary_down_predictions = np.where(scaled_down_predictions >= down_threshold, 1, 0)

        up_pred = binary_up_predictions[-1]
        down_pred = binary_down_predictions[-1]

        message = f'--------------------------\n' \
                  f'--- 5min model ---\n' \
                  f'predicted deltas: {deltas[-1]}\n' \
                  f'unscaled predictions: {predictions[-1]}\n' \
                  f'up pred: {up_pred}\n' \
                  f'down pred: {down_pred}\n' \
                  f'current price: {unscaled_data.iloc[-1, 0]}\n' \
                  f'current time(min): {current_minute}\n' \
                  f'--------------------------\n'
        if up_pred == 1 or down_pred == 1:
            message += '@everyone\n'
        message_post(token, channel_id, message)
        print(message)

    #     time.sleep(60)

    if current_minute % 1 == 0:
        encoder_name = 'autoencoder5'  # 1=window20, 2=window40, 3=window80+batchsize12, 4=1min+w80, 5=1min+w40
        delta_model_name = 'deltaModel7'  # 2=lb3, 3=lb12+u340, 4=tradewindow6+ae2, 5=ae3, 6=ae4+tw12, 7=ae5
        trade_model_name = 'tradeModel12'
        up_threshold = 0.9
        down_threshold = 0.9
        desired_delta = 0.3

        refresh_live_data('1min')
        data = csvToList('historical_data/current.csv')
        predictions, deltas, unscaled_data = run_live_pipeline(data)
        up_predictions = predictions[:, 0]
        down_predictions = predictions[:, 1]

        scaler = MinMaxScaler()
        up_predictions = up_predictions.reshape(-1, 1)
        down_predictions = down_predictions.reshape(-1, 1)
        scaled_up_predictions = scaler.fit_transform(up_predictions)
        scaled_down_predictions = scaler.fit_transform(down_predictions)

        # Round to 0 or 1 based on the threshold
        # binary_predictions = (predictions > threshold).astype(int)
        binary_up_predictions = np.where(scaled_up_predictions >= up_threshold, 1, 0)
        binary_down_predictions = np.where(scaled_down_predictions >= down_threshold, 1, 0)

        up_pred = binary_up_predictions[-1]
        down_pred = binary_down_predictions[-1]

        message = f'--------------------------\n' \
                  f'--- 1min model ---\n' \
                  f'predicted deltas: {deltas}\n' \
                  f'unscaled predictions: {predictions[-1]}\n' \
                  f'up pred: {up_pred}\n' \
                  f'down pred: {down_pred}\n' \
                  f'current price: {unscaled_data.iloc[-1, 0]}\n' \
                  f'current time(min): {current_minute}\n' \
                  f'--------------------------\n'
        if up_pred == 1 or down_pred == 1:
            message += '@everyone\n'
        message_post(token, channel_id, message)
        print(message)

        time.sleep(60)


# retrieve historical data : [datetime,close,open,high,low,vol,obv,rsi,atr,macd]
raw_list = csvToList('historical_data/SPY5min_rawCombinedFiltered.csv')[:-trade_window-2]
# split = int(len(raw_list) / 3)
# raw_list = raw_list[2*split:]
run_pipeline(raw_list)
# print('test 1 complete ---------------------')
# split_index = int(len(raw_list) * 0.8)
# raw_list = raw_list[split_index:]
# raw_list2 = csvToList('historical_data/SPY5min_rawCombinedFiltered.csv')  # [:-trade_window]



