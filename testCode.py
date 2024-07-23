# def testDF():
#     test_list = [1,2,3,4,5,6]
#     df = pd.DataFrame(test_list)
#     df.to_csv('testDF.csv', index=False)


# def testingJSONtoCSV():
#     data_rsi = {
#         'meta': {'symbol': 'SPY', 'indicator': {'name': 'RSI - Relative Strength Index'}},
#         'values': [{'datetime': '2024-04-12 15:55:00', 'rsi': '55.43500'},
#                    {'datetime': '2024-04-12 15:50:00', 'rsi': '50.31709'}]
#     }
#
#     data_obv = {
#         'meta': {'symbol': 'SPY', 'indicator': {'name': 'OBV - On Balance Volume'}},
#         'values': [{'datetime': '2024-04-12 15:55:00', 'obv': '-53888684.00000'},
#                    {'datetime': '2024-04-12 15:50:00', 'obv': '-59221368.00000'}]
#     }
#
#     # Initialize the combined list
#     combined_list = []
#
#     # Merge 'datetime', 'obv', and 'rsi' values
#     for entry_rsi, entry_obv in zip(data_rsi['values'], data_obv['values']):
#         combined_entry = {
#             'datetime': entry_rsi['datetime'],
#             'obv': entry_obv['obv'],
#             'rsi': entry_rsi['rsi']
#         }
#         combined_list.append(combined_entry)
#
#     # Print the combined list
#     print(combined_list)
#     # Create a DataFrame from the combined list
#     df = pd.DataFrame(combined_list)
#
#     # Specify the CSV file path (adjust as needed)
#     csv_file_path = 'combined_data.csv'
#
#     # Write the DataFrame to the CSV file
#     df.to_csv(csv_file_path, index=False)
#
#     print(f"Data saved to {csv_file_path}")


# ----------
# dataset = csvToList('SPY5min14.csv')[:5]
# for item in dataset:
#     item[0] = datetime.datetime.strptime(item[0], '%Y-%m-%d %H:%M:%S').timestamp()
#
# print(pd.DataFrame(dataset))
#
# scaler = preprocessing.MinMaxScaler()
# d = scaler.fit_transform(dataset)
# scaled_df = pd.DataFrame(d)
# print(scaled_df)
# ----------
# train_x, train_y = create3dDataset(training_data, training_labels, 2)

# model = keras.models.load_model('models/1LSTM_RSF_2Dense_150e10b.keras')
# predicted_labels = []
# for i in range(1, 20):
#     predicted_labels.append(model.predict(train_x[:i]))
# model = model2()

# predicted_labels -= 1.05
# plot_labels = training_labels.copy()
# plot_labels -= 1

# Create traces
# fig = go.Figure()
# fig.add_trace(go.Scatter(x=np.arange(20), y=training_data[:20, 1],
#                          mode='lines',
#                          name='lines'))
# fig.add_trace(go.Scatter(x=np.arange(20), y=plot_labels[:20, 0],  # training_labels[:, 0]
#                          mode='lines+markers',
#                          name='lines+markers'))
# fig.add_trace(go.Scatter(x=np.arange(20), y=predicted_labels,
#                          mode='lines+markers',
#                          name='lines+markers'))
# fig.show()

# ----------------------------

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

# import datetime
#
#
# def relative_time_of_day(datetime_str):
#     # Parse the datetime string
#     dt = datetime.datetime.strptime(datetime_str, "%Y-%m-%d %H:%M:%S")
#
#     # Calculate the number of seconds since the start of the day
#     seconds_since_start_of_day = dt.hour * 3600 + dt.minute * 60 + dt.second
#
#     return seconds_since_start_of_day
#
#
# # Example usage
# datetime_str = "2024-05-22 10:00:00"
# timestamp = relative_time_of_day(datetime_str)
# print(timestamp)  # Output: 35700

# def plot_predictions(trade_model, unscaled_data, features, t_labels):
#     """
#     Plots a line chart from the data and adds markers at points where predictions are 1.
#     Aligns predictions with the end of the data if they are different lengths.
#     Also plots actual values if provided.
#
#     Parameters:
#     data (list or array-like): The data to plot.
#     predictions (list or array-like): The predictions from the Keras model with two channels (up and down).
#     actual_vals (list or array-like, optional): The actual values to plot (up and down).
#     """
#     # (data, predictions, actual_vals)
#     print(f'unscaled_data.shape == {unscaled_data.shape}')
#     unscaled_data = unscaled_data[:-trade_window]
#     predictions = trade_model.predict(features)
#     dif = unscaled_data.shape[0] - features.shape[0]
#     data = unscaled_data.iloc[dif:, 0].reset_index(drop=True)
#
#     plt.figure(figsize=(10, 6))
#
#     # Plot the data as a line chart
#     plt.plot(data, label='Data', color='blue')
#
#     print(f'unscaled data shape == {unscaled_data.shape}')
#     print(f'features shape == {features.shape}')
#     print(f'dif == {dif}')
#     print(f'data shape == {data.shape}')
#     print(f'predictions shape == {predictions.shape}')
#
#     up_labels = predictions[:, 0]
#     down_labels = predictions[:, 1]
#     scaler = MinMaxScaler()
#     up_predictions = up_labels.reshape(-1, 1)
#     down_predictions = down_labels.reshape(-1, 1)
#     scaled_up_predictions = scaler.fit_transform(up_predictions)
#     scaled_down_predictions = scaler.fit_transform(down_predictions)
#
#     # Round to 0 or 1 based on the threshold
#     binary_up_predictions = np.where(scaled_up_predictions >= up_threshold, 1, 0)
#     binary_down_predictions = np.where(scaled_down_predictions >= down_threshold, 1, 0)
#     combined_predictions = np.column_stack((binary_up_predictions, binary_down_predictions))
#
#     print(f'combined preds shape == {combined_predictions.shape}')
#     print(f't_labels shape == {t_labels.shape}')
#
#     # Calculate the starting index to align predictions with the end of the data
#     start_index = len(data) - len(t_labels)
#     print(f'start index == {start_index}')
#     # Plot the actual values if provided
#     if t_labels is not None:
#         for i, (up_pred, down_pred) in enumerate(t_labels):
#             if up_pred == 1:
#                 x1 = start_index + i
#                 y1 = data[start_index + i]
#                 # plt.plot(x1, y1, color='black', marker='o')
#                 x_vals = [x1, x1 + trade_window]
#                 y_vals = [y1, y1 + desired_delta]
#                 plt.plot(x_vals, y_vals, color="green")
#             if down_pred == 1:
#                 x1 = start_index + i
#                 y1 = data[start_index + i]
#                 # plt.plot(x1, y1, color='black', marker='o')
#                 x_vals = [x1, x1 + trade_window]
#                 y_vals = [y1, y1 - desired_delta]
#                 plt.plot(x_vals, y_vals, color="red")
#
#     # Calculate the starting index to align predictions with the end of the data
#     start_index = len(data) - len(predictions)
#     wu = 0
#     lu = 0
#     wd = 0
#     ld = 0
#     # Add markers for up and down predictions
#     for i, (up_pred, down_pred) in enumerate(combined_predictions):
#         if up_pred == 1:
#             # TODO: figure out why '-12' made predictions line up w/ actual values
#             if t_labels[i - 12, 0] == 1:
#                 plt.plot(start_index + i, data[start_index + i], color='green', marker='o')
#                 wu += 1
#             else:
#                 plt.plot(start_index + i, data[start_index + i], color='k', marker='o')
#                 lu += 1
#         if down_pred == 1:
#             if t_labels[i - 12, 1] == 1:
#                 wd += 1
#                 plt.plot(start_index + i, data[start_index + i], color='red', marker='o')
#             else:
#                 ld += 1
#                 plt.plot(start_index + i, data[start_index + i], color='k', marker='o')
#
#     print(f'wu == {wu}, lu == {lu}')
#     print(f'wd == {wd}, ld == {ld}')
#     acc = (wu + wd) / (wu + wd + lu + ld)
#     print(f'accuracy == {acc}')
#
#     plt.xlabel('Index')
#     plt.ylabel('Value')
#     plt.title('Line Chart with Predictions')
#     plt.legend()
#     plt.show()
#
# def plot_signals_chart(trade_model, unscaled_data, features, split_index, t_labels):
#     # Initialize the plot and closing prices
#     plt.figure(figsize=(10, 6))
#     print(f'features shape == {features.shape}')
#     print(f'unscaled data shape == {unscaled_data.shape}')
#     labels = trade_model.predict(features)
#     dif = len(unscaled_data) - len(labels)
#     closing_prices = unscaled_data.iloc[dif:, 0].reset_index(drop=True)
#     print(f'labels shape == {labels.shape}')
#     print(f't_labels shape == {t_labels.shape}')
#     print(f'closing_prices shape == {closing_prices.shape}')
#
#     # Get indices where labels are 1
#     up_labels = labels[:, 0]
#     down_labels = labels[:, 1]
#     up_marker_indices = []
#     down_marker_indices = []
#     for i in range(len(up_labels)):
#         if up_labels[i] == 1:
#             up_marker_indices.append(i)
#     for i in range(len(down_labels)):
#         if down_labels[i] == 1:
#             down_marker_indices.append(i)
#
#     scaler = MinMaxScaler()
#     up_predictions = up_labels.reshape(-1, 1)
#     down_predictions = down_labels.reshape(-1, 1)
#
#     scaled_up_predictions = scaler.fit_transform(up_predictions)
#     # scaled_up_predictions = np.array(scaled_up_predictions)
#
#     scaled_down_predictions = scaler.fit_transform(down_predictions)
#
#     # Round to 0 or 1 based on the threshold
#     # binary_predictions = (predictions > threshold).astype(int)
#     binary_up_predictions = np.where(scaled_up_predictions >= up_threshold, 1, 0)
#     binary_down_predictions = np.where(scaled_down_predictions >= down_threshold, 1, 0)
#     up_marker_indices = np.where(binary_up_predictions == 1)[0]
#     down_marker_indices = np.where(binary_down_predictions == 1)[0]
#     tup_marker_indices = np.where(t_labels[:, 0] == 1)[0]
#     tdown_marker_indices = np.where(t_labels[:, 1] == 1)[0]
#     # print(f'up_marker_indices: {up_marker_indices}')
#     # print(f'down_marker_indices: {down_marker_indices}')
#     # print(f'closing_prices.iloc[up_marker_indices: {closing_prices.iloc[up_marker_indices]}')
#     # print(f'closing_prices.iloc[down_marker_indices: {closing_prices.iloc[down_marker_indices]}')
#     # print(f'len closing_prices: {len(closing_prices)}')
#     # print(f'len binup predictions: {len(binary_up_predictions)}')
#     # print(f'len features: {len(features)}')
#
#     # Plot markers on the same graph
#     # plt.plot(closing_prices, label='Closing Price')
#     # plt.scatter(tup_marker_indices, closing_prices.iloc[tup_marker_indices],  # '+ dif'
#     #             color='green', label='Long', marker='o')
#     # plt.scatter(tdown_marker_indices, closing_prices.iloc[tdown_marker_indices],
#     #             color='red', label='Short', marker='o')
#     #
#     # # Add lines based on the conditions
#     # for idx in tup_marker_indices:
#     #     x_values = [idx, idx + trade_window]
#     #     y_values = [closing_prices.iloc[idx], closing_prices.iloc[idx] + desired_delta]
#     #     # if t_labels[idx, 0] == 1:
#     #     if up_labels[idx] == 1:
#     #         plt.plot(x_values, y_values, color='green')
#     #     else:
#     #         plt.plot(x_values, y_values, color='black')
#     #
#     # for idx in tdown_marker_indices:
#     #     x_values = [idx, idx + trade_window]
#     #     y_values = [closing_prices.iloc[idx], closing_prices.iloc[idx] - desired_delta]
#     #     # if t_labels[idx, 1] == 1:
#     #     if down_labels[idx] == 1:
#     #         plt.plot(x_values, y_values, color='red')
#     #     else:
#     #         plt.plot(x_values, y_values, color='black')
#     #
#     # wu = 0
#     # lu = 0
#     # wd = 0
#     # ld = 0
#     # # up_labels
#     # print(f'up labels shape == {up_labels.shape}')
#     # print(f'up labels len == {len(up_labels)}')
#     # for i in range(len(up_labels)):
#     #     if up_labels[i] == 1:
#     #         if up_labels[i] == t_labels[i, 0]:
#     #             wu += 1
#     #         else:
#     #             lu += 1
#     # for i in range(len(down_labels)):
#     #     if down_labels[i] == 1:
#     #         if down_labels[i] == t_labels[i, 1]:
#     #             wd += 1
#     #         else:
#     #             ld += 1
#     #
#     # print(f'wu == {wu}, lu == {lu}')
#     # print(f'wd == {wd}, ld == {ld}')
#     # acc = (wu + wd) / (wu + wd + lu + ld)
#     # print(f'accuracy == {acc}')
#     #
#     # # Add labels and title
#     # plt.title('Closing Prices with Markers')
#     # plt.xlabel('Time')
#     # plt.ylabel('Closing Price')
#     # plt.legend()
#     # plt.show()
