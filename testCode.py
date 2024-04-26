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

