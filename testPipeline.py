from mainCode3 import *

# TODO: fix this up so I can load and test models easier without editing mainCode3
print('testPipeline is running...')

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
    print(f'delta model summary: {delta_model.summary()}')
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
    trade_labels_test = trade_labels_test[delta_lookback:]
    trade_labels_train = trade_labels_train[delta_lookback:]
    trade_model = get_trade_model(retrain_trade_model, train_features, trade_labels_train, test_features,
                                  trade_labels_test)
    print(f'trade_labels_test shape: {trade_labels_test.shape}')
    print(f'trade model summary: {trade_model.summary()}')

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