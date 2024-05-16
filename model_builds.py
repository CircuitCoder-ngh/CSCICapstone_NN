from keras.models import Sequential
from keras.layers import Dense, LSTM, Dropout
import tensorflow as tf
from tensorflow import keras
import customCallbacks
import numpy as np
from keras.constraints import max_norm

from main import create3dDataset


def model1(training_data, training_labels):
    """Assumes input shape of 7"""
    model = Sequential()

    model.add(Dense(14, input_shape=(7,), activation='relu'))
    model.add(Dense(7, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    print("Beginning Model Compilation")
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])  #
    # fit the keras model on the dataset
    print("Beginning Model Training")
    model.fit(training_data, training_labels, epochs=150, batch_size=10)
    # evaluate the keras model
    _, accuracy = model.evaluate(training_data, training_labels)
    # print('Accuracy: %.2f' % (accuracy*100))
    model.save('models/3DenselayersPrecision150e10b.keras')
    return model


def model2(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 2)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False))

    # Add a dropout layer
    # model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=150, batch_size=10, callbacks=[])
    model.save('models/1LSTM_RSF_2Dense_150e10b.keras')

    return model


def model3(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 2)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=True))
    model.add(LSTM(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2])))

    # Add a dropout layer
    # model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=150, batch_size=10, callbacks=[])
    model.save('models/2LSTM_RST_2Dense_150e10b.keras')

    return model


def model4(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 2)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False))

    # Add a dropout layer
    model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=150, batch_size=10, callbacks=[])
    model.save('models/1LSTM_RSF_Drop_2Dense_150e10bV2.keras')

    return model


def model5(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 2)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False))

    # Add a dropout layer
    # model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=250, batch_size=10, callbacks=[])
    model.save('models/1LSTM_RSF_2Dense_250e10b.keras')

    return model


def model6(training_data, training_labels):

    train_x, train_y = create3dDataset(training_data, training_labels, 2)

    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=True))
    model.add(LSTM(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False))

    # Add a dropout layer
    model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=250, batch_size=10, callbacks=[])
    model.save('models/2LSTM_RST_Drop_2Dense_250e10b.keras')

    return model


def model7(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 2)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=True))
    model.add(LSTM(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False))

    # Add a dropout layer
    model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=250, batch_size=10, callbacks=[])
    model.save('models/2LSTM_RST_Drop_3Dense_250e10bV2.keras')

    return model


def model8(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 2)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(50, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False))

    # Add a dropout layer
    # model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=250, batch_size=10, callbacks=[])
    model.save('models/1LSTM_RSF_3Dense_250e10b.keras')

    return model


def model9(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 2)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(16, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=True))
    model.add(LSTM(16, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=True))
    model.add(LSTM(16, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False))

    # Add a dropout layer
    # model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=200, batch_size=10, callbacks=[])
    model.save('models/3LSTM_RSF_2Dense_200e10b.keras')

    return model


def model10(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 2)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(32, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=True))
    model.add(LSTM(32, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=True))
    model.add(LSTM(32, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False))

    # Add a dropout layer
    # model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=200, batch_size=10, callbacks=[])
    model.save('models/3_32LSTM_RST_2Dense_200e10b.keras')

    return model


def model11(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 2)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(16, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False))

    # Add a dropout layer
    # model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=150, batch_size=6, callbacks=[])
    model.save('models/1LSTM_RSF_3Dense_150e6b.keras')

    return model


def model12(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 1)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(32, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False, kernel_constraint=max_norm(3)))

    # Add a dropout layer
    # model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=150, batch_size=6, callbacks=[])
    model.save('models/1LSTM_KC3_RSF_2Dense_150e6b_lb1.keras')

    return model


def model13(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 1)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(32, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=True, kernel_constraint=max_norm(3)))
    model.add(LSTM(32, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False, kernel_constraint=max_norm(3)))

    # Add a dropout layer
    # model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=150, batch_size=6, callbacks=[])
    model.save('models/2LSTM_KC3_RST_2Dense_150e6b_lb1.keras')

    return model


def model14(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 1)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(32, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=True, kernel_constraint=max_norm(3)))
    model.add(LSTM(32, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False, kernel_constraint=max_norm(3)))

    # Add a dropout layer
    model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    # model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=150, batch_size=6, callbacks=[])
    model.save('models/2LSTM_KC3_RST_Drop_2Dense_150e6b_lb1.keras')

    return model


def model15(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 2)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(16, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False, kernel_constraint=max_norm(3)))

    # Add a dropout layer
    # model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(16, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=150, batch_size=3, callbacks=[])
    model.save('models/1LSTM_KC3_RSF_3_16Dense_150e3b_lb2.keras')

    return model


def model16(training_data, training_labels):
    train_x, train_y = create3dDataset(training_data, training_labels, 2)

    # performance_simple = customCallbacks.PerformancePlotCallback(train_x, train_y, '1LSTM_RSF_2Dense_150e10b')
    model = Sequential()
    # Add an LSTM layer (adjust units and other hyperparameters)
    model.add(LSTM(32, activation='relu', input_shape=(train_x.shape[1], train_x.shape[2]),
                   return_sequences=False, kernel_constraint=max_norm(3)))

    # Add a dropout layer
    # model.add(Dropout(0.2))

    # Add one or more dense layers
    model.add(Dense(32, activation='relu'))
    model.add(Dense(16, activation='relu'))
    model.add(Dense(1, activation='sigmoid'))

    # compile the keras model
    model.compile(loss='binary_crossentropy', optimizer='adam', metrics=[keras.metrics.Precision()])

    # fit the keras model on the dataset
    model.fit(train_x, train_y, epochs=150, batch_size=3, callbacks=[])
    model.save('models/1_32LSTM_KC3_RSF_32_16Dense_150e3b_lb2.keras')

    return model
