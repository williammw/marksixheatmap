# %%
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.optimizers import Adam
import pandas as pd
import numpy as np

# Load the data
df = pd.read_csv("MarkSix.csv")

# Extract the columns 'Winning Number 1', '2', '3', '4', '5', '6', 'Extra Number ' into a new DataFrame
winning_numbers_data = df[['Winning Number 1',
                           '2', '3', '4', '5', '6', 'Extra Number ']]

# Define the MinMaxScaler using TensorFlow


def min_max_scaler(data, feature_range=(0, 1)):
    min_val = tf.reduce_min(data)
    max_val = tf.reduce_max(data)
    scaled_data = (data - min_val) / (max_val - min_val)
    scaled_data = scaled_data * \
        (feature_range[1] - feature_range[0]) + feature_range[0]
    return scaled_data.numpy()


# Normalize the data using the TensorFlow MinMaxScaler
data_normalized = min_max_scaler(winning_numbers_data.values)

# Split data into training and testing sets (80% train, 20% test)
train_size = int(len(data_normalized) * 0.8)
test_size = len(data_normalized) - train_size
train, test = data_normalized[0:train_size,
                              :], data_normalized[train_size:len(data_normalized), :]

# Convert data to the format of [samples, time steps, features]


def create_dataset(dataset, look_back=1):
    dataX, dataY = [], []
    for i in range(len(dataset)-look_back-1):
        a = dataset[i:(i+look_back), :]
        dataX.append(a)
        dataY.append(dataset[i + look_back, :])
    return np.array(dataX), np.array(dataY)


look_back = 1
trainX, trainY = create_dataset(train, look_back)
testX, testY = create_dataset(test, look_back)

# Define the LSTM model
model = Sequential()

# Assuming 7 features ('Winning Number 1', '2', '3', '4', '5', '6', 'Extra Number ')
input_shape = (trainX.shape[1], 7)

# Add LSTM layer with 50 units and the appropriate input shape
model.add(LSTM(50, input_shape=input_shape))

# Add Dense layer to output predictions
model.add(Dense(7))

# Compile the model
model.compile(optimizer=Adam(learning_rate=0.001), loss='mean_squared_error')

# Train the model
model.fit(trainX, trainY, epochs=50, batch_size=32,
          validation_data=(testX, testY), verbose=2)

# Make predictions (as an example, on the test set)
predictions = model.predict(testX)

# If needed, inverse transform the predictions to get the actual winning numbers
predictions_actual = min_max_scaler(predictions, feature_range=(
    winning_numbers_data.min().min(), winning_numbers_data.max().max()))

# %%
predictions_actual

# %%
