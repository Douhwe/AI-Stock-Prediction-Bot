import tensorflow as tf
from tensorflow import keras

from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout

from datacollection import xTrain, yTrain, xTest, yTest


#2.1 - Defining the Model 


def createModel(inputShape):
    model = Sequential()

    #Input Layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=inputShape))
    model.add(Dropout(0.2))

    #Adding second LSTM layer and dropout
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    #Adding a third LSTM Layer and dropout
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    #Output Layer
    model.add(Dense(units=1))

    #CompileModel
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model
#2.3 Model Instantiation
model = createModel(xTrain[0].shape)

#2.4 Data Inspection
print("xTrain:", type(xTrain), xTrain.shape)
print("yTrain:", type(yTrain), yTrain.shape)
print("xTest:", type(xTest), xTest.shape)
print("yTest:", type(yTest), yTest.shape)


#2.5 Training the Model
history = model.fit(xTrain, yTrain, validation_data=(xTest, yTest), epochs=100, batch_size=6, verbose=1)

#2.6 Model Evaluation
loss = model.evaluate(xTest, yTest, verbose=1)
print(f'Validation Loss: {loss}')

# 2.7 Predicting the Next Day's Closing Price
last_sequence = xTest[-1].reshape(1, -1, 1)
predicted_price = model.predict(last_sequence)
print(f"Predicted next day's closing price: {predicted_price[0][0]}")

# 2.8 Save the Model
#model.save("lstm_model.h5")
model.save("lstm.model.keras")
