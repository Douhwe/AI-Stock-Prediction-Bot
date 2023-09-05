import tensorflow as tf
from keras.models import Sequential
from keras.layers import LSTM, Dense, Dropout
import joblib
from sklearn.preprocessing import MinMaxScaler




# 2.1 - Defining the Model
def create_model(input_shape):
    model = Sequential()

    # Input Layer
    model.add(LSTM(units=50, return_sequences=True, input_shape=input_shape))
    model.add(Dropout(0.2))

    # Adding second LSTM layer and dropout
    model.add(LSTM(units=50, return_sequences=True))
    model.add(Dropout(0.2))

    # Adding a third LSTM Layer and dropout
    model.add(LSTM(units=50))
    model.add(Dropout(0.2))

    # Output Layer
    model.add(Dense(units=1))

    # Compile Model
    model.compile(optimizer='adam', loss='mean_squared_error')

    return model

def train_model(x_train, y_train, x_test, y_test, ticker, epochs=100, batch_size=6, verbose=1):
    """
    Train a new LSTM model on the provided data.
    """
    model = create_model(x_train[0].shape)
    
    # Training the Model#
    history = model.fit(x_train, y_train, validation_data=(x_test, y_test), epochs=epochs, batch_size=batch_size, verbose=verbose)

    # Model Evaluation
    loss = model.evaluate(x_test, y_test, verbose=verbose)
    print(f'Validation Loss: {loss}')

    # Save the Model
    model.save(f'lstm_{ticker}.model.keras')
    
    #return model_path


def predict_next_day(model, x_test):
    """
    Predict the next day's closing price using the last sequence from x_test.
    """
    last_sequence = x_test[-1].reshape(1, -1, 1)
    predicted_price = model.predict(last_sequence)
    
    return predicted_price[0][0]

