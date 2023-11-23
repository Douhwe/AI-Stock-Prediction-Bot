from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
from datacollection import fetch_and_preprocess_data, prepare_data_for_lstm
from modeltraining import train_model
from modeltraining import predict_next_day
from joblib import dump, load
import tensorflow as tf
from flask_cors import CORS



#3.1 Flask Configuration
app = Flask(__name__)

#3.2 SQLite Configuration
#app.config['TESTING'] = True
#app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_stocks.db'

app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)
CORS(app)

#3.3 Model for Stocks Data
class Stock(db.Model):
    id = db.Column(db.Integer, primary_key=True, autoincrement=True)
    ticker = db.Column(db.String, nullable=False)
    date = db.Column(db.Date, nullable=False)
    closing_price = db.Column(db.Float, nullable=False)


#3.4 Fetch All Stock
@app.route('/stocks', methods=['GET'])
def get_stocks():
    stocks = Stock.query.all()
    response = []
    for stock in stocks:
        response.append({
            "id": stock.id,
            "ticker": stock.ticker,
            "date": stock.date.strftime('%Y-%m-%d'),  # Format date as string
            "closing_price": stock.closing_price
        })
    return jsonify(response)

#3.5 Add Stock
@app.route('/stocks', methods=['POST'])
def add_stock():
    data = request.json
    try:
        date_obj = datetime.strptime(data['date'], '%Y-%m-%d')  # Convert string to date
    except ValueError:
        return jsonify({"message": "Invalid date format. Expected format: YYYY-MM-DD"}), 400

    new_stock = Stock(ticker=data['ticker'], date=date_obj, closing_price=data['closing_price'])
    db.session.add(new_stock)
    db.session.commit()
    return jsonify({"message": "Stock added successfully!"}), 201



#3.6 Fetch Stock by Ticker
@app.route('/stocks/<string:ticker>', methods=['GET'])
def get_stock_by_ticker(ticker):
    print(f"Entered get_stock_by_ticker with ticker: {ticker}")

    # Fetching data from the database
    stocks = Stock.query.filter_by(ticker=ticker).all()
    if not stocks:
        return jsonify({"message": "Stock not found"}), 404

    db_response = []
    for stock in stocks:
        db_response.append({
            "id": stock.id,
            "ticker": stock.ticker,
            "date": stock.date.strftime('%Y-%m-%d'),
            "closing_price": stock.closing_price
        })

    # Fetching the original, unscaled data
    try:
        _, _, original_data = fetch_and_preprocess_data(ticker)
        original_data_json = original_data.to_dict(orient='records')
    except Exception as e:
        print(f"Error fetching original data: {str(e)}")
        return jsonify({"message": f"Error fetching original data for {ticker}"}), 500

    # Combine both responses
    combined_response = {
        "databaseData": db_response,
        "originalData": original_data_json
    }

    return jsonify(combined_response)

#3.7 Collect and Preprocess Data
@app.route('/collect_data/<string:ticker>', methods=['POST'])
def collect_data(ticker):
    print("Entered collect_data function")
    try:
        # Unpack the values returned from fetch_and_preprocess_data function
        data, scaler, _ = fetch_and_preprocess_data(ticker)

        if data is None:
            print("Failed to fetch and preprocess data.")
            return jsonify({"message": "Failed to fetch and preprocess data."}), 500

        print(f"Fetched and preprocessed {len(data)} records.")

        for record in data.itertuples():
            existing_stock = Stock.query.filter_by(ticker=ticker, date=record.Date).first()
            if existing_stock:
                print(f"Updating record for date: {record.Date}")
                # Update the existing record if needed
                existing_stock.closing_price = record.Close
            else:
                print(f"Adding new record for date: {record.Date}")
                stock = Stock(ticker=ticker, date=record.Date, closing_price=record.Close)
                db.session.add(stock)

        db.session.commit()
        print("Data committed to the database.")

        scaler_filename = f"scalers/{ticker}_scaler.save"
        dump(scaler, scaler_filename)
        print("Scaler saved to disk.")

        return jsonify({"message": f"Data for {ticker} collected and updated successfully!"})
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        return jsonify({"message": f"An error occurred: {str(e)}"}), 500

# @app.route('/collect_data/<string:ticker>', methods=['POST'])
# def collect_data(ticker):
#     try:
#         data_and_scaler = fetch_and_preprocess_data(ticker)
#         if data_and_scaler is None:
#             return jsonify({"message": "Failed to fetch and preprocess data."}), 500
#         data, scaler = data_and_scaler  # Getting both data and scaler #

#         for record in data.itertuples():
#             # Check if record already exists in the database
#             existing_stock = Stock.query.filter_by(ticker=ticker, date=record.Index).first()

#             if existing_stock:
#                 # Update the existing record if needed
#                 existing_stock.close = record.Close
#             else:
#                 # Add a new record if it doesn't exist
#                 stock = Stock(ticker=ticker, date=record.Index, close=record.Close)
#                 db.session.add(stock)

#         db.session.commit()

#         # Save the scaler to disk#
#         scaler_filename = f"scalers/{ticker}_scaler.save"
#         dump(scaler, scaler_filename)

#         return jsonify({"message": f"Data for {ticker} collected and updated successfully!"})
#     except Exception as e:
#         return jsonify({"message": f"An error occurred: {str(e)}"}), 500

#3.8 Training Model
@app.route('/train_model/<string:ticker>', methods=['POST'])
def train(ticker):
    try:
        # Get stock data from your database
        stocks = Stock.query.filter_by(ticker=ticker).all()
        if not stocks:
            return jsonify({"message": f"No data found for {ticker}."}), 404

        # Convert stock data to DataFrame
        df = pd.DataFrame([(stock.date, stock.closing_price) for stock in stocks], columns=["Date", "Close"])

        # Integrate fetch_and_preprocess_data to get the scaled 'Close' column
        # This assumes 'fetch_and_preprocess_data' will overwrite 'Close' column with scaled data.
        # Alternatively, if you already have sufficient data in your database, you can skip this step
        df, scaler = fetch_and_preprocess_data(ticker)

        # Prepare data for LSTM
        xTrain, yTrain, xTest, yTest = prepare_data_for_lstm(df)

        # Train model
        model_path = train_model(xTrain, yTrain, xTest, yTest, epochs=100, batch_size=32)
    
        return jsonify({"message": f"Model trained for {ticker} and saved at {model_path}."})

    except Exception as e:
        return jsonify({"error": f"Error during model training: {str(e)}"})
    
#3.9 Model Prediction
@app.route('/predict/<string:ticker>', methods=['GET'])
def predict(ticker):
    stocks = Stock.query.filter_by(ticker=ticker).all()
    
    # Convert retrieved stocks into a DataFrame
    df = pd.DataFrame([{'Date': s.date, 'Close': s.closing_price} for s in stocks])

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    # Load the scaler
    scaler_filename = f"scalers/{ticker}_scaler.save"
    scaler = load(scaler_filename)

    # Prepare data for LSTM
    x_test, _, _, _ = prepare_data_for_lstm(df)  # Only interested in x_test

    # Load the trained LSTM model
    model = tf.keras.models.load_model(f'lstm_{ticker}.model.keras')

    # Make prediction and inverse transform
    scaled_prediction = predict_next_day(model, x_test)
    prediction = scaler.inverse_transform([[scaled_prediction]])[0][0]

    return jsonify({"predicted_closing_price": prediction})

@app.route('/predict_with_query_param', methods=['GET'])
def predict_with_query_param():
    ticker = request.args.get('ticker')
    # Removed date fetching since it's not used in the prediction

    # Check if ticker is provided
    if not ticker:
        return jsonify({"message": "Ticker is required as a query parameter."}), 400

    stocks = Stock.query.filter_by(ticker=ticker).all()
    
    # Convert retrieved stocks into a DataFrame
    df = pd.DataFrame([{'Date': s.date, 'Close': s.closing_price} for s in stocks])

    # Set 'Date' as the index
    df.set_index('Date', inplace=True)

    # Load the scaler
    scaler_filename = f"scalers/{ticker}_scaler.save"
    scaler = load(scaler_filename)

    # Prepare data for LSTM
    x_test, _, _, _ = prepare_data_for_lstm(df)  # Only interested in x_test

    # Load the trained LSTM model
    model = tf.keras.models.load_model(f'lstm_{ticker}.model.keras')

    # Make prediction and inverse transform
    scaled_prediction = predict_next_day(model, x_test)
    prediction = scaler.inverse_transform([[scaled_prediction]])[0][0]

    return jsonify({"predicted_closing_price": prediction})


#3.10 Running the Flask App
db.create_all()

#Viewable at http://127.0.0.1:5000/stocks to view/add stock data

if __name__ == '__main__':
    app.run(debug=True)
