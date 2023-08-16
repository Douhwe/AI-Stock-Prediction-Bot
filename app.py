from flask import Flask, jsonify, request
from flask_sqlalchemy import SQLAlchemy
from datetime import datetime
import pandas as pd
from datacollection import fetch_and_preprocess_data
from modeltraining import train_model
from modeltraining import predict_next_day


#3.1 Flask Configuration
app = Flask(__name__)

#3.2 SQLite Configuration
app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///stocks.db'
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
db = SQLAlchemy(app)


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
    new_stock = Stock(ticker=data['ticker'], 
                      date=datetime.strptime(data['date'], '%Y-%m-%d'),  # Convert string to date
                      closing_price=data['closing_price'])
    db.session.add(new_stock)
    db.session.commit()
    return jsonify({"message": "Stock added successfully!"}), 201


#3.6 Fetch Stock by Ticker
@app.route('/stocks/<string:ticker>', methods=['GET'])
def get_stock_by_ticker(ticker):
    stocks = Stock.query.filter_by(ticker=ticker).all()
    if not stocks:
        return jsonify({"message": "Stock not found"}), 404

    response = []
    for stock in stocks:
        response.append({
            "id": stock.id,
            "ticker": stock.ticker,
            "date": stock.date.strftime('%Y-%m-%d'),
            "closing_price": stock.closing_price
        })

    return jsonify(response)

#3.7 Collect and Preprocess Data
@app.route('/collect_data/<string:ticker>', methods=['POST'])
def collect_data(ticker):
    # Use the ticker from the route to fetch and preprocess data
    data = fetch_and_preprocess_data(ticker)
    
    # Insert data into the SQLite database
    for index, row in data.iterrows():
        exists = Stock.query.filter_by(ticker=ticker, date=pd.to_datetime(row['Date'])).first()
        if not exists:
            stock = Stock(ticker=ticker, date=pd.to_datetime(row['Date']), closing_price=row['Close'])
            db.session.add(stock)
    db.session.commit()

    return jsonify({"message": f"Data for {ticker} collected and saved successfully!"})

#3.8 Training Model
@app.route('/train_model/<string:ticker>', methods=['POST'])
def train(ticker):
    stocks = Stock.query.filter_by(ticker=ticker).all()
    if not stocks:
        return jsonify({"message": f"No data found for {ticker}."}), 404

    # Convert stock data to DataFrame for training
    df = pd.DataFrame([(stock.date, stock.closing_price) for stock in stocks], columns=["Date", "Close"])

    # Train model
    model_path = train_model(df)
    
    return jsonify({"message": f"Model trained for {ticker} and saved at {model_path}."})

#3.9 Model Prediction
@app.route('/predict/<string:ticker>', methods=['GET'])
def predict(ticker):
    stocks = Stock.query.filter_by(ticker=ticker).all()
    if not stocks:
        return jsonify({"message": f"No data found for {ticker}."}), 404

    # Convert stock data to DataFrame for prediction
    df = pd.DataFrame([(stock.date, stock.closing_price) for stock in stocks], columns=["Date", "Close"])

    # Get prediction
    prediction = predict_next_day(df)
    
    return jsonify({"predicted_closing_price": prediction})



#3.10 Running the Flask App
#db.create_all()

#Viewable at http://127.0.0.1:5000/stocks to view/add stock data

if __name__ == '__main__':
    app.run(debug=True)
