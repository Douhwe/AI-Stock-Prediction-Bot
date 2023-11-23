from modeltraining import train_model
from datacollection import fetch_and_preprocess_data, prepare_data_for_lstm
from app import collect_data, app  # Importing the function to collect and save data #

if __name__ == "__main__":
    ticker = input("What Stock are you making a model for: ").strip()  # Strip to remove any leading/trailing whitespaces
    
    # Collect and save the latest data to the database
    with app.app_context():
        collect_data(ticker)

    
    tickerDf, scaler, originalData = fetch_and_preprocess_data(ticker)
    
    if tickerDf is None:
        print(f"Error fetching and preprocessing data for ticker: {ticker}. Exiting...")
        exit()  # Exit the script#

    x_train, y_train, x_test, y_test = prepare_data_for_lstm(tickerDf)
    train_model(x_train, y_train, x_test, y_test, ticker=ticker)
