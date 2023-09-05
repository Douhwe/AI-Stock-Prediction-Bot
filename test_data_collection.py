from app import collect_data, app

if __name__ == "__main__":
    ticker = input("Enter the ticker for data collection: ").strip()
    with app.app_context():
        collect_data(ticker)
