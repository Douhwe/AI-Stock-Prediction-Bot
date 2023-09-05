import unittest
from app import app, db, Stock
from datetime import datetime

class FlaskTestCase(unittest.TestCase):
    #python -m unittest test_app.py
    # This method is run before each test
    def setUp(self):
        app.config['TESTING'] = True
        app.config['SQLALCHEMY_DATABASE_URI'] = 'sqlite:///test_stocks.db'
        self.app = app.test_client()
        db.create_all()   # Setup a clean test database

    # This method is run after each test
    def tearDown(self):
        db.session.remove()
        db.drop_all()   # Clean up the database

    def test_get_stocks(self):
        # Add a stock for testing
        date_str = '2023-08-11'
        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        test_stock = Stock(ticker="BIGCOMPANY", date=date_obj, closing_price=100.0)
        db.session.add(test_stock)
        db.session.commit()

        response = self.app.get('/stocks', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        # Check if our test stock is in the response
        self.assertIn(b"BIGCOMPANY", response.data)

    def test_add_stock(self):
        response = self.app.post('/stocks', json={
            "ticker": "TESTTICKER",
            "date": "2023-08-12",
            "closing_price": 150.0
        })
        self.assertEqual(response.status_code, 201)
        self.assertIn(b"Stock added successfully!", response.data)

    def test_get_stock_by_ticker(self):
        # Add a stock for testing
        
        date_str = '2023-08-11'
        date_obj = datetime.strptime(date_str, '%Y-%m-%d').date()
        test_stock = Stock(ticker="SOMETICKER", date=date_obj, closing_price=80.0)
        db.session.add(test_stock)
        db.session.commit()

        response = self.app.get('/stocks/SOMETICKER', follow_redirects=True)
        self.assertEqual(response.status_code, 200)
        self.assertIn(b"SOMETICKER", response.data)

    # You can add more test cases below...

if __name__ == '__main__':
    unittest.main()
