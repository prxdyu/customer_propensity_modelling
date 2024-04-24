import pytest
from app import app # importing the app from app.py
import re



# creating a resuable client  for testing using pytest fixture
@pytest.fixture
def client():
    return app.test_client()


# creating a test function to check the ping endpoint
def test_ping(client):
    # sending a GET request to ping endpoint using the client
    response=client.get('/ping')
    # asserting the status of the request
    assert response.status=="200"+" OK"
    # asserting the content of the request
    assert response.text=="Success"


# Creating a test function to check the predict endpoint
def test_predict(client):
    # Construct a sample query point
    sample_data = {
        "category": "Camera Accessories",
        "subcategory": "Camera Lens",
        "days_active": 3,
        "R": 4,
        "F": 4,
        "M": 4,
        "loyalty": "Platinum",
        "AvgPurchaseGap": 184.0,
        "add_to_cart_to_purchase_ratios": 1,
        "add_to_wishlist_to_purchase_ratios": 0,
        "click_wishlist_page_to_purchase_ratios": 2,
        "path": "others",
        "cart_to_purchase_ratios_category": 0.3,
        "cart_to_purchase_ratios_subcategory": 1,
        "wishlist_to_purchase_ratios_category": 0,
        "wishlist_to_purchase_ratios_subcategory": 0,
        "click_wishlist_to_purchase_ratios_category": 0.3,
        "click_wishlist_to_purchase_ratios_subcategory": 0,
        "product_view_to_purchase_ratios_category": 0.6,
        "product_view_to_purchase_ratios_subcategory": 1
    }
    
    # Sending a POST request to the /predict endpoint with sample data
    response = client.post('/predict', data=sample_data)

    print("The resposne is ",response.status_code)

    # Asserting the status of the request
    assert response.status_code == 200

    # Extracting the score using regular expression
    match = re.search(r'\d+', str(response.data))
    if match:
        score = int(match.group())
        print(score)
        # checking if the score lies between 0-100
        assert 0 <= score <= 100