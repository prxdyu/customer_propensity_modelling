import os
import sys
import pandas as pd

from src.exception.exception import CustomException
from src.logger.logger import logging
from src.utils.utils import load_object


class PredictionPipeline:

    def __init__(self):
        pass

    def predict(self,features):
        try:
            # loading the preprocesssor object
            preprocessor_path = os.path.join("artifacts","preprocessor.pkl")
            preprocessor = load_object(preprocessor_path)

            # loading the model
            model_path = os.path.join("artifacts","model.pkl")
            model = load_object(model_path)

            # pre-processing the features
            scaled_features = preprocessor.transform(features)
            # prediction
            prediction = model.predict_proba(scaled_features)[:,1]

            return prediction


        except Exception as e:
            raise CustomException(e,sys)
        

class CustomData:

   def __init__( self,
                 Category:str,
                 SubCategory:str,
                 days_active:int,
                 R:int,
                 F:int,
                 M:int,
                 Loyalty:str,
                 AvgPurchaseGap:int,
                 add_to_cart_to_purchase_ratios:float,
                 add_to_wishlist_to_purchase_ratios:float,
                 click_wishlist_page_to_purchase_ratios:float,
                 user_path:str,
                 cart_to_purchase_ratios_category:float,
                 cart_to_purchase_ratios_subcategory:float,
                 wishlist_to_purchase_ratios_category:float,
                 wishlist_to_purchase_ratios_subcategory:float,
                 click_wishlist_to_purchase_ratios_category:float,
                 click_wishlist_to_purchase_ratios_subcategory:float,
                 product_view_to_purchase_ratios_category:float,
                 product_view_to_purchase_ratios_subcategory:float
                 ):
        self.category = Category
        self.subcategory = SubCategory
        self.days_active = days_active
        self.R=R
        self.F=F
        self.M=M
        self.loyalty=Loyalty
        self.avg_purchase_gap=AvgPurchaseGap
        self.add_to_cart_to_purchase_ratios=add_to_cart_to_purchase_ratios
        self.add_to_wishlist_to_purchase_ratios=add_to_wishlist_to_purchase_ratios
        self.click_wishlist_page_to_purchase_ratios=click_wishlist_page_to_purchase_ratios
        self.user_path=user_path
        self.cart_to_purchase_ratios_category=cart_to_purchase_ratios_category
        self.cart_to_purchase_ratios_subcategory=cart_to_purchase_ratios_subcategory
        self.wishlist_to_purchase_ratios_category=wishlist_to_purchase_ratios_category
        self.wishlist_to_purchase_ratios_subcategory=wishlist_to_purchase_ratios_subcategory
        self.click_wishlist_to_purchase_ratios_category=click_wishlist_to_purchase_ratios_category
        self.click_wishlist_to_purchase_ratios_subcategory=click_wishlist_to_purchase_ratios_subcategory
        self.product_view_to_purchase_ratios_category=product_view_to_purchase_ratios_category
        self.product_view_to_purchase_ratios_subcategory=product_view_to_purchase_ratios_subcategory



   def get_data_as_df(self):
        try:
            custom_data_input_dict = {
            'Category': [self.category],
            'SubCategory': [self.subcategory],
            'days_active': [self.days_active],
            'R': [self.R],
            'F': [self.F],
            'M': [self.M],
            'Loyalty': [self.loyalty],
            'Avg Purchase Gap': [self.avg_purchase_gap],
            'add_to_cart_to_purchase_ratios': [self.add_to_cart_to_purchase_ratios],
            'add_to_wishlist_to_purchase_ratios':[self.add_to_wishlist_to_purchase_ratios],
            'click_wishlist_page_to_purchase_ratios': [self.click_wishlist_page_to_purchase_ratios],
            'user_path': [self.user_path],
            'cart_to_purchase_ratios_category': [self.cart_to_purchase_ratios_category],
            'cart_to_purchase_ratios_subcategory': [self.cart_to_purchase_ratios_subcategory],
            'wishlist_to_purchase_ratios_category': [self.wishlist_to_purchase_ratios_category],
            'wishlist_to_purchase_ratios_subcategory': [self.wishlist_to_purchase_ratios_subcategory],
            'click_wishlist_to_purchase_ratios_category': [self.click_wishlist_to_purchase_ratios_category],
            'click_wishlist_to_purchase_ratios_subcategory': [self.click_wishlist_to_purchase_ratios_subcategory],
            'product_view_to_purchase_ratios_category': [self.product_view_to_purchase_ratios_category],
            'product_view_to_purchase_ratios_subcategory': [self.product_view_to_purchase_ratios_subcategory],
            }
            df = pd.DataFrame(custom_data_input_dict)
            logging.info('Dataframe Gathered')
            logging.info(f"The cols are{df.columns}")
            return df
        except Exception as e:
            logging.info('Exception Occured in prediction pipeline')
            raise CustomException(e,sys)