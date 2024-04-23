import pandas as pd
import numpy as np


from src.logger.logger import logging
from src.exception.exception import CustomException 
from src.utils.utils import get_dataset,add_date_level_features,get_user_level_features,add_user_level_features,add_category_subcategory_level_features,get_category_subcategory_level_features

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, 'src')


@dataclass
class FeatureEngineeringConfig:
    # defining the path for the data with engineered features
    modelling_data_path:str = os.path.join("artifacts","modelling_data.csv")
    

class FeatureEngineering:

    def __init__(self):
        self.feature_engineering_config = FeatureEngineeringConfig()

    


    def initiate_feature_engineering(self,df_path):
        logging.info("Feature Engineering is initiated")
        try:
            # reading the data with RFM features
            df = pd.read_csv(df_path,parse_dates=["DateTime"])
            logging.info("Successfully read data with RFM features")

            """ Adding date level features """
            df =  add_date_level_features(df)
            logging.info("Added date level features")

            # getting the dataset for the training
            df_base = get_dataset(df)
            logging.info("Successfully filtered dataset for training")

            

            """ Adding user level features"""
            # getting user-level features such as No of Active Days, Cart-purchase ratio, Avg time between purchase
            days_active, avg_purchase_time, cart_to_purchase, wishlist_to_purchase, wishlist_click_to_purchase, paths = get_user_level_features(df)
            # adding user-level features
            df_base = add_user_level_features(df_base,df,days_active, avg_purchase_time, cart_to_purchase, wishlist_to_purchase, wishlist_click_to_purchase, paths)
            logging.info("Added user level features")


            """ Adding category and subcategory level features """
            # getitng the category level features
            cart_to_purchase_ratio_category, cart_to_purchase_ratio_subcategory, wishlist_to_purchase_ratio_category, wishlist_to_purchase_ratio_subcategory, click_wishlist_to_purchase_ratio_category, click_wishlist_to_purchase_ratio_subcategory, view_to_purchase_ratio_category, view_to_purchase_ratio_subcategory = get_category_subcategory_level_features(df)
            # adding category level features
            df_base = add_category_subcategory_level_features(df_base, cart_to_purchase_ratio_category, cart_to_purchase_ratio_subcategory, wishlist_to_purchase_ratio_category, wishlist_to_purchase_ratio_subcategory, click_wishlist_to_purchase_ratio_category, click_wishlist_to_purchase_ratio_subcategory, view_to_purchase_ratio_category, view_to_purchase_ratio_subcategory)


            # saving the data with engineered features for modelling
            df_base.to_csv(self.feature_engineering_config.modelling_data_path,index=False)
            logging.info("Successfully completed Feature Engineering and saved the data for modelling")


            return self.feature_engineering_config.modelling_data_path

        except Exception as e:
            logging.info("Exception occured in the initiate_data_transformation")
            raise CustomException(e,sys)
    
        

if __name__=="__main__":
    obj=FeatureEngineering()
    obj.initiate_feature_engineering("artifacts/raw_with_rfm_features.csv")
