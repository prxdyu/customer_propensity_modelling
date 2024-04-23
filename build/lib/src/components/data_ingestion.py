# importing the libraries
import pandas as pd
import numpy as np

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.logger.logger import logging
from src.exception.exception import CustomException 



@dataclass
class DataIngestionConfig:
    # defining the paths for the data
    raw_data_path:str=os.path.join("artifacts","raw.csv")
    raw_processed_path:str=os.path.join("artifacts","raw_processed.csv")
    


class DataIngestion:

    def __init__(self):
        # creating the object of the DataIngestionConfig class
        self.ingestion_config=DataIngestionConfig()

    def initiate_data_ingestion(self):
        logging.info("Data Ingestion Started")
        try:
            data=pd.read_excel("https://github.com/prxdyu/test/raw/main/customer_data.xlsx",parse_dates=["DateTime"])

            logging.info("Reading Dataframe")

            # creating the artifacts drectory
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            # saving the raw data in the artifacts dir
            data.to_csv(self.ingestion_config.raw_data_path,index=False)

            logging.info("Saved the raw data in the artifacts folder")
            
            # filling the NaN values with 0
            data.fillna(0,inplace=True)
            # processing the duplicate actions
            data['Action']=data['Action'].apply(lambda x:"read_reviews" if x == "read_review" else x )
            data['Action']=data['Action'].apply(lambda x:"add_to_wishlist" if x == "add_to_wishist" else x )

            logging.info("Sucessfully done Basic Preporcessing (Filling NaN, eliminating duplicate categories)")

            # saving the train data
            data.to_csv(self.ingestion_config.raw_processed_path,index=False)
            logging.info("Data Ingestion Completed")

            return self.ingestion_config.raw_processed_path
            
        except Exception as e:
            logging.info("Exception occured in the data_ingestion")
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()