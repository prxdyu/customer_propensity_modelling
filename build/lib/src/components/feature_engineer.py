# importing the libraries
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

import os
import sys
from dataclasses import dataclass
from pathlib import Path

from src.logger.logger import logging
from src.exception.exception import CustomException 



@dataclass
class FeatureEngineerConfig:
    pass

class DataIngestion:

    def __init__(self):
        # creating the object of the DataIngestionConfig class
        self.feature_engineer_config=FeatureEngineerConfig()

    def initiate_feature_engineering(self,train):
        logging.info("Feature Engineering  Started")
        try:
            data=pd.read_csv("https://github.com/prxdyu/test/raw/main/customer_data.xlsx")
            logging.info("Reading Dataframe")
            # creating the artifacts drectory
            os.makedirs(os.path.dirname(os.path.join(self.ingestion_config.raw_data_path)),exist_ok=True)
            # saving the raw data in the artifacts dir
            data.to_csv(self.ingestion_config.raw_data_path,index=False)
            logging.info("Saved the raw data in the artifacts folder")

            # train test split
            train_data,test_data = train_test_split(data,test_size=0.25)
            logging.info("Train Test Split completed")

            # saving the train data
            train_data.to_csv(self.ingestion_config.train_data_path,index=False)
            test_data.to_csv(self.ingestion_config.test_data_path,index=False)
            logging.info("Data Ingestion Completed")

            return (self.ingestion_config.train_data_path,self.ingestion_config.test_data_path)
            
        except Exception as e:
            logging.info("Exception occured in the data_ingestion")
            raise CustomException(e,sys)
        

if __name__=="__main__":
    obj=DataIngestion()
    obj.initiate_data_ingestion()