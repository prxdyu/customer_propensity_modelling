import pandas as pd
import numpy as np


from src.logger.logger import logging
from src.exception.exception import CustomException 
from src.utils.utils import save_object,get_rfm_features,assign_score,get_rfm_scores

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, 'src')


@dataclass
class DataTransformationConfig:
    # defining the path for the user's RFM data
    users_rfm_data_file_path:str = os.path.join("artifacts","user_rfm_data.csv")
    # defining the path for the data with RFM features
    data_with_rfm_features_file_path:str = os.path.join("artifacts","data_with_rfm_features.csv")
    

class DataTransformation:

    def __init__(self):
        self.data_transformation_config = DataTransformationConfig()

    

    def initiate_data_transformation(self,df_path):
        logging.info("Data Transformation is initiated")
        try:
            # reading the train and test data
            df=pd.read_csv(df_path)

            logging.info("Successfully read data")

            # getting the RFM features of users
            RFM = get_rfm_features(data=df[df["Action"]=="purchase"])
            RFM = get_rfm_scores(RFM)
            logging.info("Succesfully Generated RFM Features and Scores for each user")

            RFM.to_csv(self.data_transformation_config.users_rfm_data_file_path,index=False)
            logging.info("Saved the user's RFM data")

            # getting the data with RFM features (combining RFM features with original data)
            df_rfm = pd.merge(df, RFM, on="User_id",how="left")
            logging.info("Succesfully added RFM Features for each users in data")

            # saving the data with rfm features
            df_rfm.to_csv(self.data_transformation_config.data_with_rfm_features_file_path,index=False)
            logging.info("Saved data with RFM features")


            return df_rfm

        except Exception as e:
            logging.info("Exception occured in the initiate_data_transformation")
            raise CustomException(e,sys)
    
        

if __name__=="__main__":
    obj=DataTransformation()
    obj.initiate_data_transformation()
