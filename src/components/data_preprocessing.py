import pandas as pd
import numpy as np

import sklearn
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.pipeline import Pipeline,make_pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from category_encoders import TargetEncoder
from sklearn.preprocessing import OneHotEncoder
from sklearn.metrics import f1_score

from src.logger.logger import logging
from src.exception.exception import CustomException 
from src.utils.utils import save_object

import os
import sys
from dataclasses import dataclass
from pathlib import Path

import sys
sys.path.insert(0, 'src')


@dataclass
class DataPreProcessingConfig:
    # definfing the path for the preprocessor object
    preprocessor_path:str = os.path.join("artifacts","preprocessor.pkl")
    # defining the path for the training data
    train_path:str = os.path.join("artifacts","train.csv")
    # defining the path for the testing data
    test_path:str = os.path.join("artifacts","test.csv")



    

class DataPreProcessing:

    def __init__(self):
        self.data_preprocessing_config = DataPreProcessingConfig()

    def get_data_preprocessor(self,x_train):

        try:
             # splitting the categorical columns and numerical columns
            cat_cols = x_train.select_dtypes(["object"]).columns.tolist()
            cat_df = x_train[cat_cols]

            num_cols = x_train.select_dtypes(["int","float"]).columns.tolist()
            num_df = x_train[num_cols]

            """ IMPUTATION """
            # defining a lists of cols to impute with -1 
            cols_to_impute_with_minus_1 = ["Avg Purchase Gap"]
            # defining a lists of cols to impute with 0
            cols_to_impute_with_0 = ['days_active', 'R', 'F', 'M', 'add_to_cart_to_purchase_ratios', 'add_to_wishlist_to_purchase_ratios', 'click_wishlist_page_to_purchase_ratios', 'cart_to_purchase_ratios_category', 'cart_to_purchase_ratios_subcategory', 'wishlist_to_purchase_ratios_category', 'wishlist_to_purchase_ratios_subcategory', 'click_wishlist_to_purchase_ratios_category', 'click_wishlist_to_purchase_ratios_subcategory', 'product_view_to_purchase_ratios_category', 'product_view_to_purchase_ratios_subcategory'] 
            # definfing list of categorical cols to do OHE
            ohe_cols = ["Loyalty","user_path"]
            # definfing list of categorical cols to do Target encoding
            target_encod_cols = ['Category', 'SubCategory']


            # defining the imputer obj
            imputer = ColumnTransformer([
                                        ('imputer_0', SimpleImputer(strategy='constant', fill_value=0), cols_to_impute_with_0),
                                        ('imputer_minus_1', SimpleImputer(strategy='constant', fill_value=-1), cols_to_impute_with_minus_1),
                                        ('imputer_categorical', SimpleImputer(strategy='most_frequent'), cat_cols),
                                        ])
            imputer.set_output(transform="pandas")


            """ ENCODING """
            # getting the name of the first operation of imputer (ColumnTransformer) coz this name will be added as a prefix for all columns of the resulting df by imputer object
            numerical_prefix = imputer.transformers[0][0]+"__"
            purchase_prefix =  imputer.transformers[1][0]+"__"
            categorical_prefix = imputer.transformers[2][0]+"__"

            # prefixing the cols 
            prefixed_ohe_cols = [categorical_prefix + col for col in ohe_cols ]
            prefixed_target_encod_cols = [categorical_prefix + col for col in target_encod_cols]

            prefixed_num_cols = [numerical_prefix + col for col in num_cols ]
            # replacing the imputer_categorical__days_active imputer_minus_1__days_active
            prefixed_num_cols = [f"{purchase_prefix}Avg Purchase Gap" if i == f"{numerical_prefix}Avg Purchase Gap" else i for i in prefixed_num_cols]

            # definfing the encoder for categorical and numerical cols
            encoder = ColumnTransformer([('scaler_numeric', StandardScaler(), prefixed_num_cols),
                                        ('ohe_encoder', OneHotEncoder(sparse=False), prefixed_ohe_cols),
                                        ('target_encoders', TargetEncoder(), prefixed_target_encod_cols)
                                        ])
            encoder.set_output(transform="pandas")


            """ PIPELINES """
            # creating a pipeline which consists of imputer followed by encoder
            preprocessor = Pipeline([ ('imputer', imputer),
                                    ('encoder',encoder)
                                    ])
            preprocessor.set_output(transform="pandas")

            return preprocessor

        except Exception as e :
            logging.info("Exception occured in the get_data_preprocessor")
            raise CustomException(e,sys)





    def initiate_data_processing(self,df_path):
        logging.info("Data Preprocessing is initiated")
        try:
            # reading the data with RFM features
            data = pd.read_csv(df_path)
            logging.info("Successfully read data for modelling")

            # splitting data into dependent and independent variables
            y = data["Label"]
            x = data.drop(columns=["Label"])

            # train test splitting
            x_train,x_test,y_train,y_test = train_test_split(x,y,test_size=0.2,random_state=42)
            logging.info("Succesfully done train-test split")

            train_data = pd.concat([x_train, y_train], axis=1)
            train_data.to_csv(self.data_preprocessing_config.train_path,index=False)
            test_data =  pd.concat([x_test, y_test], axis=1)
            test_data.to_csv(self.data_preprocessing_config.test_path,index=False)
            logging.info("Saved Train and Test Data")


            """ PREPROCESSING """
            # getting the preprocessor object
            preprocessor =self.get_data_preprocessor(x_train)

            # applying the preprocessor pipeline to the x_train and x_test
            x_train = preprocessor.fit_transform(x_train,y_train)
            x_test  = preprocessor.transform(x_test)

            logging.info("Applied Transformations")

            # saving the pre-processor object
            save_object(file_path=self.data_preprocessing_config.preprocessor_path,
                        obj=preprocessor)
            logging.info("Succesfully saved preprocessor obj")

            x_train.to_csv("artifacts/potta.csv",index=False)

            return (x_train,y_train,x_test,y_test)
            


        

        except Exception as e:
            logging.info("Exception occured in the initiate_data_transformation")
            raise CustomException(e,sys)
    
        

if __name__=="__main__":
    obj=DataPreProcessing()
    obj.initiate_data_processing("artifacts/modelling_data.csv")
