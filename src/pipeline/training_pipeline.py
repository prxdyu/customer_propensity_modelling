import os
import sys
from src.logger.logger import logging
from src.exception.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.feature_engineer import FeatureEngineering
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation
from src.components.data_preprocessing import DataPreProcessing


""" class TrainingPipeline:
    
    def start_data_ingestion(self):
        try:
            data_ingestion=DataIngestion()
            raw_processed_path=data_ingestion.initiate_data_ingestion()
            return raw_processed_path
        except Exception as e:
            raise CustomException(e,sys)
        

    def start_data_transformation(self,raw_processed_path):
        try:
            data_transformation=DataTransformation()
            data_with_rfm_features_path=data_transformation.initiate_data_transformation(raw_processed_path)
            return  data_with_rfm_features_path
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_feature_engineering(self,data_with_rfm_features_path):
        try:
            feature_engineering = FeatureEngineering()
            modelling_data_path = feature_engineering.initiate_feature_engineering(data_with_rfm_features_path)
            return  modelling_data_path
        except Exception as e:
            raise CustomException(e,sys)
        
    def start_data_preprocessing(self,modelling_data_path):
        try:
            data_preprocessor = DataPreProcessing()
            (x_train,y_train,x_test,y_test) = data_preprocessor.initiate_data_processing(modelling_data_path)
            return  (x_train,y_train,x_test,y_test)
        except Exception as e:
            raise CustomException(e,sys)
        

    def start_model_trainer(self,x_train,y_train,x_test,y_test):
        try:
            model_trainer_obj=ModelTrainer()
            model_trainer_obj.initiate_model_training(x_train,y_train,x_test,y_test)
        except Exception as e:
            raise CustomException(e,sys)
        

    def start_model_evaluation(self,x_train,y_train,x_test,y_test):
        try:
           model_eval_obj=ModelEvaluation()
           model_eval_obj.initiate_model_evaluation(x_train,y_train,x_test,y_test)
        except Exception as e:
            raise CustomException(e,sys)
         """
        
    


obj=DataIngestion()
raw_processed_path=obj.initiate_data_ingestion()

data_transformation=DataTransformation()
data_with_rfm_features_file_path=data_transformation.initiate_data_transformation(raw_processed_path)

feature_engineer = FeatureEngineering()
modelling_data_path = feature_engineer.initiate_feature_engineering(data_with_rfm_features_file_path)

data_preprocessor = DataPreProcessing()
(x_train,y_train,x_test,y_test) = data_preprocessor.initiate_data_processing(modelling_data_path)


model_trainer_obj=ModelTrainer()
model_trainer_obj.initiate_model_training(x_train,y_train,x_test,y_test)

model_eval_obj=ModelEvaluation()
model_eval_obj.initiate_model_evaluation(x_train,y_train,x_test,y_test)

