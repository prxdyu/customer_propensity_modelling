import os
import sys
from src.logger.logger import logging
from src.exception.exception import CustomException
import pandas as pd

from src.components.data_ingestion import DataIngestion
from src.components.data_transformation import DataTransformation
from src.components.model_trainer import ModelTrainer
from src.components.model_evaluation import ModelEvaluation


class TrainingPipeline:
    
    def start_data_ingestion(self):
        try:
            data_ingestion=DataIngestion()
            train_data_path,test_data_path=data_ingestion.initiate_data_ingestion()
            return train_data_path,test_data_path
        except Exception as e:
            raise CustomException(e,sys)
        

    def start_data_transformation(self,train_data_path,test_data_path):
        try:
            data_transformation=DataTransformation()
            x_train,y_train,x_test,y_test=data_transformation.initiate_data_transformation(train_data_path,test_data_path)
            return  x_train,y_train,x_test,y_test
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
        
        
    


# obj=DataIngestion()
# train_data_path,test_data_path=obj.initiate_data_ingestion()

# data_transformation=DataTransformation()
# x_train,y_train,x_test,y_test=data_transformation.initiate_data_transformation(train_data_path,test_data_path)

# model_trainer_obj=ModelTrainer()
# model_trainer_obj.initiate_model_training(x_train,y_train,x_test,y_test)

# model_eval_obj=ModelEvaluation()
# model_eval_obj.initiate_model_evaluation(x_train,y_train,x_test,y_test)

