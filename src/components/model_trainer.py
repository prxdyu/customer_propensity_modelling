import pandas as pd
import numpy as np

from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier



from src.logger.logger import logging
from src.exception.exception import CustomException 
from src.utils.utils import evaluate_model,save_object

import os
import sys
from dataclasses import dataclass
from pathlib import Path


@dataclass
class ModelTrainerConfig:
    trained_model_file_path=os.path.join("artifacts","model.pkl")

class ModelTrainer:

    def __init__(self):
        self.model_trainer_config = ModelTrainerConfig()

    def initiate_model_training(self,x_train,y_train,x_test,y_test):
        try:
            models={ 'Logistic Regression': LogisticRegression(),
                     'Decision Tree': DecisionTreeClassifier(),
                     'KNN': KNeighborsClassifier(),
                     'RF':RandomForestClassifier(),
                     'GBDT':GradientBoostingClassifier(),
                     'XGBoost': XGBClassifier()
                     }
            logging.info("Model Fitting Started")
            model_report:dict = evaluate_model(x_train,y_train,x_test,y_test,models)

            print('\n=====================================================')
            print(model_report)

            logging.info(f"Model Report: {model_report} ")

            # To get best model score from dictionary 
            best_model_score = max(sorted(model_report.values()))

            best_model_name = list(model_report.keys())[
                list(model_report.values()).index(best_model_score)
            ]
            
            best_model = models[best_model_name]

            print('\n====================================================================================\n')
            print(f'Best Model Found , Model Name : {best_model_name} , F1 Score : {best_model_score}')

            logging.info(f'Best Model Found , Model Name : {best_model_name} , R2 Score : {best_model_score}')

            save_object(
                 file_path=self.model_trainer_config.trained_model_file_path,
                 obj=best_model
            )

        except Exception as e:
            logging.info("Exception occured in the initiate_data_transformation")
            raise CustomException(e,sys)
        

