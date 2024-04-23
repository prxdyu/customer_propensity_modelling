import os
import sys
import pickle

import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

from src.utils.utils import load_object
from src.logger.logger import logging
from src.exception.exception import CustomException

from urllib.parse import urlparse
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score



class ModelEvaluation:

    def __init__(self):
        pass

    def eval_metrics(self,y_true,y_pred):
        rmse=np.sqrt(mean_squared_error(y_true,y_pred))
        mae=mean_absolute_error(y_true,y_pred)
        r2=r2_score(y_true,y_pred)
        return rmse,mae,r2

    def initiate_model_evaluation(self,x_train,y_train,x_test,y_test):
        try:
            logging.info("Model Evaluation Initiated")
            # defining the model path and loading it
            model_path = os.path.join("artifacts","model.pkl")
            model = load_object(model_path)
            # setting the mlflow uri
            mlflow.set_registry_uri("")
            # getting the scheme of the mlflow model registry (local or cloud)
            tracking_url_type=urlparse(mlflow.get_tracking_uri()).scheme
            print(tracking_url_type)

            with mlflow.start_run():
                predictions=model.predict(x_test)
                # getting the metrics
                (rmse,mae,r2)=self.eval_metrics(predictions,y_test)
                # logging the metrics
                mlflow.log_metric("rmse",rmse)
                mlflow.log_metric("mae",mae)
                mlflow.log_metric("r2",r2)
                
                # if the tracking is not a file type then the mlflow is registring models in the cloud or db 
                if tracking_url_type!="file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                # if it is a file then it means mlflow registers models in local
                else:
                    mlflow.sklearn.log_model(model, "model")




        except Exception as e:
            raise CustomException(e,sys)
        
    