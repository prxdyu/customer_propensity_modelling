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
from sklearn.metrics import f1_score,confusion_matrix



class ModelEvaluation:

    def __init__(self):
        pass

    def eval_metrics(self,y_true,y_pred):
        # computing f1 score
        f1score = f1_score(y_true, y_pred)
        # computing flase positive and false negative rate
        tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
        fpr = fp / (fp + tn)
        fnr = fn / (tp + fn)
        return f1score,fpr,fnr

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
                (f1score,fpr,fnr)=self.eval_metrics(predictions,y_test)
                # logging the metrics
                mlflow.log_metric("f1_score",f1score)
                mlflow.log_metric("false_positive_rate",fpr)
                mlflow.log_metric("false_negative_rate",fnr)
                
                # if the tracking is not a file type then the mlflow is registring models in the cloud or db 
                if tracking_url_type!="file":
                    mlflow.sklearn.log_model(model, "model", registered_model_name="ml_model")
                # if it is a file then it means mlflow registers models in local
                else:
                    mlflow.sklearn.log_model(model, "model")




        except Exception as e:
            raise CustomException(e,sys)
        
    