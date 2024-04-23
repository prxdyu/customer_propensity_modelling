import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logger import logging
from src.exception.exception import CustomException

from sklearn.metrics import r2_score, mean_absolute_error,mean_squared_error

def save_object(file_path, obj):
    try:
        dir_path = os.path.dirname(file_path)

        os.makedirs(dir_path, exist_ok=True)

        with open(file_path, "wb") as file_obj:
            pickle.dump(obj, file_obj)

    except Exception as e:
        raise CustomException(e, sys)
    
def evaluate_model(X_train,y_train,X_test,y_test,models):
    try:
        report = {}
        for i in range(len(models)):
            model = list(models.values())[i]
            # Train model
            model.fit(X_train,y_train)

            

            # Predict Testing data
            y_test_pred =model.predict(X_test)

            # Get R2 scores for train and test data
            #train_model_score = r2_score(ytrain,y_train_pred)
            test_model_score = r2_score(y_test,y_test_pred)

            report[list(models.keys())[i]] =  test_model_score

        return report

    except Exception as e:
        logging.info('Exception occured during model training')
        raise CustomException(e,sys)
    
def load_object(file_path):
    try:
        with open(file_path,'rb') as file_obj:
            return pickle.load(file_obj)
    except Exception as e:
        logging.info('Exception Occured in load_object function utils')
        raise CustomException(e,sys)
    

""" Function which gets the RFM features"""
def get_rfm_features(data, cust_id="User_id", invoice_date="DateTime", total_sales="Total Price"):
    # creating a copy of the dataframe
    temp = data.copy()
    # converting the customer_id column to object type
    temp[cust_id] = temp[cust_id].astype("object")

    """ RECENCY"""
    # computing the last purchase feature of the users
    temp['LastPurchase'] = temp.groupby(cust_id)[invoice_date].transform(lambda x: x.max(skipna=True))
    # getting the last invoice
    last_invoice = temp[invoice_date].max() + pd.Timedelta(days=1)
    # compuint the recency feature
    temp['Recency'] = (last_invoice-temp['LastPurchase']).dt.days
    # creating a separate dataframe which consists receny for each users
    users_receny = temp[[cust_id,"Recency"]].drop_duplicates().sort_values(by=cust_id).set_index(cust_id)


    """ FREQUENCY """
    # creating a df which contains frequency features for each users
    users_frequency = pd.DataFrame(temp[cust_id].value_counts().sort_index()).rename(columns={"count":"Frequency"})

    """ MONETARY """
    users_monetary = pd.DataFrame(temp.groupby(cust_id)[total_sales].sum().sort_index()).rename(columns={total_sales: "Monetary"})

    """ RFM Dataframe"""
    # Joining Recency and Frequency DataFrames
    RFM = pd.merge(users_receny, users_frequency, left_index=True, right_index=True)
    # Joining with Monetary DataFrame
    RFM = pd.merge(RFM, users_monetary, left_index=True, right_index=True).reset_index()

    return RFM



""" defining  a function which assings the score based on the thresholds """
def assign_score(x,quants,attribute):
    percentile25 = quants[0.25]
    percentile50 = quants[0.5]
    percentile75 = quants[0.75]

    # low recent ==> value high R score
    if attribute == "Recency":
        if x<=percentile25:
            return 4
        elif (x>percentile25) & (x<=percentile50):
            return 3
        elif (x>percentile50) & (x<=percentile75):
            return 2
        else:
            return 1

    # low frequency/monetary ==> low F/M score
    elif attribute in ("Frequency","Monetary"):
         if x<=percentile25:
            return 1
         elif (x>percentile25) & (x<=percentile50):
            return 2
         elif (x>percentile50) & (x<=percentile75):
            return 3
         else:
            return 4



""" Defining a function which generates the final data with RFM features"""
def get_rfm_scores(data):

    # we need to assing a score from 1-4 so we need 3 quantiles for each attributes of RFM
    recency_quantiles   = data['Recency'].quantile([.25,.50,.75]).to_dict()
    frequency_quantiles = data['Frequency'].quantile([.25,.50,.75]).to_dict()
    monetary_quantiles  = data['Monetary'].quantile([.25,.50,.75]).to_dict()

    # assinging R score for recency
    data['R'] = data['Recency'].apply(lambda x:assign_score(x, recency_quantiles, "Recency"))
    # assinging F score for frequency
    data['F'] = data['Frequency'].apply(lambda x:assign_score(x, frequency_quantiles, "Frequency"))
    # assinging M score for Monetary
    data['M'] = data['Monetary'].apply(lambda x:assign_score(x, monetary_quantiles, "Monetary"))

    # finding the in which group each user falls
    data['Group'] = data['R'].apply(str) + data['F'].apply(str) + data['M'].apply(str)

    # creating RFM scores
    data['Score'] = data[['R','F','M']].sum(axis=1)

    # defining our 4 groups interms of loyalty
    loyalty = ["Bronze", "Silver", "Gold", "Platinum"]
    
    # creating a column called Loyalty
    data['Loyalty'] = pd.qcut(data['Score'], q=4, labels=loyalty)

    return data
        