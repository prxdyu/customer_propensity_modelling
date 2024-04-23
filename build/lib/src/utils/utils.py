import os
import sys
import pickle
import numpy as np
import pandas as pd
from src.logger.logger import logging
from src.exception.exception import CustomException
import re

from sklearn.metrics import f1_score

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
            test_model_score = f1_score(y_test,y_test_pred)

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


# function to remove NaN from the list
def compute_avg_purchase_gap(lst):
    assert type(lst)==list
    if len(lst)==1:
        return np.nan
    # removing the NaN values
    lst_new = list(filter(lambda x: not pd.isna(x), lst))
    # returning the avg of the list (since we want avg purchase gap)
    return sum(lst_new)/len(lst_new)


# definin a function which gets avg time between purchases
def get_avg_time_between_purchases(data,action="Action",user_id="User_id",date_time="DateTime"):
    """ 
    data      : dataframe
    action    : name of the action column in the dataframe
    user_id   : name of the user_id column in the dataframe
    date_time : datetime of the action
    """
   # creating a copy of the dataframe
    dff = data.copy()
    # filtering the rows with only purchases
    purchases = dff[dff[action]=="purchase"]
    # selecting only the user id and datetime column
    purchases = purchases[[user_id,date_time]]
    # sorting the purchases by user_id and dattime
    purchases.sort_values(by=[user_id,date_time],ascending=[True,True],inplace=True)
    # creating a purchases_duration column
    purchases.head(2)
    # creating a purchases_duration column
    purchases['Purchase Gap'] = purchases.groupby([user_id])[date_time].diff().dt.total_seconds()
    # converting the timegap to days
    purchases['Purchase Gap'] = np.round(purchases['Purchase Gap']/(3600*24),0)
    # grouping the multiple purchase gaps for an user into a single list (if a cusomter does only single purchase then purchase gap would be NaN, if a person does two purchase then the list contains [nan,purchasegap])
    purchases = pd.DataFrame(purchases.groupby(user_id)["Purchase Gap"].apply(list)).reset_index()
    # applying the compute_avg_purchase_gap func
    purchases["Avg Purchase Gap"] = purchases["Purchase Gap"].apply(compute_avg_purchase_gap)
    
    return purchases




def get_ratios(data,level,action_col,action1,action2):
    """
    data       : name of the dataframe
    level      : the level in which we want to calculate ratios (if we want it in userlevel pass 'user_id column') (if we want it in category level pass category )
    action_col : name of the col that contains the actions
    action1    : action of interest 1
    action2    : action of interest 2 purchase

    this fuctino returns ratios of action1:action2
    """
    dff = data.copy()
    # creating a dataframe which has no of action1 done by each users
    col1 = dff.groupby(level)[action_col].value_counts().loc[:, action1].reset_index()
    # creating a dataframe which has no of action1 done by each users
    col2 = dff.groupby(level)[action_col].value_counts().loc[:, action2].reset_index()
    # combining both dfs
    ratios = pd.merge(col1, col2, on=level, how="left", suffixes=("_"+action1,"_"+action2) )
    # creating the ratio column which contains the action2/action1 value                 
    ratios[f'{action1}_to_{action2}_ratios']=ratios[f"count_{action2}"]/ratios[f"count_{action1}"]
    # filling NaNs with 0
    ratios.fillna(0,inplace=True)
    
    return ratios




# defining a function to get the latest path of each user
def get_paths(data,date_time,user_id,action_col):
    """
    data       : name of the dataframe
    user_id    : name of the col that contains the userid
    action_col : name of the col that contains the actions
    
    this fuctino returns the latest path of each user
    """
    
    # creating a temp copy 
    temp = data.copy()
    
    # sorting the temp df by datetime column in ascending order
    temp.sort_values(by=date_time,ascending=True)
    
    # filtering the rows with add_to_carts action
    add_to_carts = temp[temp[action_col]=="add_to_cart"] 
    
    # finding the last add_to_cart event for each user
    latest_add_to_carts = add_to_carts.groupby(user_id)[date_time].max()
    
    # filtering the rows with actions that was done before the last add to cart action of each user
    temp = temp[temp.DateTime<=temp[user_id].map(latest_add_to_carts)] # now this dataframe consists of actions before the last addtocart event for each users
    
    # picking the last three actions done before add_to_cart by each users
    paths = temp.groupby(user_id)[action_col].apply(list).reset_index(name='Actions_list')
    
    # Joining all the elements of the lists
    paths['Actions_list'] = paths['Actions_list'].apply(lambda x: ' '.join(x))
    
    # applying regex to find the last 3 actions 
    last_3_actions = r'((?:\S+\s+){0,3}\badd_to_cart)'
    paths["Last_3_Actions"] = paths['Actions_list'].apply(lambda x: re.findall(last_3_actions,x))
    
    # getting the latest path of the user
    paths["user_path"] = paths["Last_3_Actions"].apply(lambda x:x[-1])
    paths["user_path"] = paths["user_path"].apply(lambda x:"-->".join(x.split()))
    
    # dropping the unwanted columns
    paths = paths[[user_id,"user_path"]]

    return paths
        
    
# function which combines the minority paths
def combine_paths(data,path_col):
    # getting the top 10 paths 
    top_10_paths = data['user_path'].value_counts(ascending=False).head(10).index.to_list()
    # combining the minority paths
    data['user_path'] = data['user_path'].apply(lambda x:x if x in top_10_paths else "others"  )
    return data 


"""==================================================================================================================="""



""" Defining a function which add dates level features"""
def add_date_level_features(data):
    data['Date'] = data['DateTime'].dt.date
    data['DayOfWeek'] = data['DateTime'].dt.dayofweek
    data['DayOfMonth'] = data['DateTime'].dt.day
    return data


"""=================================================================================================================="""




""" Defining a function which adds user level features"""
def get_user_level_features(data):

    # finding how many days the user were active on our platform
    days_active = data.groupby('User_id').agg({'Date':'nunique'}).reset_index()
    days_active.rename(columns={"Date":"days_active"},inplace=True)

    # computing the avg time between purchases for each user
    avg_purchase_time = get_avg_time_between_purchases(data,"Action","User_id","DateTime")

    # creating cart_purchase ratio for each user
    cart_to_purchase = get_ratios(data,"User_id",'Action',"add_to_cart","purchase")
    
    # creating wish_list_to_purchase ratio for each user
    wishlist_to_purchase = get_ratios(data,"User_id",'Action',"add_to_wishlist","purchase")

    # creating wish_list_click_to_purchase ratio for each user
    wishlist_click_to_purchase = get_ratios(data,"User_id",'Action',"click_wishlist_page","purchase")

    # getting the paths for each users
    paths = get_paths(data,"DateTime","User_id","Action")
    # combining the minority paths
    paths = combine_paths(paths,"user_path")

    return days_active, avg_purchase_time, cart_to_purchase, wishlist_to_purchase, wishlist_click_to_purchase, paths


"""============================================================================================================================="""


""" Defining a function which gives category and subcategory level features"""
def get_category_subcategory_level_features(data):

    # calculating the cart-to-purchase ratio for each categories
    cart_to_purchase_ratio_category =  get_ratios(data,"Category","Action","add_to_cart","purchase").rename(columns={"add_to_cart_to_purchase_ratios":"cart_to_purchase_ratios_category"})

    # calculating the cart-to-purchase ratio for each categories
    cart_to_purchase_ratio_subcategory =  get_ratios(data,"SubCategory","Action","add_to_cart","purchase").rename(columns={"add_to_cart_to_purchase_ratios":"cart_to_purchase_ratios_subcategory"})

    # calculating the cart-to-purchase ratio for each categories
    wishlist_to_purchase_ratio_category =  get_ratios(data,"Category","Action","add_to_wishlist","purchase").rename(columns={"add_to_wishlist_to_purchase_ratios":"wishlist_to_purchase_ratios_category"})
   
    # calculating the cart-to-purchase ratio for each categories
    wishlist_to_purchase_ratio_subcategory =  get_ratios(data,"SubCategory","Action","add_to_wishlist","purchase").rename(columns={"add_to_wishlist_to_purchase_ratios":"wishlist_to_purchase_ratios_subcategory"})
    

    # calculating the cart-to-purchase ratio for each categories
    click_wishlist_to_purchase_ratio_category =  get_ratios(data,"Category","Action","click_wishlist_page","purchase").rename(columns={"click_wishlist_page_to_purchase_ratios":"click_wishlist_to_purchase_ratios_category"})

    # calculating the cart-to-purchase ratio for each subcategories
    click_wishlist_to_purchase_ratio_subcategory =  get_ratios(data,"SubCategory","Action","click_wishlist_page","purchase").rename(columns={"click_wishlist_page_to_purchase_ratios":"click_wishlist_to_purchase_ratios_subcategory"})

    # calculating the cart-to-purchase ratio for each subcategories
    view_to_purchase_ratio_category =  get_ratios(data,"Category","Action","product_view","purchase").rename(columns={"product_view_to_purchase_ratios":"product_view_to_purchase_ratios_category"})

    # calculating the cart-to-purchase ratio for each subcategories
    view_to_purchase_ratio_subcategory =  get_ratios(data,"SubCategory","Action","product_view","purchase").rename(columns={"product_view_to_purchase_ratios":"product_view_to_purchase_ratios_subcategory"})

    return cart_to_purchase_ratio_category, cart_to_purchase_ratio_subcategory, wishlist_to_purchase_ratio_category, wishlist_to_purchase_ratio_subcategory, click_wishlist_to_purchase_ratio_category, click_wishlist_to_purchase_ratio_subcategory, view_to_purchase_ratio_category, view_to_purchase_ratio_subcategory


"""==============================================================================================================================="""


""" defining the functions to get dataset for training"""

# function to get the label for the row
def get_label(lst):
    
    if len(lst)==1: #it means the user only added_to_cart and haven't bought anything
        return 0
    # Convert strings to datetime objects
    cart_time = pd.to_datetime(lst[0])
    purchase_time = pd.to_datetime(lst[1])
    # Calculate time difference
    time_diff = purchase_time - cart_time
    # Define a timedelta representing 2 hours
    two_hours = pd.Timedelta(hours=2)
    # Check if the time difference is less than or equal to 2 hours
    if time_diff <= two_hours:
        return 1
    else:
        return 0



# function to get the dataset
def get_dataset(data):
    dff= data.copy()
    # filtering the data only for purchase and add_to_cart action
    data = dff[dff['Action'].isin(["purchase", "add_to_cart"])]
    data.sort_values(by=["User_id","DateTime"],inplace=True)

    # grouping rows using "User_id", "Category", "SubCategory" and creating the action list anf respective datetime list
    data_grouped = data.groupby(["User_id", "Category", "SubCategory"]).agg({
        "Action": list,
        "DateTime": lambda x: x.tolist()
    })
    data_grouped.reset_index(inplace=True)
    # filtering rows where add_to_cart action is followed by purchase action
    filtered_data = data_grouped[data_grouped['Action'].apply(lambda x: (x == ['add_to_cart', 'purchase']) or (x == ['add_to_cart'] ))]
    # getting the labels if the difference between the datetime elements in the list is less than 2 hrs
    filtered_data["Label"]=filtered_data["DateTime"].apply(get_label)

    # dropping the unwanted columns
    filtered_data.drop(columns=["Action","DateTime"],inplace=True)

    return filtered_data 


""" function wich adds user level features"""
def add_user_level_features(df_base,df,days_active, avg_purchase_time, cart_to_purchase, wishlist_to_purchase, wishlist_click_to_purchase, paths):
    
    # adding no of days active feature
    df_base = pd.merge(df_base,days_active,on="User_id",how="left")

    # adding RFM features for the users
    users_rfm_features = df.groupby("User_id").agg({"R":"max","F":"max","M":"max","Loyalty":"max"})
    df_base = pd.merge(df_base,users_rfm_features,on="User_id",how="left")

    # adding avg_time between purchase
    df_base = pd.merge(df_base,avg_purchase_time[["User_id","Avg Purchase Gap"]],on="User_id",how="left")
            
    # adding the carts_to_purchase ratios for each user
    df_base = pd.merge(df_base,cart_to_purchase[["User_id","add_to_cart_to_purchase_ratios"]],on="User_id",how="left")
    
    # adding the wishlist_to_purchase ratios for each user
    df_base = pd.merge(df_base,wishlist_to_purchase[["User_id","add_to_wishlist_to_purchase_ratios"]],on="User_id",how="left")

    # adding the click_wishlist_to_purchase ratios for each user
    df_base = pd.merge(df_base,wishlist_click_to_purchase[["User_id","click_wishlist_page_to_purchase_ratios"]],on="User_id",how="left")

    # adding the latest path for each user
    df_base = pd.merge(df_base,paths,on="User_id",how="left")

    return df_base


""" function wich adds user category and subcategory level  features"""
def add_category_subcategory_level_features(df_base,cart_to_purchase_ratio_category, cart_to_purchase_ratio_subcategory, wishlist_to_purchase_ratio_category, wishlist_to_purchase_ratio_subcategory, click_wishlist_to_purchase_ratio_category, click_wishlist_to_purchase_ratio_subcategory, view_to_purchase_ratio_category, view_to_purchase_ratio_subcategory):

    # adding category and sub-category level features carts_to_purchase ratios
    df_base = pd.merge(df_base,cart_to_purchase_ratio_category[["Category","cart_to_purchase_ratios_category"]],on="Category",how="left")
    df_base = pd.merge(df_base,cart_to_purchase_ratio_subcategory[["SubCategory","cart_to_purchase_ratios_subcategory"]],on="SubCategory",how="left")

    # adding category and sub-category level features wishlist_to_purchase ratios
    df_base = pd.merge(df_base,wishlist_to_purchase_ratio_category[["Category","wishlist_to_purchase_ratios_category"]],on="Category",how="left")
    df_base = pd.merge(df_base,wishlist_to_purchase_ratio_subcategory[["SubCategory","wishlist_to_purchase_ratios_subcategory"]],on="SubCategory",how="left")

    # adding category and sub-category level features click_wishlist_to_purchase ratios
    df_base = pd.merge(df_base,click_wishlist_to_purchase_ratio_category[["Category","click_wishlist_to_purchase_ratios_category"]],on="Category",how="left")
    df_base = pd.merge(df_base,click_wishlist_to_purchase_ratio_subcategory[["SubCategory","click_wishlist_to_purchase_ratios_subcategory"]],on="SubCategory",how="left")

    # adding category and sub-category level features view_to_purchase ratios
    df_base = pd.merge(df_base,view_to_purchase_ratio_category[["Category","product_view_to_purchase_ratios_category"]],on="Category",how="left")
    df_base = pd.merge(df_base,view_to_purchase_ratio_subcategory[["SubCategory","product_view_to_purchase_ratios_subcategory"]],on="SubCategory",how="left")

    # dropping the user_id column
    df_base.drop(columns=["User_id"],inplace=True)
    
    return df_base