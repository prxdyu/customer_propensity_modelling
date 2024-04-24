from flask import Flask,request,render_template
from src.pipeline.prediction_pipeline import PredictionPipeline,CustomData
import pandas as pd

app=Flask(__name__)


@app.route('/')
def home_page():
    return render_template("index.html")

@app.route("/predict",methods=["GET","POST"])
def predict_datapoint():
    if request.method=="GET":
        df=pd.read_csv("artifacts/raw_with_rfm_features.csv")
        # getting the category  list
        categories = df['Category'].unique().tolist()
        # getting the subcategory list  
        subcategories = df['SubCategory'].unique().tolist()
        # getting the loyaty list
        loyalties = df['Loyalty'].dropna().unique().tolist()
        # getting the path list
        df = pd.read_csv("artifacts/modelling_data.csv")
        paths = df['user_path'].unique().tolist()

        return render_template("form.html",categories=categories,subcategories=subcategories,loyalties=loyalties,paths=paths)
    
    else:
        data=CustomData(
            Category = request.form.get("category"),
            SubCategory = request.form.get("subcategory"),
            days_active = request.form.get("days_active"),
            R = float(request.form.get("R")),
            F = float(request.form.get("F")),
            M = float(request.form.get("M")),
            Loyalty = request.form.get("loyalty"),
            AvgPurchaseGap = request.form.get("AvgPurchaseGap"),
            add_to_cart_to_purchase_ratios = request.form.get("add_to_cart_to_purchase_ratios"),
            add_to_wishlist_to_purchase_ratios = request.form.get("add_to_wishlist_to_purchase_ratios"),
            click_wishlist_page_to_purchase_ratios = request.form.get("click_wishlist_page_to_purchase_ratios"),
            user_path = request.form.get("path"),
            cart_to_purchase_ratios_category = request.form.get("cart_to_purchase_ratios_category"),
            cart_to_purchase_ratios_subcategory = request.form.get("cart_to_purchase_ratios_subcategory"),
            wishlist_to_purchase_ratios_category = request.form.get("wishlist_to_purchase_ratios_category"),
            wishlist_to_purchase_ratios_subcategory = request.form.get("wishlist_to_purchase_ratios_subcategory"),
            click_wishlist_to_purchase_ratios_category = request.form.get("click_wishlist_to_purchase_ratios_category"),
            click_wishlist_to_purchase_ratios_subcategory = request.form.get("click_wishlist_to_purchase_ratios_subcategory"),
            product_view_to_purchase_ratios_category = request.form.get("product_view_to_purchase_ratios_category"),
            product_view_to_purchase_ratios_subcategory = request.form.get("product_view_to_purchase_ratios_subcategory")
            
            )
        final_data=data.get_data_as_df()

        predict_pipeline=PredictionPipeline()

        pred=predict_pipeline.predict(final_data)

        result=round(pred[0],2)

        return render_template("result.html",final_result=result*100)



if __name__=="__main__":
    app.run(host="0.0.0.0",port=8000)