# Customer Propensity Model

## 1. Introduction to the Project
The Customer Propensity Model project aims to assist an early-stage e-commerce company in increasing its conversion rates by predicting the likelihood of a user making a purchase. By analyzing user behavior and historical data, the model predicts the probability of a user purchasing a product within a specified time frame, allowing the company to target users with personalized marketing campaigns effectively.
## 2. Project Structure

```bash
CustomerPropensityModel
│
├── .github
│   └── workflows
│       └── automate.yml
│
├── CustomerPropensityModel.egg-info
│
├── artifacts
│   ├── model.pkl
│   ├── modelling_data.csv
│   ├── preprocessor.pkl
│   ├── raw.csv
│   ├── raw_processed.csv
│   └── raw_with_rfm_features.csv
│
├── build
│
├── dist
│
├── logs
│
├── mlruns
│   └── 0
│
├── notebooks
│   ├── EDA with RFM Modelling.ipynb
│   ├── Feature Engineering.ipynb
│   └── Model building.ipynb
│
├── src
│   ├── components
│   │   ├── data_ingestion.py
│   │   ├── data_preprocessing.py
│   │   ├── data_transformation.py
│   │   ├── feature_engineer.py
│   │   ├── model_evaluation.py
│   │   └── model_trainer.py
│   │
│   ├── exception
│   │   └── logger.py
│   │
│   ├── pipeline
│   │   ├── prediction_pipeline.py
│   │   └── training_pipeline.py
│   │
│   └── utils
│       └── utils.py
│
├── templates
│   ├── form.html
│   ├── index.html
│   └── result.html
│
├── .dockerignore
├── .gitignore
├── Dockerfile
├── app.py
├── init_setup.sh
├── requirements.txt
├── requirements_dev.txt
├── setup.py
└── test.py

```
The project is organized into several components:

- **artifacts**: Contains datasets and pre-trained models.
- **src**: packages the source code for data processing, model training, and evaluation.
- **app.py**: A simple flask application to build API for getting the model predictions.
- **Dockerfile**: Defines the Docker image for containerizing the flask application.
- **automate.yml**: Configures CI/CD pipelines for automated deployment.

# 3. Src Package and Training/Prediction Pipeline

The src package contains modules for data processing, model training, and evaluation
* **Data Ingestion**: Handles data loading and preprocessing.
* **Feature Engineering**: Performs feature engineering and generates RFM (Recency, Frequency, Monetary) features.
* **Data Preprocessing**: Preprocesses data for model training.
* **Model Trainer**: Trains the machine learning model using preprocessed data.
* **Model Evaluation**: Evaluates the trained model's performance.

We have 2 pipelines which are 
- **Training Pipeline**: The training pipeline consists of several steps including Data ingestion, Feature engineering, Data Preprocessing etc.. to train the model
- **Prediction Pipeline**: The prediction pipeline utilizes the trained model to predict user purchase propensity based on input data.

# 4. Flask Application

The Flask application serves as the interface for interacting with the Customer Propensity Model. It provides endpoints for viewing the home page, testing the server's availability, and making predictions using the trained model.

### Endpoints

1. **Home Page** (`/`): Renders the `index.html` template, which provides options for making predictions.
2. **Ping Endpoint** (`/ping`): A simple endpoint for testing the server's availability. Returns "Success" when accessed via a GET request.
3. **Prediction Endpoint** (`/predict`): Accepts both GET and POST requests. When accessed via GET, it renders a form (`form.html`) populated with options for selecting input features. Upon submitting the form via POST request, it processes the input data, makes predictions using the trained model, and renders the result (`result.html`).

### Input Features

The prediction endpoint accepts the following input features:

- **Category**: The category of the product.
- **Subcategory**: The subcategory of the product.
- **Days Active**: Number of days the user has been active.
- **R**: Recency (a measure of how recently the user made a purchase).
- **F**: Frequency (a measure of how often the user makes purchases).
- **M**: Monetary value (a measure of how much money the user spends).
- **Loyalty**: Loyalty status of the user.
- **Avg Purchase Gap**: Average time gap between purchases.
- **Add to Cart to Purchase Ratios**: Ratio of add-to-cart actions to purchases.
- **Add to Wishlist to Purchase Ratios**: Ratio of add-to-wishlist actions to purchases.
- **Click Wishlist Page to Purchase Ratios**: Ratio of clicks on wishlist page to purchases.
- **User Path**: Path followed by the user on the website.
- **Cart to Purchase Ratios (Category and Subcategory)**: Ratios of cart actions to purchases for both category and subcategory.
- **Wishlist to Purchase Ratios (Category and Subcategory)**: Ratios of wishlist actions to purchases for both category and subcategory.
- **Click Wishlist to Purchase Ratios (Category and Subcategory)**: Ratios of clicks on wishlist to purchases for both category and subcategory.
- **Product View to Purchase Ratios (Category and Subcategory)**: Ratios of product views to purchases for both category and subcategory.

### Output

The prediction endpoint returns the predicted probability of the user making a purchase, expressed as a percentage.

#5. Dockerfile and Containerization

The Dockerfile provided in the project repository allows for containerizing the Customer Propensity Model application using a multi-stage build strategy. This strategy helps reduce the size of the final Docker image by separating the build dependencies from the runtime environment.

### Dockerfile Explanation

The Dockerfile consists of two stages:

1. **Builder Stage**: In this stage, a Python 3.8 slim-buster image is used to install the project dependencies specified in the `requirements.txt` file. This stage sets the working directory to `/install` and copies only the `requirements.txt` file to leverage Docker's caching mechanism. It then installs the dependencies into the `/install` directory using `pip`. This stage is responsible for creating a temporary image used for building the dependencies.

2. **Final Stage**: The final Docker image is created based on another Python 3.8 slim-buster image. This stage sets the working directory to `/app` and copies the installed dependencies from the builder stage into the `/usr/local` directory. It then copies the rest of the application files into the `/app` directory. After copying, any unnecessary files are cleaned up to reduce the image size. Finally, the command to run the Flask application is specified using the `CMD` directive, which starts the Flask server on `0.0.0.0:5000`.

### Building the Docker Image

To build the Docker image for the Customer Propensity Model application, navigate to the project directory containing the Dockerfile and execute the following command:

```bash
docker build -t customer-propensity-model .
```

Once the Docker image is built, you can run a Docker container using the following command:
```bash
docker run -d -p 5000:5000 customer-propensity-model
```
This command will start a Docker container based on the customer-propensity-model image, exposing port 5000 on the host machine. You can then access the Customer Propensity Model application by visiting http://localhost:5000 in your web browser


