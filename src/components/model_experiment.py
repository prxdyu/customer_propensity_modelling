import os
import sys
import pickle


import mlflow
import mlflow.sklearn

import pandas as pd
import numpy as np

from src.utils.utils import load_object
from urllib.parse import urlparse
from sklearn.metrics import mean_absolute_error,mean_squared_error,r2_score