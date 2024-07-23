import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import mlflow
import mlflow.sklearn
from datetime import datetime

# Set up MLflow
mlflow.set_experiment("Bike Sharing Demand Prediction")