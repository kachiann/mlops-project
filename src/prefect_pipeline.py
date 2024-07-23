# prefect_pipeline.py

import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score

import mlflow
from mlflow.tracking import MlflowClient

from prefect import flow, task
from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from prefect.task_runners import SequentialTaskRunner

from datetime import datetime
import joblib
import os


# Set up MLflow
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("bike-sharing-prediction-experiment")

@task
def load_data(filepath):
    """Load the raw data from CSV."""
    print('Loading the data...')
    return pd.read_csv(filepath)

@task
def preprocess_data(df):
    """Clean and preprocess the data."""
    print('Preprocessing the data...')
    df['dteday'] = pd.to_datetime(df['dteday'])
    df['day_of_week'] = df['dteday'].dt.dayofweek
    df['day'] = df['dteday'].dt.month
    
    # Handle missing values if any
    df = df.dropna()
    
    return df

@task
def feature_engineering(df):
    """Create new features."""
    print('Performing feature engineering...')
    df['is_weekend'] = df['day_of_week'].isin([5, 6]).astype(int)
    df['weathersit'] = df['weathersit'].astype('category')
    df['season'] = df['season'].astype('category')
    
    return df

@task
def split_data(df):
    """Split the data into features and target, and then into train and test sets."""
    print('Splitting the data...')
    features = ['season', 'holiday', 'workingday', 'weathersit', 'temp', 'atemp', 
                'hum', 'windspeed', 'hr', 'day_of_week', 'mnth', 
                'is_weekend']
    X = df[features]
    y = df['cnt']
    
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test

@task
def train_model(X_train, y_train):
    """Train a Random Forest model."""
    print('Training the model...')
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    print("training is complete")

    # Save the model locally
    model_dir = "models"
    os.makedirs(model_dir, exist_ok=True)
    model_path = os.path.join(model_dir, f"bike_sharing_model_{datetime.now().strftime('%Y%m%d_%H%M%S')}.joblib")
    joblib.dump(model, model_path)
    print(f"Model saved locally at: {model_path}")
    
    return model, model_path

@task
def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return performance metrics i.e., mse,rmse, and r_squared."""
    print('Evaluating the model...')
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    print('done calculating')
    return rmse, r2

@task
def log_model(model, model_path, rmse, r2):
    """Log the model and its metrics to MLflow."""
    print('Logging the model...')
    with mlflow.start_run():
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Log the locally saved model file
        mlflow.log_artifact(model_path)
        
        # Log the model to MLflow model registry
        mlflow.sklearn.log_model(model, "model")
        
        run_id = mlflow.active_run().info.run_id
    return run_id

@task
def register_model(run_id):
    """Register the model in MLflow Model Registry."""
    print('Registering the model...')
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"
    model_name = "bike-sharing-model"
    
    # Register the model
    model_details = mlflow.register_model(model_uri, model_name)
    
    # Transition the model to Production stage
    client.transition_model_version_stage(
        name=model_name,
        version=model_details.version,
        stage="Production",
        archive_existing_versions=True
    )
    print(f"Model version {model_details.version} registered and set to Production stage.")

@flow(task_runner=SequentialTaskRunner())
def bike_sharing_prediction_pipeline():
    """Main pipeline for bike sharing prediction."""
    # Data ingestion
    df = load_data("/Users/kachiemenike/My Documents/Personal/mlops-project/data/raw/hour.csv")
    
    # Data preprocessing
    df_cleaned = preprocess_data(df)
    
    # Feature engineering
    df_featured = feature_engineering(df_cleaned)
    
    # Data splitting
    X_train, X_test, y_train, y_test = split_data(df_featured)
    
    # Model training
    model = train_model(X_train, y_train)
    
    # Model evaluation
    rmse, r2 = evaluate_model(model, X_test, y_test)
    
    # Log model to MLflow
    run_id = log_model(model, rmse, r2)
    
    # Register model
    register_model(run_id)
    
    print(f"Pipeline completed. Model performance: RMSE = {rmse:.4f}, R2 = {r2:.4f}")

if __name__ == "__main__":
    deployment = Deployment.build_from_flow(
        flow=bike_sharing_prediction_pipeline,
        name="bike-sharing-prediction-pipeline",
        schedule=(CronSchedule(cron="5 4 1 * *", timezone="UTC")),
        tags=["final_project"]
    )
    deployment.apply()
    print("Deployment has been created")