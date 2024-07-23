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
mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("Bike Sharing Demand Prediction")

def load_data(filepath):
    """Load the processed data from CSV."""
    print('Loading the data...')
    return pd.read_csv(filepath)

def split_data(df):
    """Split the data for model training."""
    print('Splitting the data...')
    
    # Select features for model
    features = ['season', 'holiday', 'workingday', 'weathersit', 'temp', 'atemp', 
                'hum', 'windspeed', 'hr', 'day', 'mnth', 'yr', 
                'day_of_week', 'rush_hour']
    
    X = df[features]
    y = df['cnt']
    
    return X, y

def train_model(X_train, y_train):
    """Train a Random Forest model."""
    print('Training the model...')
    model = RandomForestRegressor(n_estimators=100, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluate the model and return performance metrics."""
    print('Evaluating the model...')
    y_pred = model.predict(X_test)
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    r2 = r2_score(y_test, y_pred)
    return rmse, r2

def save_model(model, filepath):
    """Save the trained model to a file."""
    print('Saving the model...')
    joblib.dump(model, filepath)

def register_model(run_id):
    model_name = "Bike_Sharing_Demand_Model"
    model_uri = f"runs:/{run_id}/model"
    mlflow.register_model(model_uri, model_name)

def main():
    # Load data
    data_path = '/Users/kachiemenike/My Documents/Personal/mlops-project/data/processed/train_bike_sharing_data_processed.csv'
    df = load_data(data_path)
    
    # Preprocess data
    X, y = split_data(df)
    
    # Split data into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Start MLflow run
    with mlflow.start_run() as run:
        print('Started MLflow run')
        
        # Train model
        model = train_model(X_train, y_train)
        
        # Evaluate model
        rmse, r2 = evaluate_model(model, X_test, y_test)
        
        # Log parameters
        mlflow.log_param("model_type", "RandomForestRegressor")
        mlflow.log_param("n_estimators", 100)
        
        # Log metrics
        mlflow.log_metric("rmse", rmse)
        mlflow.log_metric("r2", r2)
        
        # Log model
        mlflow.sklearn.log_model(model, "model")
        
        # Save model locally
        model_path = f'/Users/kachiemenike/My Documents/Personal/mlops-project/models/bike_sharing_model_{datetime.now().strftime("%Y%m%d_%H%M%S")}.joblib'
        save_model(model, model_path)
        
        run_id = run.info.run_id
        print(f"Run ID: {run_id}")

        print(f"Model training completed. RMSE: {rmse:.2f}, R2: {r2:.2f}")
        print(f"Model saved to {model_path}")
        print(f"Run ID: {run_id}")
        print('Done')
        
        # Register the model
        register_model(run_id)
        
    return run_id

if __name__ == "__main__":
    main()
