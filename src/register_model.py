import mlflow
from mlflow.tracking import MlflowClient

def register_model(model_name, run_id):
    client = MlflowClient()
    
    # Register the model
    result = mlflow.register_model(
        f"runs:/{run_id}/model",
        model_name
    )
    
    # Get the latest version
    version = result.version
    
    # Transition the model to 'Production' stage
    client.transition_model_version_stage(
        name=model_name,
        version=version,
        stage="Production"
    )
    
    print(f"Model {model_name} version {version} registered and set to Production stage.")

def main():
    model_name = "BikeShareModel"
    run_id = input("Enter the run ID of the model to register: ")
    
    register_model(model_name, run_id)

if __name__ == "__main__":
    main()