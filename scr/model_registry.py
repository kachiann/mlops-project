import mlflow
from mlflow.tracking import MlflowClient
from datetime import datetime

# Set up MLflow tracking URI and experiment
MLFLOW_TRACKING_URI = "sqlite:///mlflow.db"
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
mlflow.set_experiment("bike-sharing-prediction-experiment")

def get_current_model_parameters():
    """
    Get the current parameters of a registered model in production.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    current_accuracy = 0
    run_id_current = None

    for model_version in client.search_model_versions("name='bike-sharing-model'"):
        if model_version.current_stage == "Production":
            prod_model = model_version
            run_id_current = prod_model.run_id
            current_run = client.get_run(run_id=run_id_current)
            current_accuracy = current_run.data.metrics["r2"]
            break

    print(f"run_id_current={run_id_current}")
    print(f"current accuracy = {current_accuracy}")
    return run_id_current, current_accuracy

def get_best_model_parameters():
    """
    Get the parameters of the best trained model.
    """
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    best_run = client.search_runs(
        experiment_ids=['1'],
        filter_string="metrics.r2 > 0",
        order_by=["metrics.r2 DESC"]
    )[0]

    best_run_id = best_run.info.run_id
    best_accuracy = best_run.data.metrics["r2"]
    
    print(f"best_run_id={best_run_id}")
    print(f"best accuracy = {best_accuracy}")

    version = None
    for model_version in client.search_model_versions("name='bike-sharing-model'"):
        if model_version.run_id == best_run_id:
            version = model_version.version
            break

    return best_run_id, best_accuracy, version

if __name__ == '__main__':
    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)
    date = datetime.today().date()
    new_stage = "Production"
    model_name = "bike-sharing-model"

    run_id_current, current_accuracy = get_current_model_parameters()
    best_run_id, best_accuracy, version = get_best_model_parameters()

    if best_accuracy > current_accuracy:
        client.transition_model_version_stage(
            name=model_name,
            version=version, 
            stage=new_stage,
            archive_existing_versions=True
        )
        print(f"Model version {version} registered and set to Production stage.")
    else:
        print("No model update required. The current production model is already the best.")