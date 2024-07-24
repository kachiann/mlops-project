from prefect import flow
from prefect.deployments import Deployment

@flow
def bike_sharing_prediction_pipeline():
    print("This is the bike sharing prediction pipeline")

if __name__ == "__main__":
    deployment = Deployment.build_from_flow(
        flow=bike_sharing_prediction_pipeline,
        name="bike-sharing-prediction-pipeline",
        version="1",
        work_queue_name="default",
    )
    deployment_id = deployment.apply()
    print(f"Deployment created with ID: {deployment_id}")