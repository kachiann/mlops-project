from prefect.deployments import Deployment
from prefect.server.schemas.schedules import CronSchedule
from prefect.infrastructure.docker import DockerContainer
from prefect.task_runners import SequentialTaskRunner

# Import your flow
from prefect_pipeline import bike_sharing_prediction_pipeline 

if __name__ == "__main__":
    docker_container = DockerContainer(
        image="bike-sharing-predictor:latest",
        image_pull_policy="ALWAYS",
        auto_remove=True,
    )

    # Build the deployment
    deployment = Deployment.build_from_flow(
        flow=bike_sharing_prediction_pipeline,
        name="bike-sharing-prediction-pipeline",
        version="1",
        work_queue_name="default",
        schedule=(CronSchedule(cron="5 4 1 * *", timezone="UTC")),
        infrastructure=docker_container,
        tags=["final_project"],
        parameters={"param1": "value1", "param2": "value2"},
    )

    # Apply the deployment
    deployment_id = deployment.apply()

    print(f"Deployment has been created with ID: {deployment_id}")