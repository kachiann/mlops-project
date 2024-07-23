# Bike Sharing Demand Prediction

## Project Description
This is the implementation of my project for the course mlops-zoomcamp from [DataTalksClub](https://github.com/DataTalksClub/mlops-zoomcamp).
The goal of this project is to build an end-to-end machine learning pipeline to predict bike-sharing demand using historical data. This prediction will help optimize bike distribution and availability in a bike-sharing system. The main focus of the project is on creating a production service with experiment tracking, pipeline automation, and observability.

## Problem Statement
Bike-sharing systems are becoming increasingly popular in urban areas as a convenient and eco-friendly mode of transportation. However, managing the distribution of bikes to meet demand is challenging. The objective of this project is to predict the number of bikes required at different stations at different times of the day to ensure optimal availability and customer satisfaction.
By addressing these challenges through data analysis, the project aims to enhance the overall user experience, increase operational efficiency, and promote sustainable urban transportation.

## Dataset
The dataset used for this project is the "Bike Sharing Demand" dataset, which includes historical data on bike rentals, weather conditions, and timestamps. This dataset is available on [UCI](https://archive.ics.uci.edu/dataset/275/bike+sharing+dataset) Machine Learning Repository.

## Project details
This repository has four folders: *scr*, *notebooks*, *models*, and *data*.
- The `data` folder contains the dataset for the project. It is further divided into:
     - `raw/`: Contains the original, unprocessed dataset.
     - `processed/`: Contains the cleaned and preprocessed data ready for analysis and model training.
- The `notebooks` folder contains Jupyter notebooks used for exploratory data analysis (EDA), feature engineering, and initial model experimentation.
- The `models` folder stores the trained machine learning models and any related artifacts.
- The `scr` folder contains the source code for the project.

## Additional files
- **requirements.txt**
  - Lists all the Python dependencies required for the project.
- **Dockerfile**
  - Defines the Docker image for the project, specifying the environment and dependencies required to run the code.

## Implementation Details

**Experiment Tracking and Model Registry**:
- **MLflow** is used to track experiments, including hyperparameters, metrics, and artifacts.
- Trained models are registered in the MLflow Model Registry.

**Workflow Orchestration**:
**Prefect** is used to create and manage the entire ML pipeline.
The pipeline includes data ingestion, preprocessing, feature engineering, model training, and evaluation steps.

**Model Deployment**:

**Model Monitoring**:

**Reproducibility**:
- Detailed instructions are below to explain how to set up the environment and run the code.
- All dependencies and their versions are specified in `requirements.txt`.


---

