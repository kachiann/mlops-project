import pandas as pd
import numpy as np

def load_data(filepath):
    """Load the raw data from a CSV file."""
    return pd.read_csv(filepath)

def clean_data(df):
    """Clean the data by handling missing values and correcting data types."""
    # Check for missing values
    missing_values = df.isnull().sum()
    print(f"Missing values before cleaning:\n{missing_values}")
    
    # Convert datetime column to datetime object
    df['dteday'] = pd.to_datetime(df['dteday'])
    
    return df

def feature_engineering(df):
    """Create new features from the existing data."""
    # Extract additional features from the datetime column
    df['day_of_week'] = df['dteday'].dt.dayofweek
    df['day'] = df['dteday'].dt.day

    # Convert season to categorical
    df['season'] = df['season'].astype('category')

    # Convert weather to categorical
    df['weathersit'] = df['weathersit'].astype('category')

    # Create a binary feature for rush hour (7-9 AM and 4-7 PM)
    df['rush_hour'] = df['hr'].apply(lambda x: 1 if (7 <= x <= 9) or (16 <= x <= 19) else 0)
    
    return df

def save_data(df, filepath):
    """Save the cleaned and processed data to a CSV file."""
    df.to_csv(filepath, index=False)
    print(f"Processed data saved to {filepath}")

def main():
    # Load raw data
    raw_data_path = '/Users/kachiemenike/My Documents/Personal/mlops-project/data/raw/hour.csv'
    df = load_data(raw_data_path)
    
    # Clean data
    df = clean_data(df)
    
    # Feature engineering
    df = feature_engineering(df)
    
    # Save processed data
    processed_data_path = '/Users/kachiemenike/My Documents/Personal/mlops-project/data/processed/train_bike_sharing_data_processed.csv'
    save_data(df, processed_data_path)
    
    print("Data processing completed successfully.")

if __name__ == "__main__":
    main()