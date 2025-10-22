import pandas as pd
import logging
from sklearn.model_selection import train_test_split
import os
import kagglehub
from kagglehub import KaggleDatasetAdapter

# Logging setup
log_dir = "logs"
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_ingestion")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

log_file_path = os.path.join(log_dir, "data_ingestion.log")
file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)


def download_the_data(data_url) -> str:
    """Download the dataset from KaggleHub and return the file path"""
    logger.debug("Downloading dataset: %s", data_url)
    dataset_path = kagglehub.dataset_download(data_url)
    files = os.listdir(dataset_path)

    csv_files = [f for f in files if f.endswith('.csv')]
    if not csv_files:
        raise FileNotFoundError("No CSV dataset found in this Kaggle dataset")

    csv_file = csv_files[0]
    file_path = os.path.join(dataset_path, csv_file)
    logger.debug("Dataset downloaded and found file: %s", file_path)
    return file_path


def load_data(file_path: str) -> pd.DataFrame:
    """Load the data from CSV"""
    try:
        df = pd.read_csv(file_path)
        logger.debug("Loaded data successfully from %s", file_path)
        return df
    except pd.errors.ParserError as e:
        logger.error("Failed to parse the CSV file: %s", e)
        raise
    except Exception as e:
        logger.error("Unexpected error while loading data: %s", e)
        raise


def save_data(train_data: pd.DataFrame, test_data: pd.DataFrame, data_path: str) -> None:
    """Save the train and test datasets."""
    try:
        raw_data_path = os.path.join(data_path, 'raw')
        os.makedirs(raw_data_path, exist_ok=True)

        train_data.to_csv(os.path.join(raw_data_path, "train.csv"), index=False)
        test_data.to_csv(os.path.join(raw_data_path, "test.csv"), index=False)

        logger.debug("Train and test data saved to %s", raw_data_path)
    except Exception as e:
        logger.error("Unexpected error occurred while saving data: %s", e)
        raise


def main():
    try:
        test_size = 0.2
        logger.info("Starting data ingestion pipeline...")

        # Download the data from KaggleHub
        data_path = download_the_data("alfathterry/bbc-full-text-document-classification")

        # Load the data
        df = load_data(data_path)
        logger.info("Data loaded successfully. Shape: %s", df.shape)

        # Split into train/test
        train_data, test_data = train_test_split(df, test_size=test_size, random_state=42)
        logger.info("Data split into train and test sets.")

        # Save locally
        save_data(train_data, test_data, "./data")
        logger.info("Data ingestion pipeline completed successfully.")

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()