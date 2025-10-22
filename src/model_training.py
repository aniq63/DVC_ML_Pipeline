import os
from sklearn.ensemble import RandomForestClassifier
import logging
import pandas as pd
import pickle

# =======================
# Logging Setup
# =======================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_training")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_path = os.path.join(log_dir, "model_training.log")
file_handler = logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Avoid duplicate handlers during re-runs
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
else:
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)

# =======================
# Model Function
# =======================
def apply_RF_Model(X_train, y_train):
    """Train a Random Forest model."""
    logger.info("Initializing Random Forest model...")
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=None,
        random_state=42,
        n_jobs=-1
    )
    logger.info("Training Random Forest model...")
    rf_model.fit(X_train, y_train)
    logger.info("Model training completed.")
    return rf_model


# =======================
# Main Function
# =======================
def main():
    try:
        logger.info("Model Training Started")

        # Load encoded features
        logger.info("Loading encoded training data...")
        X_train = pd.read_csv("./data/encoded_data/encoded_train_data.csv")

        # Load labels
        logger.info("Loading label data...")
        y_train = pd.read_csv("./data/pre_process/processed_train.csv")

        # Ensure correct label column
        if 'labels' not in y_train.columns:
            raise KeyError("Column 'labels' not found in processed_train.csv")

        y_train = y_train['labels']

        # Train the model
        model = apply_RF_Model(X_train, y_train)

        # Save the model
        model_dir = "models"
        os.makedirs(model_dir, exist_ok=True)
        model_path = os.path.join(model_dir, "rf_model.pkl")

        logger.info(f"Saving trained model to {model_path}")
        with open(model_path, "wb") as f:
            pickle.dump(model, f)

        logger.info("Model saved successfully!")
        logger.info("Training pipeline completed.")

    except Exception as e:
        logger.error(f"Model training failed: {e}")
        raise


if __name__ == "__main__":
    main()