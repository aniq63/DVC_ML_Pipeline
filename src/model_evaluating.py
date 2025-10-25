import os
import pickle
import logging
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix,precision_score, recall_score
import yaml
from dvclive import Live

# =======================
# Logging Setup
# =======================
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("model_evaluation")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)

file_path = os.path.join(log_dir, "model_evaluation.log")
file_handler = logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Avoid duplicate handlers
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
else:
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


# =======================
# Load the Parameters Function
# =======================

def load_params(params_path: str) -> dict:
    """Load parameters from a YAML file."""
    try:
        with open(params_path, 'r') as file:
            params = yaml.safe_load(file)
        logger.debug('Parameters retrieved from %s', params_path)
        return params
    except FileNotFoundError:
        logger.error('File not found: %s', params_path)
        raise
    except yaml.YAMLError as e:
        logger.error('YAML error: %s', e)
        raise
    except Exception as e:
        logger.error('Unexpected error: %s', e)
        raise

# =======================
# Main Function
# =======================
def main():
    try:
        logger.info("Model Evaluation Started")

        # ------------------------------
        # Load Params
        # ------------------------------
        params = load_params(params_path = "params.yaml")

        # ------------------------------
        # Load model
        # ------------------------------
        model_path = "./models/rf_model.pkl"
        if not os.path.exists(model_path):
            raise FileNotFoundError(" Trained model file not found at './models/rf_model.pkl'")

        logger.info("Loading trained model...")
        with open(model_path, "rb") as f:
            model = pickle.load(f)
        logger.info("odel loaded successfully.")

        # ------------------------------
        # Load test data
        # ------------------------------
        logger.info("Loading encoded test data and labels...")
        X_test = pd.read_csv("./data/encoded_data/encoded_test_data.csv")
        y_test = pd.read_csv("./data/pre_process/processed_test.csv")

        if 'labels' not in y_test.columns:
            raise KeyError("Column 'labels' not found in processed_test.csv")

        y_test = y_test['labels']

        # ------------------------------
        # Predict
        # ------------------------------
        logger.info("Generating predictions...")
        y_pred = model.predict(X_test)

        # ------------------------------
        # Evaluate
        # ------------------------------
        logger.info("Evaluating model performance...")
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred)
        cm = confusion_matrix(y_test, y_pred)

        # Log key metrics
        logger.info(f"Accuracy: {acc:.4f}")
        logger.info("Classification Report:\n" + report)
        logger.info("Confusion Matrix:\n" + str(cm))

        # ------------------------------
        # Save evaluation results
        # ------------------------------
        results_dir = "evaluation_results"
        os.makedirs(results_dir, exist_ok=True)

        results_path = os.path.join(results_dir, "evaluation_results.txt")
        with open(results_path, "w") as f:
            f.write("=== Model Evaluation Results ===\n\n")
            f.write(f"Accuracy: {acc:.4f}\n\n")
            f.write("Classification Report:\n")
            f.write(report + "\n")
            f.write("Confusion Matrix:\n")
            f.write(str(cm) + "\n")

        logger.info(f"Evaluation results saved to {results_path}")
        logger.info("Model Evaluation Completed Successfully.")

        with Live(save_dvc_exp=True) as live:
         live.log_metric('accuracy', acc)
         live.log_metric('precision', precision_score(y_test,y_pred,average='weighted'))
         live.log_metric('recall', recall_score(y_test,y_pred, average='weighted'))

         live.log_params(params)

    except Exception as e:
        logger.error(f"Model evaluation failed: {e}")
        raise


if __name__ == "__main__":
    main()