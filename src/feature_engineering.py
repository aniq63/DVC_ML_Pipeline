import os
from sklearn.feature_extraction.text import TfidfVectorizer
import logging
import pandas as pd

# Logging setup
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("feature_engineering")
logger.setLevel(logging.DEBUG)

console_handler = logging.StreamHandler()
console_handler.setLevel(logging.INFO)  # show only INFO in console

file_path = os.path.join(log_dir, "feature_engineering.log")
file_handler = logging.FileHandler(file_path)
file_handler.setLevel(logging.DEBUG)  # full logs in file

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

# Avoid duplicate log handlers on re-runs
if not logger.hasHandlers():
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
else:
    logger.handlers.clear()
    logger.addHandler(console_handler)
    logger.addHandler(file_handler)


def main():
    try:
        logger.info("Transformation of Processed Column Started...")

        max_features = 10000

        # Load the preprocessed data
        logger.info("Loading preprocessed CSV files...")
        x_train = pd.read_csv("./data/pre_process/processed_train.csv")
        x_test = pd.read_csv("./data/pre_process/processed_test.csv")

        # Check if column exists
        if 'pre_process_data' not in x_train.columns:
            raise KeyError("Column 'pre_process_data' not found in train data.")
        if 'pre_process_data' not in x_test.columns:
            raise KeyError("Column 'pre_process_data' not found in test data.")

        # TF-IDF Transformation
        logger.info("Applying TF-IDF transformation...")
        tfidf_vectorizer = TfidfVectorizer(max_features=max_features)

        X_train_tfidf = tfidf_vectorizer.fit_transform(x_train['pre_process_data'])
        X_test_tfidf = tfidf_vectorizer.transform(x_test['pre_process_data'])

        # Save the encoded data
        encoded_data_path = os.path.join("data", "encoded_data")
        os.makedirs(encoded_data_path, exist_ok=True)

        logger.info("Saving encoded TF-IDF data as CSV...")
        # Convert sparse matrix to DataFrame before saving
        train_tfidf_df = pd.DataFrame(X_train_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())
        test_tfidf_df = pd.DataFrame(X_test_tfidf.toarray(), columns=tfidf_vectorizer.get_feature_names_out())

        train_tfidf_df.to_csv(os.path.join(encoded_data_path, "encoded_train_data.csv"), index=False)
        test_tfidf_df.to_csv(os.path.join(encoded_data_path, "encoded_test_data.csv"), index=False)

        logger.info("Transformation completed successfully!")

    except Exception as e:
        logger.error("Transformation failed: %s", e)
        raise


if __name__ == "__main__":
    main()
