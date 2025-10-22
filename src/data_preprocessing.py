import os
import logging
import re
import string
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.stem.porter import PorterStemmer

# Download NLTK resources
nltk.download('stopwords')
nltk.download('wordnet')
nltk.download('omw-1.4')
nltk.download('punkt')

# Logging setup
log_dir = 'logs'
os.makedirs(log_dir, exist_ok=True)

logger = logging.getLogger("data_preprocessing")
logger.setLevel("DEBUG")

console_handler = logging.StreamHandler()
console_handler.setLevel("DEBUG")

file_path = os.path.join(log_dir, "data_preprocessing.log")
file_handler = logging.FileHandler(file_path)
file_handler.setLevel("DEBUG")

formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
console_handler.setFormatter(formatter)
file_handler.setFormatter(formatter)

logger.addHandler(console_handler)
logger.addHandler(file_handler)

# Preprocessing helpers
stop_words = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()
stemmer = PorterStemmer()


def preprocess_text(text):
    """Clean text: lowercase, remove punctuation, URLs, and stopwords."""
    text = text.lower()
    text = re.sub(r"http\S+|www\S+|https\S+", '', text, flags=re.MULTILINE)
    text = text.encode('ascii', 'ignore').decode('ascii')
    text = re.sub(r"#\S+", '', text)
    text = text.translate(str.maketrans('', '', string.punctuation))

    words = text.split()
    words = [word for word in words if word not in stop_words]
    text = ' '.join(words)

    return text


def lemmatize_and_stem(text):
    """Lemmatize and stem a text string."""
    tokens = nltk.word_tokenize(text)
    processed_tokens = []
    for token in tokens:
        lema = lemmatizer.lemmatize(token)
        stem = stemmer.stem(lema)
        processed_tokens.append(stem)
    return " ".join(processed_tokens)


def save_data(processed_train_data: pd.DataFrame, processed_test_data: pd.DataFrame, data_path: str) -> None:
    """Save the preprocessed train and test datasets."""
    try:
        pre_process_data_path = os.path.join(data_path, 'pre_process')
        os.makedirs(pre_process_data_path, exist_ok=True)
        processed_train_data.to_csv(os.path.join(pre_process_data_path, "processed_train.csv"), index=False)
        processed_test_data.to_csv(os.path.join(pre_process_data_path, "processed_test.csv"), index=False)
        logger.debug("Processed train and test data saved to %s", pre_process_data_path)
    except Exception as e:
        logger.error("Unexpected error while saving data: %s", e)
        raise


def main():
    try:
        logger.info("Starting Data Preprocessing pipeline...")
        train_data = pd.read_csv("./data/raw/train.csv")
        test_data = pd.read_csv("./data/raw/test.csv")

        # Auto-detect text column
        text_col = 'text' if 'text' in train_data.columns else 'data'

        # Preprocess and lemmatize
        logger.info("Preprocessing train text...")
        train_data['pre_process_data'] = train_data[text_col].apply(preprocess_text).apply(lemmatize_and_stem)

        
        logger.info("Preprocessing test text...")
        test_data['pre_process_data'] = test_data[text_col].apply(preprocess_text).apply(lemmatize_and_stem)

        # Save processed data
        save_data(train_data, test_data, "./data")
        logger.info("Data Preprocessing pipeline completed successfully.")
        logger.info("Processed train shape: %s, test shape: %s", train_data.shape, test_data.shape)

    except Exception as e:
        logger.error("Pipeline failed: %s", e)
        raise


if __name__ == "__main__":
    main()