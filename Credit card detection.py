import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score
from sklearn.preprocessing import StandardScaler
import urllib.request
import os

def download_data(data_url, filename):
    """Downloads data from a URL if it doesn't exist locally."""
    if not os.path.exists(filename):
        try:
            urllib.request.urlretrieve(data_url, filename)
        except Exception as e:
            print(f"Error downloading or reading data from URL: {e}")
            return False
    return True

def load_and_preprocess_data(data_path):
    """Loads and preprocesses the data."""
    data = pd.read_csv(data_path)
    scaler = StandardScaler()
    data['Amount'] = scaler.fit_transform(data['Amount'].values.reshape(-1, 1))
    data['Time'] = scaler.fit_transform(data['Time'].values.reshape(-1, 1))
    data.drop_duplicates(inplace=True)
    return data, scaler

def split_data(data):
    """Splits the data into features (X) and target (y), then into training and test sets."""
    X = data.drop('Class', axis=1)
    y = data['Class']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    return X_train, X_test, y_train, y_test

def train_model(X_train, y_train):
    """Trains a RandomForestClassifier model."""
    model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    model.fit(X_train, y_train)
    return model

def evaluate_model(model, X_test, y_test):
    """Evaluates the model using classification report and accuracy score."""
    y_pred = model.predict(X_test)
    print("Classification Report:\n", classification_report(y_test, y_pred))
    print("Accuracy:", accuracy_score(y_test, y_pred))

def predict_new_transaction(model, scaler, new_transaction_data):
    """Predicts whether a new transaction is fraud or not."""
    new_transaction_df = pd.DataFrame([new_transaction_data])
    new_transaction_df['Amount'] = scaler.transform(new_transaction_df['Amount'].values.reshape(-1, 1))
    new_transaction_df['Time'] = scaler.transform(new_transaction_df['Time'].values.reshape(-1, 1))
    prediction = model.predict(new_transaction_df)
    return "Fraud" if prediction[0] == 1 else "Not Fraud"

def credit_card_fraud_detection(data_path_or_url, new_transaction_data=None):
    """Implements a credit card fraud detection system, prints evaluation, and predicts.
    Args:
        data_path_or_url (str): Path to local CSV or URL of the credit card data.
        new_transaction_data (dict, optional): Dictionary for a new transaction. Defaults to None.
    Returns:
        str or None: "Fraud"/"Not Fraud" if new_transaction_data, else None.
    """
    try:
        if data_path_or_url.startswith('http'):
            filename = "creditcard.csv"
            if not download_data(data_path_or_url, filename):
                return None
            data_path = filename
        else:
            data_path = data_path_or_url

        # Load and preprocess data
        data, scaler = load_and_preprocess_data(data_path)

        # Split data
        X_train, X_test, y_train, y_test = split_data(data)

        # Train model
        model = train_model(X_train, y_train)

        # Evaluate model
        evaluate_model(model, X_test, y_test)

        # Predict new transaction if data is provided
        if new_transaction_data:
            return predict_new_transaction(model, scaler, new_transaction_data)
        else:
            return None
    except FileNotFoundError:
        print(f"Error: File not found at {data_path_or_url}")
        return None
    except Exception as e:
        print(f"An error occurred: {e}")
        return None

if __name__ == "__main__":
    data_url = "https://storage.googleapis.com/download.tensorflow.org/data/creditcard.csv"

    # Example new transaction
    new_transaction = {
        'Time': 1234, 'V1': 1.2, 'V2': -0.5, 'V3': 0.8, 'V4': -1.1, 'V5': 0.9, 'V6': 1.2,
        'V7': -0.7, 'V8': 0.4, 'V9': -0.8, 'V10': 0.1, 'V11': -1.3, 'V12': 0.6,
        'V13': -0.3, 'V14': 0.9, 'V15': -0.2, 'V16': 1.1, 'V17': -0.6, 'V18': 0.3,
        'V19': -0.5, 'V20': 0.1, 'V21': -0.2, 'V22': 0.7, 'V23': -0.1, 'V24': 0.2,
        'V25': -0.3, 'V26': -0.1, 'V27': 0.05, 'V28': -0.02, 'Amount': 50.0
    }

    # Run with new transaction data
    result = credit_card_fraud_detection(data_url, new_transaction)
    if result:
        print(f"Prediction: {result}")

    # Run without new transaction data (only evaluation)
    credit_card_fraud_detection(data_url)
