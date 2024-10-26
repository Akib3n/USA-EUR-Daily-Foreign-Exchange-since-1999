import os
import joblib
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import classification_report

# Constants
preprocessed = 'datasets/preprocessed.csv'
SVM_MODEL_DIR = 'models'
SVM_MODEL_PATH = os.path.join(SVM_MODEL_DIR, 'svm.pkl')

# Function for testing and training selected country
def train_and_test_svm(selected_country):
    # Load the dataset
    fexchange = pd.read_csv(preprocessed)

    # Convert 'Date' column to datetime format
    fexchange['Date'] = pd.to_datetime(fexchange['Date'])

    # Filter data for the selected country
    fexchange = fexchange[fexchange['Country'] == selected_country]

    # Create lagged features directly from daily data
    fexchange['Lagged_1'] = fexchange.groupby('Country')['Exchange rate'].shift(1)
    fexchange['Lagged_2'] = fexchange.groupby('Country')['Exchange rate'].shift(2)
    fexchange['Lagged_3'] = fexchange.groupby('Country')['Exchange rate'].shift(3)
    fexchange['Lagged_4'] = fexchange.groupby('Country')['Exchange rate'].shift(4)
    fexchange['Lagged_5'] = fexchange.groupby('Country')['Exchange rate'].shift(5)
    fexchange['Lagged_6'] = fexchange.groupby('Country')['Exchange rate'].shift(6)

    # Moving Average feature (10-day moving average)
    fexchange['MA10'] = fexchange.groupby('Country')['Exchange rate'].transform(lambda x: x.rolling(window=10).mean())

    # Predicted Rate
    fexchange['Predicted rate'] = (fexchange['Exchange rate'].shift(-1) > fexchange['Exchange rate']).astype(int)

    # Remove rows with NaN values from adding lagged and moving average
    fexchange.dropna(inplace=True)

    print(f"Data count for {selected_country}:")
    print(fexchange.count())

    # Feature Engineering
    features = ['Lagged_1', 'Lagged_2', 'Lagged_3', 'Lagged_4', 'Lagged_5', 'Lagged_6','MA10']
    target = 'Predicted rate'

    # Train model
    X = np.asarray(fexchange[features])
    y = np.asarray(fexchange[target])

    # Split dataset into train and test sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Initialize and train SVM model
    svm_model = SVC(kernel='linear', gamma='auto', class_weight='balanced')
    svm_model.fit(X_train, y_train)

    # Evaluate the SVM model
    svm_pred = svm_model.predict(X_test)

    # Check prediction
    print(classification_report(y_test, svm_pred))

    # Ensure the models directory exists
    if not os.path.exists(SVM_MODEL_DIR):
        os.makedirs(SVM_MODEL_DIR)

    # Save SVM model
    joblib.dump(svm_model, SVM_MODEL_PATH)

# Select Country to train and test
train_and_test_svm(selected_country='Malaysia')  

