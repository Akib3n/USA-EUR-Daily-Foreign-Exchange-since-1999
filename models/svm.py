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

# Load the dataset
fexchange = pd.read_csv(preprocessed)

# Convert 'Date' column to datetime format
fexchange['Date'] = pd.to_datetime(fexchange['Date'])

# Create lagged features
fexchange['Lagged_1'] = fexchange.groupby('Country')['Exchange rate'].shift(1)
fexchange['Lagged_2'] = fexchange.groupby('Country')['Exchange rate'].shift(2)

# Moving Average feature
fexchange['MA3'] = fexchange.groupby('Country')['Exchange rate'].transform(lambda x: x.rolling(window=3).mean())

# Target Variable
fexchange['Predicted rate'] = (fexchange['Exchange rate'].shift(-1) > fexchange['Exchange rate']).astype(int)

# Remove Rows with no values from lagged and moving average
fexchange.dropna(inplace=True)

# Feature Engineering
features = ['Lagged_1', 'Lagged_2', 'MA3']
target = 'Predicted rate'

# Train model
X = np.asarray(fexchange[features])
y = np.asarray(fexchange[target])

# Split dataset into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.2, random_state=42)

# Initialize and train SVM model
svm_model = SVC(kernel='linear', gamma='auto')
svm_model.fit(X_train, y_train)

# Evaluate the SVM model
svm_pred = svm_model.predict(X_test)

# Check prediction
print(classification_report(y_test, svm_pred))

# Save SVM model
joblib.dump(svm_model, SVM_MODEL_PATH)