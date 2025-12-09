import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import pickle

print("Loading data...")

# 1. Load Data (Header=None because we appended raw numbers)

try:

    data = pd.read_csv('gesture_data.csv', header=None)

except FileNotFoundError:

    print("Error: 'gesture_data.csv' not found. Run the collection script first!")

    exit()

# 2. Separation: Column 0 is Label, the rest are Features

X = data.iloc[:, 1:]  # Features (Coordinates)

y = data.iloc[:, 0]   # Labels (0 or 1 or 2 or 3)

# 3. Split Training and Testing data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Classifier

print("Training model...")

model = RandomForestClassifier(n_estimators=100)

model.fit(X_train, y_train)

# 5. Evaluate Accuracy

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Model

with open('gesture_model.pkl', 'wb') as f:

    pickle.dump(model, f)

print("Success! Model saved as 'gesture_model.pkl'")
