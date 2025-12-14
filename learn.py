import pandas as pd

import numpy as np

from sklearn.model_selection import train_test_split

from sklearn.ensemble import RandomForestClassifier

from sklearn.metrics import accuracy_score

import pickle

print("Loading data...")

# 1. Load Data

try:

    data = pd.read_csv('gesture_data.csv', header=None)

except FileNotFoundError:

    print("Error: 'gesture_data.csv' not found.")

    exit()

# 2. Separation: Column 0 is Label, the rest are Features

X = data.iloc[:, 1:]  # Features (Coordinates)

y = data.iloc[:, 0]   # Labels (0 or 1 or 2 or 3)

# 3. Split Training and Testing data
#When its split X_train are the co ordinate values of the hand, X-test are these again but are used to evaluate
#the training of the model, the same goes for the y's except the y's are the expected output
#known as the target vector to scikit, this is what the model should output for each given array of data

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 4. Train the Classifier

print("Training model...")

model = RandomForestClassifier(n_estimators=100) #Creates a new class

model.fit(X_train, y_train) # x for the training input y for class labels

# 5. Evaluate Accuracy

y_pred = model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)

print(f"Model Accuracy: {accuracy * 100:.2f}%")

# 6. Save the Model

with open('gesture_model.pkl', 'wb') as f: #opens it for saving in a binary mode as it isnt text data

    pickle.dump(model, f) # saves the model in its current state, eg the trained Random forest object

print("Model saved as 'gesture_model.pkl'")
