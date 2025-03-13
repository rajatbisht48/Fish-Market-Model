import pandas as pd
import numpy as np
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("Fish.csv")

# Encode categorical variable (Species)
label_encoder = LabelEncoder()
data["Species"] = label_encoder.fit_transform(data["Species"])

# Define features and target variable
features = data.drop(columns=["Weight"])
target = data["Weight"]

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, target, test_size=0.2, random_state=42)

# Initialize and train the model
fish_model = RandomForestRegressor(n_estimators=100, random_state=42)
fish_model.fit(X_train, y_train)

# Save the trained model to a file as 'fish_model.pkl'
with open("fish_model.pkl", "wb") as model_file:
    pickle.dump(fish_model, model_file)

print("Model saved as 'fish_model.pkl'")
