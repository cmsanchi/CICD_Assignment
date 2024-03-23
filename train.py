import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.metrics import accuracy_score
import pickle

# Load the dataset
df = pd.read_csv("data/train.csv")

# Split the data into features and target variable
X = df.drop(columns=['Disease'])
y = df['Disease']

# Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Define preprocessing steps (scaling)
preprocessor = StandardScaler()

# Define the logistic regression model
model = LogisticRegression(random_state=42)

# Create a pipeline with preprocessing and modeling steps
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('model', model)])

# Train the model
pipeline.fit(X_train, y_train)

# Evaluate the model
train_accuracy = accuracy_score(y_train, pipeline.predict(X_train))
test_accuracy = accuracy_score(y_test, pipeline.predict(X_test))

print("Train Accuracy:", train_accuracy)
print("Test Accuracy:", test_accuracy)

# Save the trained model to a file using pickle
with open("model.pkl", 'wb') as f:
    pickle.dump(pipeline, f)

