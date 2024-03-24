import pandas as pd
from sklearn.ensemble import RandomForestClassifier
import pickle
import numpy as np

# Read the dataset
df = pd.read_csv("data/train.csv")

# Shuffle the dataset
df = df.sample(frac=1).reset_index(drop=True)

# Separate features and target variable
X = df.drop(columns=['Disease']).to_numpy()
y = df['Disease'].to_numpy()

# Encode target variable
labels = np.sort(np.unique(y))
y = np.array([np.where(labels == x) for x in y]).flatten()

# Define the Random Forest classifier with specified hyperparameters
rf = RandomForestClassifier(bootstrap=True, random_state=42, max_depth=50, max_features=2,
                                    min_samples_leaf=5, min_samples_split=8, n_estimators=200)

# Fit the classifier to the data
model = rf.fit(X, y)

# Save the trained model to a file using pickle
with open("model.pkl", 'wb') as f:
    pickle.dump(model, f)
