import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
import pickle

print("Starting model training process...")

# 1. Load the dataset
try:
    df = pd.read_csv('deceptive-opinion.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: 'deceptive-opinion.csv' not found. Make sure it's in the 'data/' directory.")
    exit()

# 2. Preprocessing and Feature Extraction
# Selecting the required features
df1 = df[['deceptive', 'text']]

# Mapping labels to numerical values: deceptive -> 0, truthful -> 1
df1.loc[df1['deceptive'] == 'deceptive', 'deceptive'] = 0
df1.loc[df1['deceptive'] == 'truthful', 'deceptive'] = 1
print("Labels mapped to numerical values.")

# Defining features (X) and target (y)
X = df1['text']
y = np.asarray(df1['deceptive'], dtype=int)

# 3. Splitting the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=109) # 70% training and 30% test
print("Data split into training and testing sets.")

# 4. Converting text reviews to numerical features using CountVectorizer
cv = CountVectorizer()
X_train_vect = cv.fit_transform(X_train)
X_test_vect = cv.transform(X_test)
print("Text data vectorized.")

# 5. Training the Multinomial Naive Bayes model
nb = MultinomialNB()
nb.fit(X_train_vect, y_train)
print("Model training complete.")

# 6. Evaluating the model
train_accuracy = nb.score(X_train_vect, y_train)
test_accuracy = nb.score(X_test_vect, y_test)
print(f"Training Accuracy: {train_accuracy:.4f}")
print(f"Testing Accuracy: {test_accuracy:.4f}")

# 7. Saving the model and the vectorizer
# We need to save BOTH the model and the vectorizer to use them in the app
with open('model.pkl', 'wb') as model_file:
    pickle.dump(nb, model_file)

with open('vectorizer.pkl', 'wb') as vectorizer_file:
    pickle.dump(cv, vectorizer_file)

print("Model and vectorizer have been saved as 'model.pkl' and 'vectorizer.pkl'.")