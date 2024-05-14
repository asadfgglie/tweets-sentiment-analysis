import csv
import copy
import pandas as pd
from typing import Union
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.neighbors import KNeighborsClassifier  # Importing KNeighborsClassifier
from sklearn.metrics import confusion_matrix, accuracy_score
import random  
import seaborn as sns  
import matplotlib.pyplot as plt

# Load dataset
_dataset: dict[str, list] = {
    'target': [],
    'text': []
}

def load_dataset(path: str='./dataset/training.1600000.processed.noemoticon.csv', return_dict: bool=False, sample_size: int=1000) -> Union[dict, pd.DataFrame]:
    d = copy.deepcopy(_dataset)

    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        reader = csv.reader(f)
        next(reader)  # Read header row
        sampled_data = random.sample(list(reader), sample_size)  # Randomly sample 'sample_size' rows
        for target, _, _, _, _, text in sampled_data:
            text = re.sub(r'@\w+\s?', '', text)  # Remove @username
            text = re.sub(r'http\S+', '', text)  # Remove URLs
            
            d['target'].append(int(target))
            d['text'].append(text if len(text) else None)

    if return_dict:
        return d
    else:
        return pd.DataFrame(d)

# Load dataset and filter out rows with empty text
df = load_dataset(sample_size=1000)
df = df[df['text'].notna()]

"Divide the data into training set and test set (70% training set, 30% test set)"
X_train, X_test, y_train, y_test = train_test_split(df['text'], df['target'], test_size=0.3, random_state=42)

"Then divide the test set into a test set and a verification set (20% test set, 10% verification set)"
X_test, X_val, y_test, y_val = train_test_split(X_test, y_test, test_size=0.33, random_state=42)

# TF-IDF vectorization
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X_train_tfidf = tfidf_vectorizer.fit_transform(X_train)
X_test_tfidf = tfidf_vectorizer.transform(X_test)
'-----------------------------------------------------------------------------'
print(X_train_tfidf.shape)
# Build KNN model
knn_classifier = KNeighborsClassifier(n_neighbors=5)  # You can adjust the number of neighbors

# Train model
knn_classifier.fit(X_train_tfidf, y_train)
print('Train Accuracy:', knn_classifier.score(X_train_tfidf, y_train))

# Predict
y_pred = knn_classifier.predict(X_test_tfidf)

# Evaluate model
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

# Visualiz
sns.heatmap(conf_matrix, annot=True, fmt=".2f")
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()




