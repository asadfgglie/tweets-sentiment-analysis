import csv
import copy
import pandas as pd
from typing import Union
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix
import random  
import seaborn as sns  
import matplotlib.pyplot as plt
import numpy as np  
'-----------------------------------------------------------------------------'
#Load the dataset
_dataset: dict[str, list] = {
    'target': [],
    'ids': [],
    'date': [],
    'user': [],
    'text': []
}

def load_dataset(path: str='./training.1600000.processed.noemoticon.csv', return_dict: bool=False, sample_size: int=1000) -> Union[dict, pd.DataFrame]:
    d = copy.deepcopy(_dataset)

    with open(path, 'r', encoding='utf8', errors='ignore') as f:
        reader = csv.reader(f)
        # Read the header line
        next(reader)
        # Randomly extract sample_size pieces of data
        sampled_data = random.sample(list(reader), sample_size)
        for target, ids, date, flag, user, text in sampled_data:
            # Use regular expressions to remove the @ symbol and the user name that follows it from the text
            text = re.sub(r'@\w+\s?', '', text)
            # Remove links from text
            text = re.sub(r'http\S+', '', text)
            
            d['target'].append(int(target))
            d['ids'].append(int(ids))
            d['date'].append(date if len(date) else None)
            d['user'].append(user if len(user) else None)
            d['text'].append(text if len(text) else None)

    if return_dict:
        return d
    else:
        return pd.DataFrame(d)

# Load the dataset and filter out empty text lines
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
'------------------------------------------------------------------------------'
# Build a random forest model
rf_classifier = RandomForestClassifier(n_estimators=100, max_depth=10, min_samples_split=2, random_state=42)

#Train model
rf_classifier.fit(X_train_tfidf, y_train)

# predict
y_pred = rf_classifier.predict(X_test_tfidf)

# Evaluate the model
test_accuracy = accuracy_score(y_test, y_pred)
print("Test Accuracy:", test_accuracy)

# Confusion matrix
conf_matrix = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(conf_matrix)

#Convert the numbers in the confusion matrix to probabilities
conf_matrix = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis]

# Use Seaborn to draw a heat map of the confusion matrix
sns.heatmap(conf_matrix, annot=True, fmt=".2f")  # Set the number format to two decimal places
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.show()





