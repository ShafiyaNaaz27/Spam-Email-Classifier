import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, classification_report
import joblib

# Load the dataset
df = pd.read_csv("data/sms.tsv.txt", sep='\t', header=None, names=['label','message'])
print(df.head())
print(df.shape)

# Drop any missig value
# df.dropna(inplace=True)

# Convert labels to binary: ham=0 , spam=1
df['label'] = df['label'].map({'ham':0, 'spam':1})

# split the data
x = df['message'].astype(str)
y = df['label']

# split inti train and test sets
x_train,x_test,y_train, y_test = train_test_split(x , y, test_size=0.2, random_state= 42)

# convert text into numerical data
vectorizer = TfidfVectorizer()
X_train_tfidf = vectorizer.fit_transform(x_train)
X_test_tfidf = vectorizer.transform(x_test)

# Train the model

model = MultinomialNB()
model.fit(X_train_tfidf, y_train)

# Evaluate the model
y_pred = model.predict(X_test_tfidf)

print("Accuracy:", accuracy_score(y_test, y_pred))
print("\nClassification Report:\n", classification_report(y_test, y_pred))

import os

os.makedirs("model",exist_ok=True)

# save both model and vectorizer
joblib.dump({
    'model':model,
    'vectorizer':vectorizer
},
    "model/spam_classifier_model.pkl"

)


# save the model
joblib.dump(model, "model/spam_classifier_model.pkl")
joblib.dump(vectorizer, "models/tfidf_vectorizer.pkl")


