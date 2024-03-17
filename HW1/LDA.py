import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.decomposition import LatentDirichletAllocation
from sklearn.metrics import accuracy_score, classification_report

train_data = pd.read_csv('French.csv')
test_data = pd.read_csv('testing_french.csv')

X_train = train_data['Title']
y_train = train_data['Label']

X_test = test_data['Title']
y_test = test_data['Label']

# Preprocess the text data
vectorizer = CountVectorizer()
X_train_counts = vectorizer.fit_transform(X_train)
X_test_counts = vectorizer.transform(X_test)

# Train the LDA model
lda = LatentDirichletAllocation(n_components=2, random_state=42)
X_train_lda = lda.fit_transform(X_train_counts)
X_test_lda = lda.transform(X_test_counts)

# Evaluate the model
# Predict the labels for the testing data
y_pred = [0 if topic[0] > topic[1] else 1 for topic in X_test_lda]

# Compute accuracy
accuracy = accuracy_score(y_test, y_pred)
print("Accuracy:", accuracy)

# Generate classification report
print("Classification Report:")
print(classification_report(y_test, y_pred))
