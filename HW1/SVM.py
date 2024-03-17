import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, roc_curve, roc_auc_score
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.pipeline import Pipeline
from sklearn.model_selection import cross_val_predict, cross_val_score


train_data = pd.read_csv('French.csv')
test_data = pd.read_csv('testing_french.csv')

X_train = train_data['Title']
y_train = train_data['Label']

X_test = test_data['Title']
y_test = test_data['Label']

# Define a pipeline for SVM classifier
svm_classifier = Pipeline([
    ('tfidf', TfidfVectorizer()),  # Convert text to TF-IDF features
    ('svm', SVC(kernel='linear', probability=True))  # Support Vector Machine classifier
])

# Use predict_proba for probability estimates
y_pred_cv = cross_val_predict(svm_classifier, X_train, y_train, cv=5, method='predict_proba')[:, 1]  

# Calculate evaluation metrics using cross-validation
accuracy_cv = cross_val_score(svm_classifier, X_train, y_train, cv=5, scoring='accuracy').mean()
precision_cv = cross_val_score(svm_classifier, X_train, y_train, cv=5, scoring='precision').mean()
recall_cv = cross_val_score(svm_classifier, X_train, y_train, cv=5, scoring='recall').mean()
f1_cv = cross_val_score(svm_classifier, X_train, y_train, cv=5, scoring='f1').mean()
roc_auc_cv = cross_val_score(svm_classifier, X_train, y_train, cv=5, scoring='roc_auc').mean() 

print("Cross-Validation Metrics:")
print("Accuracy:", accuracy_cv)
print("Precision:", precision_cv)
print("Recall:", recall_cv)
print("F1 Score:", f1_cv)
print('ROU-AUC:', roc_auc_cv)

# Fit the classifier on the full training data
svm_classifier.fit(X_train, y_train)

# Use predict_proba for probability estimates
y_pred_probs = svm_classifier.predict_proba(X_test)[:, 1]  

fpr, tpr, thresholds = roc_curve(y_test, y_pred_probs)
auc_score = roc_auc_score(y_test, y_pred_probs)

conf_matrix = confusion_matrix(y_test, y_pred_probs.round())
print(conf_matrix)

# Plot confusion matrix
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('SVM Confusion Matrix French')
plt.show()
