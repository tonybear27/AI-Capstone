import pandas as pd
import numpy as np
from sklearn.model_selection import cross_val_predict, cross_val_score, KFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Embedding, Conv1D, GlobalMaxPooling1D, Dense
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import matplotlib.pyplot as plt
import seaborn as sns

# Load training dataset from CSV file
train_data = pd.read_csv('French.csv')

# Load testing dataset from CSV file
test_data = pd.read_csv('testing_french.csv')

# Combine train and test data to fit tokenizer on full dataset
combined_data = pd.concat([train_data['Title'], test_data['Title']], ignore_index=True)

# Tokenize text data and convert to sequences
tokenizer = Tokenizer()
tokenizer.fit_on_texts(combined_data)
X_train_sequences = tokenizer.texts_to_sequences(train_data['Title'])
X_test_sequences = tokenizer.texts_to_sequences(test_data['Title'])

# Pad sequences to ensure uniform length
max_sequence_length = max([len(seq) for seq in X_train_sequences + X_test_sequences])
X_train_padded = pad_sequences(X_train_sequences, maxlen=max_sequence_length)
X_test_padded = pad_sequences(X_test_sequences, maxlen=max_sequence_length)

# Define model architecture
model = Sequential()
model.add(Embedding(input_dim=len(tokenizer.word_index) + 1, output_dim=100, input_shape=(max_sequence_length,)))
model.add(Conv1D(128, 5, activation='relu'))
model.add(GlobalMaxPooling1D())
model.add(Dense(1, activation='sigmoid'))

# Compile model
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Perform cross-validation on training data
kf = KFold(n_splits=5, shuffle=True, random_state=42)
accuracy_scores = []
precision_scores = []
recall_scores = []
f1_scores = []
auc_scores = []
conf_matrices = []

for train_index, val_index in kf.split(X_train_padded):
    X_train, X_val = X_train_padded[train_index], X_train_padded[val_index]
    y_train, y_val = train_data['Label'].iloc[train_index], train_data['Label'].iloc[val_index]
    
    # Train model
    model.fit(X_train, y_train, epochs=5, batch_size=64, verbose=0)
    
    # Predict Labels for validation data
    y_pred = (model.predict(X_val) > 0.5).astype(int)
    
    # Calculate evaluation metrics
    accuracy_scores.append(accuracy_score(y_val, y_pred))
    precision_scores.append(precision_score(y_val, y_pred))
    recall_scores.append(recall_score(y_val, y_pred))
    f1_scores.append(f1_score(y_val, y_pred))
    auc_scores.append(roc_auc_score(y_val, y_pred))
    conf_matrices.append(confusion_matrix(y_val, y_pred))

# Print average evaluation metrics from cross-validation
print("Average Accuracy:", np.mean(accuracy_scores))
print("Average Precision:", np.mean(precision_scores))
print("Average Recall:", np.mean(recall_scores))
print("Average F1 Score:", np.mean(f1_scores))
print("Average AUROC Score:", np.mean(auc_scores))

# Train model on full training data
history = model.fit(X_train_padded, train_data['Label'], epochs=5, batch_size=64, verbose=0)

# Predict Labels for testing data
y_test_pred = (model.predict(X_test_padded) > 0.5).astype(int)

test_conf_matrix = confusion_matrix(test_data['Label'], y_test_pred)

print("Confusion Matrix:")
print(test_conf_matrix)

plt.figure(figsize=(8, 6))
sns.heatmap(test_conf_matrix, annot=True, cmap='Blues', fmt='g')
plt.xlabel('Predicted labels')
plt.ylabel('True labels')
plt.title('CNN Confusion Matrix Unbiased')
plt.show()