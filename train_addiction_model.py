import pandas as pd
import matplotlib.pyplot as plt
import pickle
from sklearn.svm import SVC
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import accuracy_score
from imblearn.over_sampling import SMOTE
from collections import Counter
import os

# Step 1: Load the cleaned and labeled dataset
data = pd.read_csv("/Users/rubaka/idaourproject/Processed_Survey_Data_with_Addiction_Levels-3.csv")

# Step 2: Separate features (X) and target label (y)
X = data.drop(['Total Score', 'Addiction_Level'], axis=1)
y = data['Addiction_Level']

# Step 3: Display original class distribution
order = ['No Addiction', 'Low Addiction', 'Medium Addiction', 'High Addiction']
print("Original class distribution:", Counter(y))
y.value_counts().reindex(order).plot(kind='bar', title="Original Class Distribution", xlabel="Addiction Level", ylabel="Count")
plt.tight_layout()
plt.show()

# Step 4: Handle class imbalance using SMOTE
smote = SMOTE(random_state=42)
X_balanced, y_balanced = smote.fit_resample(X, y)

# Step 5: Display class distribution after SMOTE
count_after_smote = pd.Series(Counter(y_balanced)).reindex(order)
print("Balanced class distribution:", Counter(y_balanced))
count_after_smote.plot(kind='bar', title="Class Distribution after SMOTE", xlabel="Addiction Level", ylabel="Count")
plt.tight_layout()
plt.show()

# Step 6: Split the data into train and test sets
X_train, X_test, y_train, y_test = train_test_split(X_balanced, y_balanced, test_size=0.2, random_state=42)

# Step 7: Display class distribution of training set
count_train = pd.Series(Counter(y_train)).reindex(order)
print("Train set class distribution:", Counter(y_train))
count_train.plot(kind='bar', title="Class Distribution of Training Set", xlabel="Addiction Level", ylabel="Count")
plt.tight_layout()
plt.show()

# Step 8: Train the model with SVM
svm_model = SVC()
cv_scores = cross_val_score(svm_model, X_train, y_train, cv=10)
print(f"Cross-validation Accuracy: {cv_scores.mean() * 100:.2f}%")

svm_model.fit(X_train, y_train)
y_pred = svm_model.predict(X_test)
test_accuracy = accuracy_score(y_test, y_pred)
print(f"Test Accuracy: {test_accuracy * 100:.2f}%")

# Step 9: Save the trained model
model_filename = "/Users/rubaka/idaourproject/addiction_model_pkl"
with open(model_filename, "wb") as file:
    pickle.dump(svm_model, file)

# Step 10: Confirm the model was saved
print(f" Model training complete and saved as '{model_filename}'")
print("Model exists on disk:", os.path.exists(model_filename))