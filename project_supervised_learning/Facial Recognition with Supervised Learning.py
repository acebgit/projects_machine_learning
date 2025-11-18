# Import required libraries
import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GridSearchCV, KFold, train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt

# 1) Read the CSV file
df = pd.read_csv("lfw_arnie_nonarnie.csv")
# print(df.head())

# 2) Separate the predictor and class label, and split the data into training and testing sets using stratify to balance the class
X = df.drop('Label', axis=1)
y = df['Label']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=21, stratify=y)
# [print(i.shape) for i in (df, X, y, X_train, X_test, y_train, y_test)]

# 1) Construct machine learning pipelines for three classification models. Store these initialized models in a dictionary named models
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

models = {
    "LogisticRegression": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', LogisticRegression(random_state=21, max_iter=1000))
    ]),
    "RandomForest": Pipeline([
        ('clf', RandomForestClassifier(random_state=21))
    ]),
    "SVM": Pipeline([
        ('scaler', StandardScaler()),
        ('clf', SVC(probability=True, random_state=21))
    ])
}

# Determine the best performing model based on cross-validation scores.
# Save the model's name as best_model_name, its parameters as best_model_info, and its cross-validation score as best_model_cv_score
best_model_name = None
best_model_info = None
best_model_cv_score = 0

for name, model in models.items():
    cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
    mean_score = cv_scores.mean()
    print(f"{name} CV Accuracy: {mean_score:.4f}")

    if mean_score > best_model_cv_score:
        best_model_cv_score = mean_score
        best_model_name = name
        best_model_info = model.get_params()
print("-----------")

# Fit the best model on the full training set
best_model = models[best_model_name]
best_model.fit(X_train, y_train)

# 3) Evaluate the selected model and store accuracy, precision, recall, and f1 on the test set.
y_pred = best_model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)

# 4) Save the best score. Aim to achieve a minimum accuracy of 80% for at least one of the models. Save your best accuracy score as score.
score = accuracy

# Print results
print(f"Best Model: {best_model_name}")
print(f"Best Model Parameters: {best_model_info}")
print(f"Best CV Score: {best_model_cv_score:.4f}")
print("Test Set Performance:")
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall: {recall:.4f}")
print(f"F1 Score: {f1:.4f}")

# 5) Use Matplotlib to create a confusion matrix visualization for the predictions made by your best model
cm = confusion_matrix(y_test, y_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.xlabel("Predicted")
plt.ylabel("Actual")
plt.title(f"{best_model_name} Confusion Matrix")
plt.show()
