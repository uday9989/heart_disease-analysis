#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Importing libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix, roc_auc_score, precision_recall_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline as ImbPipeline
from sklearn.pipeline import Pipeline

# Load the dataset
df = pd.read_csv(r"C:\Users\udayv\OneDrive\Desktop\edurekha project\ml project.csv")

# Display basic info
print(df.info())
print(df.describe())

# Check for missing values
print(df.isnull().sum())

# Target variable distribution
sns.countplot(x='HeartDiseaseorAttack', data=df)
plt.title("Class Distribution")
plt.show()

# Correlation heatmap
plt.figure(figsize=(12, 10))
sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm')
plt.title("Correlation Heatmap")
plt.show()

# Feature engineering: Create HealthyDiet flag
df['HealthyDiet'] = (df['Fruits'] & df['Veggies']).astype(int)

# Feature-target relationship
sns.boxplot(x='HeartDiseaseorAttack', y='BMI', data=df)
plt.title("BMI vs Target")
plt.show()

# Split data into features and target
X = df.drop(['HeartDiseaseorAttack'], axis=1)
y = df['HeartDiseaseorAttack']

# Address class imbalance using SMOTE
oversample = SMOTE()
X_resampled, y_resampled = oversample.fit_resample(X, y)

# Split data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X_resampled, y_resampled, test_size=0.2, random_state=42)

# Preprocessing pipeline
scaler = StandardScaler()

# Experimenting with Logistic Regression and Random Forest
models = {
    "Logistic Regression": LogisticRegression(),
    "Random Forest": RandomForestClassifier()
}

results = {}
for name, model in models.items():
    pipeline = Pipeline([
        ('scaler', scaler),
        ('classifier', model)
    ])
    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)
    
    # Evaluate performance
    print(f"Model: {name}")
    print(classification_report(y_test, y_pred))
    print("Confusion Matrix:")
    print(confusion_matrix(y_test, y_pred))
    results[name] = roc_auc_score(y_test, y_pred)

# Precision-recall curve for the best-performing model
best_model_name = max(results, key=results.get)
best_model = models[best_model_name]
pipeline = Pipeline([
    ('scaler', scaler),
    ('classifier', best_model)
])
pipeline.fit(X_train, y_train)
y_scores = pipeline.predict_proba(X_test)[:, 1]

precision, recall, thresholds = precision_recall_curve(y_test, y_scores)

plt.figure()
plt.plot(recall, precision, marker='.')
plt.title("Precision-Recall Curve")
plt.xlabel("Recall")
plt.ylabel("Precision")
plt.show()

# Feature importance for Random Forest
if isinstance(best_model, RandomForestClassifier):
    feature_importances = best_model.feature_importances_
    sorted_indices = np.argsort(feature_importances)[::-1]
    features = X.columns

    plt.figure(figsize=(10, 6))
    sns.barplot(x=feature_importances[sorted_indices], y=features[sorted_indices])
    plt.title("Feature Importance")
    plt.xlabel("Importance")
    plt.ylabel("Feature")
    plt.show()

# Save the best model
import joblib
joblib.dump(pipeline, "heart_disease_model.pkl")


# In[ ]:





# In[ ]:




