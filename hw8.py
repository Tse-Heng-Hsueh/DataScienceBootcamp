# Import necessary libraries
import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    roc_curve,
    auc,
    silhouette_score
)
import matplotlib.pyplot as plt

# === Logistic Regression Analysis ===

# 1. Load your dataset
# Assuming X_train, X_test, y_train, y_test are pre-split datasets
# Replace with your actual data loading process
X_train, X_test, y_train, y_test = np.random.rand(100, 5), np.random.rand(50, 5), np.random.randint(2, size=100), np.random.randint(2, size=50)

# Define numerical and categorical columns for preprocessing
numerical_columns = [0, 1, 2]  # Replace with actual numerical column indices
categorical_columns = [3, 4]   # Replace with actual categorical column indices

# 2. Preprocess data (scaling + encoding) and fit Logistic Regression
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), numerical_columns),
        ('cat', OneHotEncoder(), categorical_columns)
    ]
)

# Create pipeline with Logistic Regression
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('classifier', LogisticRegression())])

# Fit the model
pipeline.fit(X_train, y_train)

# Predict probabilities
probabilities = pipeline.predict_proba(X_test)[:, 1]

# 3. Experiment with different thresholds
thresholds = [0.3, 0.5, 0.7]  # Define custom thresholds
for t in thresholds:
    predictions = (probabilities >= t).astype(int)
    print(f"Threshold: {t}")
    print(f"Accuracy: {accuracy_score(y_test, predictions)}")
    print(f"Precision: {precision_score(y_test, predictions)}")
    print(f"Recall: {recall_score(y_test, predictions)}\n")

# 4. Plot ROC Curve
fpr, tpr, _ = roc_curve(y_test, probabilities)
roc_auc = auc(fpr, tpr)

plt.plot(fpr, tpr, label=f'ROC Curve (AUC = {roc_auc:.2f})')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.legend()
plt.title('ROC Curve')
plt.show()

# === Clustering Analysis ===

# 1. Load dataset and preprocess (Scaling features)
# Replace with your actual dataset
X = np.random.rand(100, 5)  # Replace with actual data
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 2. Perform KMeans clustering for different values of k
k_values = range(2, 11)  # Test k from 2 to 10
inertia = []  # To store inertia for each k
silhouette_scores = []  # To store silhouette scores for each k

for k in k_values:
    kmeans = KMeans(n_clusters=k, random_state=0)
    labels = kmeans.fit_predict(X_scaled)
    inertia.append(kmeans.inertia_)
    silhouette_scores.append(silhouette_score(X_scaled, labels))

# 3. Plot Inertia and Silhouette Score vs. k
plt.figure(figsize=(10, 5))

# Inertia plot
plt.subplot(1, 2, 1)
plt.plot(k_values, inertia, marker='o', label='Inertia')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Inertia')
plt.title('Elbow Method (Inertia)')
plt.legend()

# Silhouette Score plot
plt.subplot(1, 2, 2)
plt.plot(k_values, silhouette_scores, marker='o', label='Silhouette Score')
plt.xlabel('Number of Clusters (k)')
plt.ylabel('Silhouette Score')
plt.title('Silhouette Analysis')
plt.legend()

plt.tight_layout()
plt.show()

# 4. Discussion:
# - If features are not scaled, inertia and silhouette scores may become less reliable due to different feature magnitudes.
# - There is no single "right" k. Use the Elbow Method or Silhouette Score to find a reasonable k value based on your dataset and goals.
