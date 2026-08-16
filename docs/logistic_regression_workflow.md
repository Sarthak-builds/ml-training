# End-to-End Logistic Regression Workflow

This document describes the standard pipeline for a supervised classification problem (predicting a binary or categorical target).

---

### 1. Problem Definition & Data Collection
- Identify the target column to predict (binary/categorical class).
- Load the dataset into a Pandas DataFrame.

```python
import pandas as pd
# Load the dataset from CSV
df = pd.read_csv("your_classification_data.csv")
```

---

### 2. Import Required Libraries
Import all standard numerical processing, visualization, modeling, and evaluation libraries at the top of the file.

```python
import numpy as np # Numerical operations
import pandas as pd # Data manipulation
import matplotlib.pyplot as plt # Core visualization plotting
import seaborn as sns # Advanced visualization styling

# Preprocessing & split libraries
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression

# Classification evaluation metric libraries
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    roc_curve,
    auc
)
import joblib # Model serialization
```

---

### 3. Exploratory Data Analysis (EDA)
Inspect the dataset's sizes, distributions, and class balance.

```python
# Check row and column counts
print(df.shape)

# View first 5 sample rows
print(df.head())

# Review column data types and non-null counts
print(df.info())

# Check class balance on the target column
print(df['target_class'].value_counts())

# Check count of missing values per column
print(df.isnull().sum())
```

---

### 4. Data Visualization
Check distributions, correlations, class distributions, and balance.

```python
# Plot bar chart of classes to check balance
sns.countplot(x='target_class', data=df)
plt.show()

# Compute correlation matrix for numeric columns
corr = df.corr(numeric_only=True)

# Generate a correlation heatmap
plt.figure(figsize=(12, 8))
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
plt.show()

# Boxplot of a feature split by target class
sns.boxplot(x='target_class', y='feature1', data=df)
plt.show()
```

---

### 5. Features and Target Separation
Separate features (`X`) and target variable (`y`).

```python
# Drop target class column to extract features
X = df.drop('target_class', axis=1)

# Extract only target class column
y = df['target_class']
```

---

### 6. Preprocessing: Encoding and Scaling
Use `ColumnTransformer` to scale numerical columns and encode categorical columns.

```python
# Get names of categorical columns
cat_cols = X.select_dtypes(include='object').columns

# Get names of numerical columns
num_cols = X.select_dtypes(exclude='object').columns

# Construct the data preprocessing pipeline
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols), # Apply scaling to numeric features
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols) # Apply One-Hot Encoding to categorical features
    ]
)
```

---

### 7. Train-Test Split
Split dataset into training (80%) and testing (20%) sets.

```python
# Split data into training and validation sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y # stratify ensures balanced classes in splits
)
```

---

### 8. Model Training using Pipeline
Train a logistic regression model within a Scikit-Learn Pipeline.

```python
# Construct the end-to-end model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor), # Preprocessing step
    ('classifier', LogisticRegression(max_iter=1000)) # Estimator step with higher iterations
])

# Fit model on training data
model.fit(X_train, y_train)
```

---

### 9. Predictions and Evaluation Metrics
Compute predictions, class probabilities, and evaluate the classifier (Accuracy, Precision, Recall, F1-Score, Confusion Matrix, and ROC-AUC).

```python
# Make class predictions on test set
y_pred = model.predict(X_test)

# Predict probability scores for the positive class (used for ROC-AUC)
y_prob = model.predict_proba(X_test)[:, 1]

# Calculate classification accuracy
accuracy = accuracy_score(y_test, y_pred)

# Calculate precision score
precision = precision_score(y_test, y_pred, average='binary')

# Calculate recall score
recall = recall_score(y_test, y_pred, average='binary')

# Calculate F1 score
f1 = f1_score(y_test, y_pred, average='binary')

# Print metrics
print(f"Accuracy : {accuracy:.4f}")
print(f"Precision: {precision:.4f}")
print(f"Recall   : {recall:.4f}")
print(f"F1 Score : {f1:.4f}")

# Generate and display confusion matrix
cm = confusion_matrix(y_test, y_pred)
print("Confusion Matrix:")
print(cm)

# Display full classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
```

---

### 10. ROC Curve & Model Saving
Visualize the Receiver Operating Characteristic (ROC) curve and save the pipeline.

```python
# Compute ROC curve metrics
fpr, tpr, thresholds = roc_curve(y_test, y_prob)

# Compute Area Under the Curve (AUC)
roc_auc = auc(fpr, tpr)
print(f"ROC-AUC  : {roc_auc:.4f}")

# Plot ROC curve
plt.figure()
plt.plot(fpr, tpr, color='darkorange', label=f'ROC curve (area = {roc_auc:.2f})')
plt.plot([0, 1], [0, 1], color='navy', linestyle='--')
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Receiver Operating Characteristic')
plt.legend(loc="lower right")
plt.show()

# Save the trained model pipeline to disk
joblib.dump(model, "logistic_regression_model.pkl")
```
