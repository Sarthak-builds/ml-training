# Set matplotlib backend to non-interactive 'Agg' to prevent blocking in background execution
import matplotlib
matplotlib.use('Agg')

# Import numerical processing, dataframe management, and plotting libraries
import numpy as np # Numerical calculations
import pandas as pd # Dataframe management
import matplotlib.pyplot as plt # Visualization base
import seaborn as sns # Premium visualizations

# Import sklearn data generation, prep, pipeline, and linear model modules
from sklearn.datasets import make_regression # Synthetic data generator
from sklearn.model_selection import train_test_split # Split dataset
from sklearn.preprocessing import StandardScaler, OneHotEncoder # Scaler and encoder
from sklearn.compose import ColumnTransformer # Join preprocessing steps
from sklearn.pipeline import Pipeline # Sequential execution
from sklearn.linear_model import LinearRegression # Core estimator
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score # Metrics
import joblib # Model serializer

# Step 1: Generate synthetic regression data representing a real-world scenario
X_raw, y_raw = make_regression(n_samples=500, n_features=3, noise=15.0, random_state=42)

# Convert feature matrix to a pandas DataFrame
df = pd.DataFrame(X_raw, columns=['feature_1', 'feature_2', 'feature_3'])

# Add target column to DataFrame
df['target'] = y_raw

# Create synthetic categorical column (e.g. Regions: 'North', 'South', 'West')
np.random.seed(42) # Set seed for reproducibility
regions = np.random.choice(['North', 'South', 'West'], size=500) # Draw random values
df['region'] = regions # Inject categorical column to dataset

# Step 2: Exploratory Data Analysis (EDA)
print("Dataset size:", df.shape) # View rows and columns
print("\nFirst 5 rows:\n", df.head()) # Inspect structure of the data
print("\nDataset general details:") # Section header
df.info() # Check schema and data types
print("\nStatistical description:\n", df.describe()) # Examine statistical metrics
print("\nMissing values check:\n", df.isnull().sum()) # Confirm missing values count

# Step 3: Separate features (X) and target variable (y)
X = df.drop('target', axis=1) # Extract feature columns
y = df['target'] # Extract target column

# Step 4: Preprocessing setup (Scaling and Encoding)
cat_cols = X.select_dtypes(include=['object']).columns # Identify categorical features
num_cols = X.select_dtypes(exclude=['object']).columns # Identify numerical features

# Configure the preprocessing steps for numerical and categorical variables
preprocessor = ColumnTransformer(
    transformers=[
        ('num', StandardScaler(), num_cols), # Standardize numerical features
        ('cat', OneHotEncoder(drop='first', handle_unknown='ignore'), cat_cols) # Encode categorical features
    ]
)

# Step 5: Split the dataset into training (80%) and testing (20%) sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42 # Perform random split
)

# Step 6: Construct and train model pipeline
model = Pipeline(steps=[
    ('preprocessor', preprocessor), # Preprocessing stage
    ('regressor', LinearRegression()) # Estimator stage
])

# Fit model using training data
model.fit(X_train, y_train) # Learn linear coefficients

# Step 7: Predict and evaluate model
y_pred = model.predict(X_test) # Predict targets for testing subset

# Compute test metrics
mae = mean_absolute_error(y_test, y_pred) # Calculate Mean Absolute Error
mse = mean_squared_error(y_test, y_pred) # Calculate Mean Squared Error
rmse = np.sqrt(mse) # Calculate Root Mean Squared Error
r2 = r2_score(y_test, y_pred) # Calculate R-squared Score

# Retrieve size metrics to compute Adjusted R-squared
n = X_test.shape[0] # Test set size
p = X_test.shape[1] # Number of features

# Calculate Adjusted R-squared Score
adj_r2 = 1 - (1 - r2) * (n - 1) / (n - p - 1)

# Display evaluation results
print(f"\nEvaluation Results:")
print(f"MAE        : {mae:.2f}") # Print Mean Absolute Error
print(f"MSE        : {mse:.2f}") # Print Mean Squared Error
print(f"RMSE       : {rmse:.2f}") # Print Root Mean Squared Error
print(f"R² Score   : {r2:.4f}") # Print R-squared Score
print(f"Adjusted R²: {adj_r2:.4f}") # Print Adjusted R-squared Score

# Save the trained model pipeline to disk
joblib.dump(model, "linear_regression_model.pkl") # Serialize pipeline
print("\nModel saved successfully as 'linear_regression_model.pkl'")
