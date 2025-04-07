import numpy as np  # For number calculations
import pandas as pd  # For working with data
from sklearn.model_selection import train_test_split  # To split data into train and test sets
from sklearn.metrics import mean_squared_error  # To check how accurate our model is

# Load the dataset
file_path = "Advertising.csv"  # File path
df = pd.read_csv(file_path)  # Read the CSV file into a table

# Select the features (inputs) and the target (output)
X = df[['TV', 'radio', 'newspaper']].values  # Input features
y = df['sales'].values  # Target (sales)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Setup for training the SVM
learning_rate = 0.001  # How much to change weights in each step
lambda_param = 0.01  # Regularization to prevent overfitting
n_iters = 1000  # Number of training loops
n_samples, n_features = X_train.shape  # Get how many rows and columns we have

# Convert sales into two groups: -1 = low sales, 1 = high sales
y_train_binary = np.where(y_train <= np.median(y_train), -1, 1)

# Start with zero weights and zero bias
w = np.zeros(n_features)
b = 0

# Training loop
for _ in range(n_iters):
    for idx, x_i in enumerate(X_train):
        condition = y_train_binary[idx] * (np.dot(x_i, w) - b) >= 1  # Check if prediction is correct
        if condition:
            w -= learning_rate * (2 * lambda_param * w)  # Only apply regularization
        else:
            w -= learning_rate * (2 * lambda_param * w - np.dot(x_i, y_train_binary[idx]))  # Update weights
            b -= learning_rate * y_train_binary[idx]  # Update bias

# Function to make predictions
def predict(X):
    approx = np.dot(X, w) - b  # Calculate the output
    return np.where(approx >= 0, 1, -1)  # Return 1 or -1

# Predict for the test set
y_pred = predict(X_test)

# Translate predictions back to original scale for MSE:
# If -1 → predict min sales; if 1 → predict max sales
y_pred_values = np.where(y_pred == -1, np.min(y_train), np.max(y_train))

# Measure how close predictions are to real sales values
mse = mean_squared_error(y_pred_values, y_test)
print(f"Mean Squared Error using SVM: {mse}")
