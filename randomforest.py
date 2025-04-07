import numpy as np  # For numerical operations
import pandas as pd  # For working with data tables
from sklearn.model_selection import train_test_split  # To split data into train and test sets
from sklearn.ensemble import RandomForestRegressor  # For building a Random Forest model
from sklearn.metrics import mean_squared_error  # To measure prediction error

# Load the dataset
file_path = "Advertising.csv"  # Path to the dataset file
df = pd.read_csv(file_path)  # Read the CSV file into a DataFrame

# Select the input features and the target column
X = df[['TV', 'radio', 'newspaper']].values  # Features (inputs)
y = df['sales'].values  # Target variable (what we want to predict)

# Split the data: 80% for training, 20% for testing
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Number of Random Forest models to train
n_estimators = 5

# Store predictions from each model
predictions = []

# Train multiple models with different seeds
for i in range(n_estimators):
    model = RandomForestRegressor(n_estimators=10, random_state=i)  # A Random Forest with 10 trees
    model.fit(X_train, y_train)  # Train the model
    preds = model.predict(X_test)  # Predict on test set
    predictions.append(preds)  # Save the predictions

# Average the predictions from all models
final_predictions = np.mean(predictions, axis=0)

# Calculate the Mean Squared Error
mse = mean_squared_error(y_test, final_predictions)

# Display the error
print(f"Mean Squared Error: {mse}")
