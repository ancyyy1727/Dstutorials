import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression

# Sample data: TV budget vs. whether a sale happened (1 = Sale, 0 = No Sale)
data = {
    "TV": [50, 60, 70, 80, 90, 100, 110, 120, 130, 140],
    "Sale_Success": [0, 0, 0, 0, 1, 1, 1, 1, 1, 1]  # Did a sale happen?
}

# Create a DataFrame
df = pd.DataFrame(data)

# Split data into input (TV budget) and output (Sale success)
X = df[["TV"]]  # Feature
y = df["Sale_Success"]  # Target (what we’re predicting)

# Split into training and testing sets (80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a Logistic Regression model
model = LogisticRegression()
model.fit(X_train, y_train)

# Predict if a sale happens for a TV budget of $95
unknown_input = [[95]]  # Example budget
predicted_class = model.predict(unknown_input)

# Show the result
print(f"Predicted Sale Success for TV budget {unknown_input[0][0]}: {'Yes' if predicted_class[0] == 1 else 'No'}")
