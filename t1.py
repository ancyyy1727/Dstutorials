import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.metrics import mean_squared_error

# Load the dataset
df = pd.read_csv("adv.csv")

# Define independent (input) and dependent (output) variables
X = df[['TV', 'Radio', 'Newspaper']]  # Features affecting sales
y = df['Sales']  # What we want to predict (sales)
X = sm.add_constant(X)  # Add a constant term for the intercept

# Train the Multiple Linear Regression model
model = sm.OLS(y, X).fit()

# Get model evaluation metrics
r_squared = model.rsquared  # R-squared (how well the model fits the data)
rss = sum((model.resid) ** 2)  # Residual Sum of Squares (RSS)
rse = np.sqrt(rss / (len(y) - len(X.columns)))  # Residual Standard Error (RSE)

# Get F-statistic and its p-value (checks if the model is useful)
f_statistic = model.fvalue
p_value = model.f_pvalue

# Print the results
print(f"R² Value: {r_squared:.4f}")
print(f"Residual Standard Error (RSE): {rse:.4f}")
print(f"F-Statistic: {f_statistic:.4f}")
print(f"P-Value (F-Test): {p_value:.4e}")

# Show full model details
print(model.summary())
