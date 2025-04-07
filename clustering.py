import numpy as np  # For number calculations
import pandas as pd  # For handling data tables
from sklearn.model_selection import train_test_split  # To split data (not used here but often helpful)
from sklearn.metrics import mean_squared_error  # To measure error (not used in this example)
from scipy.cluster.hierarchy import linkage, dendrogram  # For hierarchical clustering
import matplotlib.pyplot as plt  # For drawing charts

# Load the dataset
file_path = "Advertising.csv"  # Path to the data file
df = pd.read_csv(file_path)  # Read the data into a table

# Choose the features (columns) we want to use for clustering
X = df[['TV', 'radio', 'newspaper']].values  # Turn selected columns into a NumPy array

# Create the linkage matrix using 'complete' method (furthest distance)
linkage_matrix = linkage(X, method='complete')

# Draw the dendrogram (tree-like diagram)
plt.figure(figsize=(10, 5))  # Set the size of the plot
plt.title("Hierarchical Clustering Dendrogram (Complete Linkage)")  # Title of the plot
plt.xlabel("Data Points")  # Label for x-axis
plt.ylabel("Distance")  # Label for y-axis
dendrogram(linkage_matrix)  # Create and plot the dendrogram
plt.show()  # Show the plot
