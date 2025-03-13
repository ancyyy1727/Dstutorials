import numpy as np

# Get matrix dimensions from the user
r1 = int(input("Enter number of rows for Matrix A: "))
c1 = int(input("Enter number of columns for Matrix A: "))
r2 = int(input("Enter number of rows for Matrix B: "))
c2 = int(input("Enter number of columns for Matrix B: "))

# Check if multiplication is possible
if c1 != r2:
    print("Matrix multiplication not possible! Number of columns in A must match rows in B.")
    exit()

# Initialize matrices
A = []
B = []

# Get Matrix A values from the user
print("Enter values for Matrix A:")
for i in range(r1):
    row = [int(input()) for j in range(c1)]
    A.append(row)

# Get Matrix B values from the user
print("Enter values for Matrix B:")
for i in range(r2):
    row = [int(input()) for j in range(c2)]
    B.append(row)

# Initialize result matrix with zeros
C = [[0] * c2 for _ in range(r1)]

# Perform matrix multiplication manually
for i in range(r1):
    for j in range(c2):
        for k in range(c1):
            C[i][j] += A[i][k] * B[k][j]

# Display manual multiplication result
print("\nManual Multiplication Result:")
for row in C:
    print(row)

# Display NumPy multiplication result
print("\nNumPy Multiplication Result:")
print(np.dot(A, B))
