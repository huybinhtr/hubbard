import numpy as np
from src_code import Householder

# Define input vectors with same norm
X = np.array([1, 2, 2])
Y = np.array([2, -1, 2])

# Compute the Householder matrix
P_matrix = Householder.P(X, Y)
print("P_matrix:")
print(P_matrix)

# Compute P @ X to check if the Householder matrix is correct Y=PX
PX = P_matrix @ X
print("PX:", PX)

# Check if P is Hermitian and Unitary
print("Is P Hermitian?", Householder.hermitian_check(P_matrix))
print("Is P Unitary?", Householder.unitary_check(P_matrix))