import numpy as np
from src_code import Hubbard1DModel
import os

# Define system parameters
L = 4
U_values = np.linspace(0, 8, 20)  # Generates 20 values between 0 and 8

# Ensure the data folder exists
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Compute double occupancy
results = Hubbard1DModel.compute_double_occupancy(U_values, L=L)

# Save results to a CSV file
filename = f"double_occupancy_results.csv"
file_path = os.path.join(data_folder, filename)
Hubbard1DModel.save_results_to_csv(results, file_path)

print(f"Results saved to: {file_path}")
