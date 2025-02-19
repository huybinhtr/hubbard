import numpy as np
from src_code import hubbard_1Dmodel
import os

# Define system parameters
L = 4  # System size
U_values = np.linspace(0, 8, 20)  # Range of U values

# Define target occupancies
target_occupancies = [1]

# Create the 'data' folder if it doesn't exist
data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)

# Compute double occupancy for each target occupancy and save to CSV
for target_occupancy in target_occupancies:
    # Compute results
    results = hubbard_1Dmodel.compute_double_occupancy(U_values, L, target_occupancy)
    
    # Save results to CSV in the 'data' folder
    filename = f"double_occupancy_results_{target_occupancy}.csv"
    file_path = os.path.join(data_folder, filename)  # Save in the 'data' folder
    hubbard_1Dmodel.save_results_to_csv(results, file_path)
    print(f"Results saved to: {file_path}")