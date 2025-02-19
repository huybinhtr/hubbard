import numpy as np
from src_code import hubbard_1Dmodel
import os

# Define system parameters
L = 4  
U_values = np.linspace(0, 8, 20) 


target_occupancies = [1] # chage here  1/2 and 2/3


data_folder = 'data'
if not os.path.exists(data_folder):
    os.makedirs(data_folder)


for target_occupancy in target_occupancies:

    results = hubbard_1Dmodel.compute_double_occupancy(U_values, L, target_occupancy)
    

    filename = f"double_occupancy_results_{target_occupancy}.csv"
    file_path = os.path.join(data_folder, filename)  
    hubbard_1Dmodel.save_results_to_csv(results, file_path)
    print(f"Results saved to: {file_path}")