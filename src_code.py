import numpy as np
import matplotlib.pyplot as plt
from scipy.linalg import eigh
from qat.core import Term
from qat.fermion.hamiltonians import FermionHamiltonian, make_hubbard_model
import csv
import os

t = 1  # Hopping amplitude

class Householder:
    @staticmethod
    def P(x, y):
        x, y = np.asarray(x, dtype=np.float64), np.asarray(y, dtype=np.float64)
        alpha = np.linalg.norm(x) / np.linalg.norm(y)
        v = x - alpha * y
        v /= np.linalg.norm(v) if np.linalg.norm(v) != 0 else 1
        return np.identity(v.size) - 2 * np.outer(v, v)

    @staticmethod
    def hermitian_check(mat):
        return np.allclose(mat, mat.T)

    @staticmethod
    def unitary_check(mat):
        return np.allclose(np.eye(mat.shape[0]), mat @ mat.T)

class Hubbard1DModel:
    @staticmethod
    def t_mat_1D(L: int, pbc: bool = False) -> np.ndarray:
        t_mat = np.zeros((L, L))
        for i in range(L - 1):
            t_mat[i, i + 1] = -1
            t_mat[i + 1, i] = -1
        if pbc:
            t_mat[0, L - 1] = -1
            t_mat[L - 1, 0] = -1
        return t_mat

    @staticmethod
    def number_operator(L, index):
        return FermionHamiltonian(2 * L, [Term(1.0, "Cc", [index, index])])

    @staticmethod
    def total_number_operator(L):
        """Construct the total number operator for all sites."""
        nqbits = 2 * L
        terms = []
        for i in range(L):
            # Spin-up and spin-down number operators for site i
            terms.append(Term(1.0, "Cc", [2 * i, 2 * i]))  
            terms.append(Term(1.0, "Cc", [2 * i + 1, 2 * i + 1]))  
        return FermionHamiltonian(nqbits, terms)

    @staticmethod
    def double_occupancy_operator(L):
        nqbits = 2 * L
        terms = []
        for i in range(L):
            up_idx = 2 * i
            down_idx = 2 * i + 1
            terms.append(Term(1.0, "CcCc", [up_idx, up_idx, down_idx, down_idx]))
        return FermionHamiltonian(nqbits, terms)

    @staticmethod
    def ground_state_wf(hamiltonian):
        h_matrix = hamiltonian.get_matrix()
        eigvals, eigvecs = eigh(h_matrix)
        return eigvecs[:, 0]  # Ground state wavefunction

    @staticmethod
    def compute_electron_number(ground_state, L):
        """Compute the total electron number using the total number operator."""
        N_op = Hubbard1DModel.total_number_operator(L)
        N_matrix = N_op.get_matrix()
        n_total = np.real(np.vdot(ground_state, N_matrix @ ground_state))
        return n_total

    @staticmethod
    def compute_double_occupancy(U_values, L):
        results = []
        for U in U_values:
            mu = U / 2  # Fixed chemical potential
            hubbard_hamiltonian = make_hubbard_model(Hubbard1DModel.t_mat_1D(L, pbc=True), U, mu)

            ground_state = Hubbard1DModel.ground_state_wf(hubbard_hamiltonian)
            electron_density = Hubbard1DModel.compute_electron_number(ground_state, L) / L

            if not np.isclose(electron_density, 1.0, atol=1e-6):
                print(f"Skipping U/t = {U/t} due to electron density mismatch: {electron_density}")
                continue

            D_op = Hubbard1DModel.double_occupancy_operator(L)
            D_matrix = D_op.get_matrix()
            double_occupancy = np.real(np.vdot(ground_state, D_matrix @ ground_state)) / L

            results.append((U, double_occupancy))

        return results

    @staticmethod
    def save_results_to_csv(results, filename='double_occupancy_results.csv'):
        save_path = os.path.join(os.getcwd(), filename)  # Save in the current working directory
        with open(save_path, 'w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(["U/t", "Double Occupancy"])
            for U, double_occupancy in results:
                writer.writerow([U / t, double_occupancy])
        print(f"Results saved to: {save_path}")