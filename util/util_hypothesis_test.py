from PIL import Image
import requests
import torch
from transformers import CLIPProcessor, CLIPModel
from torchvision.datasets import ImageFolder
from torchvision import datasets, transforms
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
import random
import open_clip
from joblib import Parallel, delayed

from util import *
from scipy.stats import kurtosis
import numpy as np

# Define custom dataset for white noise images
class WhiteNoiseDataset(Dataset):
    def __init__(self, num_images, height, width, channels):
        self.num_images = num_images
        self.height = height
        self.width = width
        self.channels = channels
    
    def __len__(self):
        return self.num_images
    
    def __getitem__(self, idx):
        # Generate a single white noise image
        white_noise_image = np.random.rand(self.height, self.width, self.channels) * 255
        return white_noise_image.astype(np.uint8)  # Return as uint8 to simulate image
    

def random_rotation_matrix(n=3):
    A = np.random.randn(n, n)
    Q, R = np.linalg.qr(A)
    # Ensure a positive determinant
    if np.linalg.det(Q) < 0:
        Q[:, -1] = -Q[:, -1]
    return Q

# def rotate_matrix(Z):
#     n_rows, n_cols = Z.shape

#     # Generate a batch of random rotation matrices
#     rotation_matrices = np.array([random_rotation_matrix(n_cols) for _ in range(n_rows)])

#     # Efficiently apply the rotation to each row of Z using batch matrix multiplication
#     rotated_matrix = np.einsum('ijk,ik->ij', rotation_matrices, Z)
#     return rotated_matrix


def rotate_matrix(Z):
    n_rows, n_cols = Z.shape

    # Generate a random rotation matrix for each row in parallel
    rotation_matrices = Parallel(n_jobs=-1)(
        delayed(random_rotation_matrix)(n_cols) for _ in range(n_rows)
    )

    # Convert the list of rotation matrices into a NumPy array
    rotation_matrices = np.array(rotation_matrices)

    # Efficiently apply the rotation to each row of Z using batch matrix multiplication
    rotated_matrix = np.einsum('ijk,ik->ij', rotation_matrices, Z)
    
    return rotated_matrix

def compute_kurtosis_stat(Z):
    return kurtosis(Z, axis = 0)

# def compute_varimax_stat(Z):
#     n, k = Z.shape  # AssZming Z is n x k and R is k x k
#     total_sum = 0
#     for ell in range(k):
#         # Compute the first term: Sum of the fourth powers of elements in column `ell` of UR
#         first_term = np.sum(Z[:, ell]**4)
#         # Compute the second term: (Sum of squares of the elements in column `ell` of U)^2
#         second_term = (np.sum(Z[:, ell]**2)**2) / n
#         # Combine the terms and add to total sum
#         total_sum += (first_term - second_term) / n
#     return total_sum

def compute_varimax_stat(Z):
    n, k = Z.shape
    first_term = np.sum(Z**4, axis=0)
    second_term = (np.sum(Z**2, axis=0)**2) / n
    total_sum = np.sum((first_term - second_term) / n)
    return total_sum

def rotate_matrix_cuda(Z, chunk_size=5000):
    Z = Z.to('cuda')
    n_rows, n_cols = Z.shape

    rotated_matrix = torch.zeros_like(Z, device='cuda')

    for i in range(0, n_rows, chunk_size):
        # Process in chunks
        chunk = Z[i:i+chunk_size]

        # Generate random rotation matrices for the chunk
        rotation_matrices = torch.stack([random_rotation_matrix_cuda(n_cols).to('cuda') for _ in range(chunk.shape[0])])

        # Perform batch matrix multiplication for the chunk
        rotated_matrix[i:i+chunk_size] = torch.bmm(rotation_matrices, chunk.unsqueeze(2)).squeeze(2)

    return rotated_matrix

# Example random rotation matrix function (ensure this is CUDA-friendly)
def random_rotation_matrix_cuda(n):
    # Generate a random orthogonal matrix (this is an example, ensure it's on the correct device)
    q, _ = torch.qr(torch.randn(n, n, device='cuda'))
    return q

def compute_rescaled_kurtosis(Z):
    n = np.size(Z)
    A = np.mean(kurtosis(Z, axis = 0)*np.sqrt(n)/np.sqrt(33))
    return (A)

def hypothesis_testing(U, num_resamples=50, return_test_statistic=True):
    # Preallocate lists for results
    aggregated_kurtosis = []
    all_kurtosis_values = []
    null_varimax = []
    aggregated_rescaled_kurtosis = []
    def single_resample(_):
        U_rot = rotate_matrix(U)  # Rotate the matrix U
        Z_rot = varimax(U_rot)  # Perform Varimax rotation
        kurtosis_values = compute_kurtosis_stat(Z_rot)  # Compute kurtosis statistics
        varimax_obj = compute_varimax_stat(Z_rot)  # Compute Varimax objective
        rescaled_kurtosis_values = compute_rescaled_kurtosis(Z_rot) # TS3
        return np.mean(kurtosis_values), varimax_obj, rescaled_kurtosis_values
    results = Parallel(n_jobs=-1)(delayed(single_resample)(_) for _ in range(num_resamples))
    aggregated_kurtosis, null_varimax, aggregated_rescaled_kurtosis = zip(*results)  
  
    # Compute test statistics
    Z_hat = varimax(U)
    test_kurtosis_value = kurtosis(Z_hat, axis=0)
    test_varimax_obj = compute_varimax_stat(Z_hat)
    test_rescaled_kutosis_value = compute_rescaled_kurtosis(Z_hat)

    # Calculate p-values
    p_val_kurtosis = np.mean(np.array(aggregated_kurtosis) >= np.mean(test_kurtosis_value))
    p_val_rescaled_kurtosis = np.mean(np.array(aggregated_rescaled_kurtosis) >= np.mean(test_rescaled_kutosis_value))
    p_val_varimax = np.mean(np.abs(null_varimax) >= np.abs(test_varimax_obj))

    # Return p-values and optionally test statistics
    if return_test_statistic:
        return p_val_kurtosis, p_val_varimax, p_val_rescaled_kurtosis, aggregated_kurtosis, aggregated_rescaled_kurtosis, null_varimax, np.mean(test_kurtosis_value), test_varimax_obj
    else:
        return p_val_kurtosis, p_val_varimax, p_val_rescaled_kurtosis