import torch
import random
import open_clip
import pandas as pd
from util import *
import numpy as np
from util_hypothesis_test import *

# Load model and preprocessor from open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-B-32', pretrained='openai')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()  # Set the model to evaluation mode


# Set parameters for white noise images and batch size
num_images = 10000
embedding_dim = 768

results = []  # List to hold the results

random_seeds = [42, 1234, 2021, 8675309, 314159]
k_list = [10, 25, 50, 75, 100, 125, 150]
for random_seed in random_seeds:
    random.seed(random_seed)
    print('random seed: ', random_seed)
    # Create the white noise dataset and DataLoader

    white_noise_embeddings = np.random.normal(0, 1, num_images * embedding_dim)
    white_noise_embeddings = white_noise_embeddings.reshape(num_images, embedding_dim)

    # Determine the embedding dimension from the model
    L = glaplacian(white_noise_embeddings)
    for k in k_list:
        U, S, Vt = randomized_svd(L, n_components=k)
        res = hypothesis_testing(U[:, 1:], num_resamples=100, return_test_statistic=True)
        print(res[:3])

        # Extract the data
        p_values = res[:3]
        null_kurtosis_avg = np.mean(res[3])
        observed_kurtosis = np.mean(res[5])
        null_varimax = np.mean(res[-2])
        observed_varimax = res[-1]

        # Append to results
        results.append({
            'random_seed': random_seed,
            'k': k,
            'p_value_kurtosis': p_values[0],
            'p_value_varimax': p_values[1],
            'p_value_rescaled_kurtosis': p_values[2],
            'null_kurtosis_average': null_kurtosis_avg,
            'observed_kurtosis': observed_kurtosis,
            'null_varimax': null_varimax,
            'observed_varimax': observed_varimax
        })

# After the loops, create dataframe
df = pd.DataFrame(results)

# Save to csv
df.to_csv('wn_results.csv', index=False)
print("Results saved to 'wn_results.csv'")