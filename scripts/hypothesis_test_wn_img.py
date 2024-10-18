import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
import random
import open_clip
import pandas as pd
from util import *
import numpy as np
from util_hypothesis_test import *


# Load model and preprocessor from open_clip
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
device = "cuda" if torch.cuda.is_available() else "cpu"
model = model.to(device)
model.eval()  # Set the model to evaluation mode

# Define the collate function for the DataLoader
def collate_fn_whitenoise(batch):
    # batch is a list of images
    images = [preprocess(image) for image in batch]
    inputs = torch.stack(images)
    return inputs

# Set parameters for white noise images and batch size
num_images = 10000
height, width, channels = 224, 224, 3
batch_size = 64


results = []  # List to hold the results

random_seeds = [42, 1234, 2021, 8675309, 314159]
k_list = [10, 25, 50, 75, 100, 125, 150]
for random_seed in random_seeds:
    random.seed(random_seed)
    print('random seed: ', random_seed)
    # Create the white noise dataset and DataLoader
    white_noise_dataset = WhiteNoiseDataset(num_images, height, width, channels)
    dataloader = DataLoader(
        white_noise_dataset,
        batch_size=batch_size,
        shuffle=False,
        collate_fn=collate_fn_whitenoise
    )

    # Determine the embedding dimension from the model
    embedding_dim = model.visual.output_dim  # open_clip model's embedding dimension
    # Preallocate a tensor to store all embeddings
    white_noise_image_embeddings = torch.empty((num_images, embedding_dim), device=device)

    # Process and feed the white noise images into the model
    start_idx = 0
    for inputs in tqdm(dataloader):
        # Move inputs to the device
        inputs = inputs.to(device)

        # Get embeddings with the open_clip model
        with torch.no_grad():
            image_features = model.encode_image(inputs)  # (batch_size, embedding_dim)

        # Store the batch embeddings into the preallocated tensor
        end_idx = start_idx + image_features.size(0)
        white_noise_image_embeddings[start_idx:end_idx] = image_features
        start_idx = end_idx

    L = glaplacian(white_noise_image_embeddings.to('cpu'))
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
df.to_csv('wn_img_vit-l-14.csv', index=False)
print("Results saved to 'wn_img_vit-l-14.csv'")