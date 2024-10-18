#!/usr/bin/env python3

import argparse
import os
import random
import numpy as np
import torch
from PIL import Image
from scipy.sparse import csr_matrix
from sklearn.decomposition import randomized_svd
from torchvision import transforms
from torchvision.datasets import ImageFolder
import matplotlib.pyplot as plt
from util import *  

# python concept_decomposition.py \
#     --image_features_path ~/embedding_hc/computed_embeddings/image_features_vit-b-32.pt \
#     --text_features_path ../computed_embeddings/text_features_general_descriptions_vit-b-32.pt \
#     --num_components 50 \
#     --test_images_path ../data/imagenet/val/ \
#     --concepts_file ./text_descriptions/image_descriptions_per_class.txt \
#     --class_map_file ../data/imagenet/map_clsloc.txt \
#     --output_path ~/embedding_hc/imgs/clip/vit_b/ortho_concept_512d/

def main(args):
    # Set random seed for reproducibility
    random.seed(1234)

    # Load image features
    image_features = torch.load(args.image_features_path)
    # Load text features
    text_features = torch.load(args.text_features_path)
    k = args.num_components  # Number of components for SVD
    
    # Compute graph Laplacian
    L = glaplacian(torch.tensor(image_features))
    # Perform randomized SVD
    U, S, Vt = randomized_svd(L, n_components=k)
    S_diag = np.diag(S) 
    Z = U @ S_diag
    Z_rotated, Rz = varimax_with_rotation(Z)
    sign_Z = np.diag(np.where(np.mean(Z_rotated**3, axis=0) >= 0, 1, -1))
    Z_hat = Z_rotated @ sign_Z
    Y_hat = sign_Z @ Rz.T @ Vt
    
    # Load ImageNet class names mapping
    wnid_to_class = {}
    with open(args.class_map_file, 'r') as file:
        for line in file:
            parts = line.strip().split(' ')
            wnid = parts[0]
            class_name = ' '.join(parts[2:])
            wnid_to_class[wnid] = class_name 
    
    # Prepare dataset
    transform = transforms.Compose([
        transforms.Resize((224,224)),
        transforms.ToTensor()
    ])
    raw_dataset = ImageFolder(root=args.test_images_path, transform=transform)
    
    # Ensure output path exists
    os.makedirs(args.output_path, exist_ok=True)
    
    # save top positive images per cluster
    plot_top_pos_images_per_cluster(Z_hat, raw_dataset, wnid_to_class, args.output_path, n=9, nrows=3, ncols=3)
    
    # Load concepts
    concepts = []
    with open(args.concepts_file, "r") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            concepts.append(line)
    
    # Project the text embeddings
    projected_text_embedding = text_features @ Y_hat.T
    
    # Get top positive and negative texts
    top_texts, _ = get_top_n_elements(csr_matrix(projected_text_embedding), n=10, axis=0)

    # Prepare the figure with a grid layout
    fig, axs = plt.subplots(nrows=2, ncols=3, figsize=(20, 20))  # Adjust the size as needed
    axs = axs.flatten()  # Flatten the array to simplify indexing
    
    # Loop over the range of clusters/images you have
    for i in range(6):
        image_path = os.path.join(args.output_path, f'cluster_{i + 1}.png')
        image = Image.open(image_path).convert("RGB")
    
        # Display image
        axs[i].imshow(np.array(image))
        axs[i].axis('off')  # Turn off axis
    
        # Create text description combining top texts
        description = "\n".join([concepts[txt] for txt in top_texts[i]])
        
        # Set title or use text annotation inside the subplot
        axs[i].set_title(f'Concept {i + 1}', fontsize=20)
        axs[i].text(0.5, -0.1, description, va='top', ha='center', fontsize=15, transform=axs[i].transAxes)
    
    plt.tight_layout(pad=4.0)  # Adjust padding as necessary
    plt.savefig('concept_decomposition.png', dpi=150)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Compute concept decomposition')
    parser.add_argument('--image_features_path', type=str, required=True, help='Path to image features file')
    parser.add_argument('--text_features_path', type=str, required=True, help='Path to text features file')
    parser.add_argument('--num_components', type=int, default=50, help='Number of components for SVD')
    parser.add_argument('--test_images_path', type=str, required=True, help='Path to test images directory')
    parser.add_argument('--concepts_file', type=str, required=True, help='Path to concepts file')
    parser.add_argument('--class_map_file', type=str, default=None, help='Path to class mapping file for Image')
    parser.add_argument('--output_path', type=str, required=True, help='Output directory for plots')

    args = parser.parse_args()
    main(args)