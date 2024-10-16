import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
from tqdm import tqdm
import open_clip
import random

random.seed(1234)

# Load model and preprocessor
model, _, preprocess = open_clip.create_model_and_transforms('ViT-L-14', pretrained='openai')
tokenizer = open_clip.get_tokenizer('ViT-L-14')
device = "cuda"
model = model.to(device)

# collate functions
def collate_fn(batch):
    images, labels = zip(*batch)  # Unpack the batch into images and labels
    # Apply the preprocess function to each image
    images = [preprocess(image) for image in images]
    # Stack the preprocessed images into a single tensor
    inputs = torch.stack(images)
    return inputs, labels

def collate_fn_whitenoise(batch):
    images = zip(*batch)  # Unpack the batch into images and labels
    # Apply the preprocess function to each image
    images = [preprocess(image) for image in images]
    # Stack the preprocessed images into a single tensor
    inputs = torch.stack(images)
    return inputs

# Load the dataset
test_images_path = '../data/imagenet/val/'
dataset = datasets.ImageFolder(root=test_images_path)
dataloader = DataLoader(dataset, batch_size=64, shuffle=False, collate_fn=collate_fn, num_workers=16)  # Use 4 CPU workers, you can adjust this based on your CPU
num_images = len(dataset)
embedding_dim = 768  # open_clip model's embedding dimension
all_image_features = torch.empty((num_images, embedding_dim), device=device)
start_idx = 0

for inputs, _ in tqdm(dataloader):  # We ignore the labels here
    # Move the inputs to the device (in this case CPU)
    inputs = inputs.to(device)
    
    with torch.no_grad():
        image_features = model.encode_image(inputs)  # (batch_size, 512)
    # Store the batch embeddings into the preallocated tensor
    end_idx = start_idx + image_features.size(0)
    all_image_features[start_idx:end_idx] = image_features
    start_idx = end_idx

torch.save(all_image_features, '../computed_embeddings/image_features_vit-l-14.pt')