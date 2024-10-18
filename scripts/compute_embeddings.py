import argparse
import torch
from torch.utils.data import DataLoader
from torchvision import datasets
from PIL import Image
from tqdm import tqdm
import open_clip
import random
from wilds import get_dataset

# usage:
# python compute_embeddings.py --dataset imagenet --model_name ViT-B-32 --pretrained openai --output_path ../computed_embeddings/imagenet_vit-b-32.npy --batch_size 64 --seed 1234

def main():
    # Parse command-line arguments
    parser = argparse.ArgumentParser(description='Compute embeddings using OpenCLIP model.')
    parser.add_argument('--dataset', type=str, required=True, choices=['imagenet', 'waterbirds'],
                        help='Dataset to use: imagenet or waterbirds')
    parser.add_argument('--model_name', type=str, default='ViT-B-32',
                        help='Model name for OpenCLIP (e.g., ViT-L-14)')
    parser.add_argument('--pretrained', type=str, default='openai',
                        help='Pretrained weights to use (e.g., openai)')
    parser.add_argument('--output_path', type=str, required=True,
                        help='Path to save the computed embeddings')
    parser.add_argument('--batch_size', type=int, default=64,
                        help='Batch size for DataLoader')
    parser.add_argument('--seed', type=int, default=1234,
                        help='Random seed')
    parser.add_argument('--device', type=str, default='cuda',
                        help='Device to use (cuda or cpu)')
    parser.add_argument('--imagenet_path', type=str, default='../data/imagenet/val/',
                        help='Path to ImageNet validation images')
    args = parser.parse_args()

    # Set random seed
    random.seed(args.seed)
    torch.manual_seed(args.seed)

    # Load model and preprocessor
    model, _, preprocess = open_clip.create_model_and_transforms(
        args.model_name, pretrained=args.pretrained)
    device = args.device
    model = model.to(device)
    model.eval()  # Set model to evaluation mode

    # Depending on dataset, load the dataset and set up DataLoader
    if args.dataset == 'imagenet':
        # Load ImageNet validation set
        dataset = datasets.ImageFolder(root=args.imagenet_path)
        # Define collate function
        def collate_fn(batch):
            images, labels = zip(*batch)
            images = [preprocess(image) for image in images]
            inputs = torch.stack(images)
            return inputs, labels
        dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=8)
        num_images = len(dataset)
    elif args.dataset == 'waterbirds':
        # Load Waterbirds dataset
        dataset = get_dataset(dataset="waterbirds", download=True)
        test_data = dataset.get_subset('test')
        # Define collate function
        def collate_fn(batch):
            images = [img for img, _, _ in batch]
            labels = [label for _, label, _ in batch]
            attributes = [torch.argmax(attr).item() for _, _, attr in batch]
            images = [preprocess(image) for image in images]
            inputs = torch.stack(images)
            labels = torch.tensor(labels)
            attributes = torch.tensor(attributes)
            return inputs, labels, attributes
        dataloader = DataLoader(test_data, batch_size=args.batch_size, shuffle=False,
                                collate_fn=collate_fn, num_workers=8)
        num_images = len(test_data)
    else:
        raise ValueError('Unsupported dataset')

    # Initialize tensor to store embeddings
    embedding_dim = model.visual.output_dim  # Get the embedding dimension from the model
    all_image_features = torch.empty((num_images, embedding_dim), device=device)

    start_idx = 0

    # Compute embeddings
    for batch in tqdm(dataloader):
        if args.dataset == 'imagenet':
            inputs, _ = batch  # Ignore labels
        elif args.dataset == 'waterbirds':
            inputs, _, _ = batch  # Ignore labels and attributes
        else:
            raise ValueError('Unsupported dataset')
        inputs = inputs.to(device)
        with torch.no_grad():
            image_features = model.encode_image(inputs)

        # Store the embeddings
        end_idx = start_idx + image_features.size(0)
        all_image_features[start_idx:end_idx] = image_features
        start_idx = end_idx

    # Save the embeddings
    torch.save(all_image_features.cpu(), args.output_path)
    print(f"Embeddings saved to {args.output_path}")

if __name__ == '__main__':
    main()