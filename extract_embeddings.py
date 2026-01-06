"""
Extract image embeddings using CLIP for use with Auto-Annotation Engine.

This script processes a folder of images and creates:
1. embeddings.npy - Image embeddings [N, D]
2. image_paths.txt - List of processed image paths

Requirements: pip install git+https://github.com/openai/CLIP.git torch torchvision pillow
"""

import torch
import clip
from PIL import Image
import numpy as np
from pathlib import Path
from typing import List
import sys


def extract_clip_embeddings(
    image_folder: str,
    output_dir: str = "data",
    model_name: str = "ViT-B/32",
    batch_size: int = 32,
    image_extensions: List[str] = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
):
    """
    Extract CLIP embeddings from all images in a folder.
    
    Args:
        image_folder: Path to folder containing images
        output_dir: Where to save embeddings and paths
        model_name: CLIP model to use
        batch_size: Number of images to process at once
        image_extensions: List of valid image extensions
    """
    print(f"\n{'='*70}")
    print("CLIP EMBEDDING EXTRACTION")
    print(f"{'='*70}\n")
    
    # Load CLIP model
    print(f"Loading CLIP model: {model_name}...")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    model, preprocess = clip.load(model_name, device=device)
    model.eval()
    print("✓ Model loaded\n")
    
    # Find all images
    image_folder = Path(image_folder)
    print(f"Scanning for images in: {image_folder}")
    
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(list(image_folder.rglob(f"*{ext}")))
        image_paths.extend(list(image_folder.rglob(f"*{ext.upper()}")))
    
    image_paths = sorted(set(image_paths))  # Remove duplicates and sort
    
    if len(image_paths) == 0:
        print(f"❌ No images found in {image_folder}")
        print(f"   Looking for: {', '.join(image_extensions)}")
        sys.exit(1)
    
    print(f"✓ Found {len(image_paths)} images\n")
    
    # Extract embeddings
    print("Extracting embeddings...")
    embeddings = []
    valid_paths = []
    
    for i in range(0, len(image_paths), batch_size):
        batch_paths = image_paths[i:i + batch_size]
        batch_images = []
        batch_valid_paths = []
        
        for img_path in batch_paths:
            try:
                image = Image.open(img_path).convert('RGB')
                batch_images.append(preprocess(image))
                batch_valid_paths.append(str(img_path))
            except Exception as e:
                print(f"  Warning: Failed to load {img_path}: {e}")
                continue
        
        if len(batch_images) == 0:
            continue
        
        # Process batch
        with torch.no_grad():
            image_input = torch.stack(batch_images).to(device)
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy())
            valid_paths.extend(batch_valid_paths)
        
        print(f"  Processed {len(valid_paths)}/{len(image_paths)} images...", end='\r')
    
    print(f"  Processed {len(valid_paths)}/{len(image_paths)} images... Done!")
    
    if len(embeddings) == 0:
        print("❌ No embeddings extracted")
        sys.exit(1)
    
    # Combine all embeddings
    embeddings = np.vstack(embeddings)
    
    print(f"\n✓ Extracted embeddings: shape {embeddings.shape}")
    print(f"  Dimension: {embeddings.shape[1]}")
    print(f"  Successfully processed: {len(valid_paths)} images\n")
    
    # Save outputs
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    embeddings_path = output_dir / 'embeddings.npy'
    paths_path = output_dir / 'image_paths.txt'
    
    np.save(embeddings_path, embeddings)
    print(f"✓ Saved embeddings to: {embeddings_path}")
    
    with open(paths_path, 'w') as f:
        f.write('\n'.join(valid_paths))
    print(f"✓ Saved image paths to: {paths_path}")
    
    print(f"\n{'='*70}")
    print("READY FOR AUTO-ANNOTATION!")
    print(f"{'='*70}")
    print("\nNext step:")
    print(f"  python main.py \\")
    print(f"    --embeddings {embeddings_path} \\")
    print(f"    --image-paths {paths_path} \\")
    print(f"    --classes class1 class2 class3 \\")
    print(f"    --output annotated_dataset")
    print()


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(
        description="Extract CLIP embeddings from images",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
  # Extract from images folder
  python extract_embeddings.py data/images
  
  # Specify output location
  python extract_embeddings.py data/images --output my_data
  
  # Use larger CLIP model
  python extract_embeddings.py data/images --model ViT-L/14
        """
    )
    
    parser.add_argument(
        "image_folder",
        type=str,
        help="Path to folder containing images"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="data",
        help="Output directory for embeddings and paths (default: data)"
    )
    parser.add_argument(
        "--model",
        type=str,
        default="ViT-B/32",
        choices=["RN50", "RN101", "RN50x4", "RN50x16", "RN50x64", 
                 "ViT-B/32", "ViT-B/16", "ViT-L/14", "ViT-L/14@336px"],
        help="CLIP model to use (default: ViT-B/32)"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=32,
        help="Batch size for processing (default: 32)"
    )
    
    args = parser.parse_args()
    
    try:
        extract_clip_embeddings(
            image_folder=args.image_folder,
            output_dir=args.output,
            model_name=args.model,
            batch_size=args.batch_size
        )
    except ImportError as e:
        print(f"\n❌ Error: {e}")
        print("\nPlease install required packages:")
        print("  pip install git+https://github.com/openai/CLIP.git torch torchvision pillow")
        sys.exit(1)
