"""
Simple Robust Auto-Annotation Pipeline
Works for ANY dataset size with direct CLIP + light refinement
"""

import numpy as np
import sys
import torch
import clip
from PIL import Image
from pathlib import Path
from typing import List
import json
import shutil


def extract_embeddings(image_folder: str, batch_size: int = 32):
    """Extract CLIP embeddings from images."""
    print(f"\n{'='*70}")
    print("STEP 1: EXTRACTING IMAGE EMBEDDINGS")
    print(f"{'='*70}\n")
    
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model (device: {device})...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print("✓ Model loaded\n")
    
    image_folder = Path(image_folder)
    image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
    image_paths = []
    
    for ext in image_extensions:
        image_paths.extend(list(image_folder.rglob(f"*{ext}")))
        image_paths.extend(list(image_folder.rglob(f"*{ext.upper()}")))
    
    image_paths = sorted(set(image_paths))
    
    if len(image_paths) == 0:
        print(f"❌ No images found in {image_folder}")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} images\n")
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
                print(f"Warning: Failed to load {img_path}: {e}")
                continue
        
        if len(batch_images) == 0:
            continue
        
        with torch.no_grad():
            image_input = torch.stack(batch_images).to(device)
            image_features = model.encode_image(image_input)
            image_features = image_features / image_features.norm(dim=-1, keepdim=True)
            embeddings.append(image_features.cpu().numpy())
            valid_paths.extend(batch_valid_paths)
        
        print(f"  Processed {len(valid_paths)}/{len(image_paths)} images...", end='\r')
    
    print(f"  Processed {len(valid_paths)}/{len(image_paths)} images... Done!\n")
    
    embeddings = np.vstack(embeddings)
    print(f"✓ Extracted embeddings: {embeddings.shape}\n")
    
    return embeddings, valid_paths, model, device


def generate_text_embeddings(class_names: List[str], model, device):
    """Generate text embeddings for class names."""
    print(f"{'='*70}")
    print("STEP 2: GENERATING CLASS TEXT EMBEDDINGS")
    print(f"{'='*70}\n")
    
    # Check if all class names are digits (0-9)
    is_digit_dataset = all(name.strip() in '0123456789' for name in class_names)
    
    if is_digit_dataset:
        digit_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        prompts = []
        for name in class_names:
            digit_idx = int(name.strip())
            multi_prompts = [
                f"the digit {digit_names[digit_idx]}",
                f"handwritten number {name}",
                f"the number {name}",
                f"digit {name}"
            ]
            prompts.append(multi_prompts)
        print(f"Digit classification mode enabled")
        print(f"Classes: {', '.join(class_names)}\n")
    else:
        prompts = [[f"a photo of a {name}"] for name in class_names]
        print(f"Classes: {', '.join(class_names)}\n")
    
    text_embeddings_list = []
    
    with torch.no_grad():
        for class_prompts in prompts:
            class_features = []
            for prompt in class_prompts:
                text_tokens = clip.tokenize([prompt]).to(device)
                features = model.encode_text(text_tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                class_features.append(features)
            avg_features = torch.stack(class_features).mean(dim=0)
            text_embeddings_list.append(avg_features.cpu().numpy())
    
    text_embeddings = np.vstack(text_embeddings_list)
    
    print("✓ Generated text embeddings\n")
    return text_embeddings


def annotate_with_simple_refinement(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    class_names: List[str]
):
    """Simple robust annotation with light refinement."""
    print(f"{'='*70}")
    print("STEP 3: ANNOTATION WITH REFINEMENT")
    print(f"{'='*70}\n")
    
    # Normalize
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    num_classes = len(class_names)
    num_images = len(image_embeddings)
    
    print(f"Dataset size: {num_images} images")
    print(f"Number of classes: {num_classes}\n")
    
    # Initialize prototypes from text
    prototypes = text_embeddings.copy()
    
    # Light refinement (3 iterations only)
    print("Refining prototypes (3 iterations)...")
    for iteration in range(3):
        # Compute similarities
        similarities = image_embeddings @ prototypes.T
        
        # Hard assignment
        labels = np.argmax(similarities, axis=1)
        
        # Update prototypes as class means
        new_prototypes = np.zeros_like(prototypes)
        for c in range(num_classes):
            class_mask = (labels == c)
            if np.sum(class_mask) > 0:
                # Average of images in this class
                class_mean = np.mean(image_embeddings[class_mask], axis=0)
                class_mean /= np.linalg.norm(class_mean)
                # Blend: 80% text, 20% visual (keep strong text influence)
                new_prototypes[c] = 0.8 * text_embeddings[c] + 0.2 * class_mean
                new_prototypes[c] /= np.linalg.norm(new_prototypes[c])
            else:
                # No images assigned - keep text embedding
                new_prototypes[c] = text_embeddings[c]
        
        prototypes = new_prototypes
        
        # Monitor
        confidences = np.max(similarities, axis=1)
        unique_classes = len(np.unique(labels))
        print(f"  Iteration {iteration + 1}/3: mean conf={np.mean(confidences):.3f}, active classes={unique_classes}/{num_classes}")
    
    print()
    
    # Final assignment
    similarities = image_embeddings @ prototypes.T
    labels = np.argmax(similarities, axis=1)
    confidences = np.max(similarities, axis=1)
    
    print("Final Class Distribution:")
    for class_idx, class_name in enumerate(class_names):
        count = np.sum(labels == class_idx)
        percentage = 100 * count / len(labels) if len(labels) > 0 else 0
        avg_conf = np.mean(confidences[labels == class_idx]) if count > 0 else 0
        print(f"  {class_name:15} {count:4} images ({percentage:5.1f}%) - avg confidence: {avg_conf:.3f}")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print(f"  Std confidence:  {np.std(confidences):.3f}")
    print(f"  Min confidence:  {np.min(confidences):.3f}")
    print(f"  Max confidence:  {np.max(confidences):.3f}\n")
    
    return labels, confidences, similarities


def organize_output(
    image_paths: List[str],
    labels: np.ndarray,
    confidences: np.ndarray,
    similarities: np.ndarray,
    class_names: List[str],
    output_dir: str
):
    """Organize images into class folders."""
    print(f"{'='*70}")
    print("STEP 4: ORGANIZING OUTPUT")
    print(f"{'='*70}\n")
    
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    # Create class folders
    for class_name in class_names:
        (output_path / class_name).mkdir(exist_ok=True)
    
    organized_paths = {name: [] for name in class_names}
    
    # Copy images
    print("Copying images to class folders...")
    for img_path, label in zip(image_paths, labels):
        src = Path(img_path)
        if not src.exists():
            continue
        
        class_name = class_names[label]
        dst = output_path / class_name / src.name
        
        counter = 1
        while dst.exists():
            dst = output_path / class_name / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        
        shutil.copy2(src, dst)
        organized_paths[class_name].append(str(dst.relative_to(output_path)))
    
    print("✓ Images organized\n")
    
    # Save metadata
    metadata = {
        'method': 'simple_refinement',
        'class_names': class_names,
        'num_images': len(image_paths),
        'class_distribution': {name: int(np.sum(labels == i)) for i, name in enumerate(class_names)},
        'organized_paths': organized_paths,
        'statistics': {
            'mean_confidence': float(np.mean(confidences)),
            'std_confidence': float(np.std(confidences)),
            'min_confidence': float(np.min(confidences)),
            'max_confidence': float(np.max(confidences))
        }
    }
    
    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)
    
    # Save report
    report_lines = [
        "="*70,
        "AUTO-ANNOTATION REPORT",
        "="*70,
        "",
        f"Total images: {len(image_paths)}",
        f"Classes: {len(class_names)}",
        "",
        "CLASS DISTRIBUTION:",
        "-"*70
    ]
    
    for class_idx, class_name in enumerate(class_names):
        count = np.sum(labels == class_idx)
        percentage = 100 * count / len(labels)
        avg_conf = np.mean(confidences[labels == class_idx]) if count > 0 else 0
        report_lines.append(f"  {class_name:15} {count:3} images ({percentage:5.1f}%) - avg conf: {avg_conf:.3f}")
    
    report_lines.extend([
        "",
        "CONFIDENCE STATISTICS:",
        "-"*70,
        f"  Mean: {np.mean(confidences):.3f}",
        f"  Std:  {np.std(confidences):.3f}",
        f"  Min:  {np.min(confidences):.3f}",
        f"  Max:  {np.max(confidences):.3f}",
        "",
        "="*70
    ])
    
    with open(output_path / 'report.txt', 'w') as f:
        f.write('\n'.join(report_lines))
    
    print(f"✓ Metadata saved: {output_path / 'metadata.json'}")
    print(f"✓ Report saved: {output_path / 'report.txt'}\n")
    
    # Create train/val/test splits
    print("Creating train/val/test splits...")
    splits_dir = output_path / 'splits'
    splits_dir.mkdir(exist_ok=True)
    
    for split_name in ['train', 'val', 'test']:
        split_dir = splits_dir / split_name
        split_dir.mkdir(exist_ok=True)
        for class_name in class_names:
            (split_dir / class_name).mkdir(exist_ok=True)
    
    np.random.seed(42)
    split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}
    
    for class_name in class_names:
        class_folder = output_path / class_name
        images = list(class_folder.glob('*'))
        
        if len(images) == 0:
            continue
        
        indices = np.random.permutation(len(images))
        splits = ['train', 'val', 'test']
        ratios = [split_ratios[s] for s in splits]
        ratios = np.array(ratios) / sum(ratios)
        split_points = np.cumsum(ratios * len(images)).astype(int)
        
        start = 0
        for split_idx, split_name in enumerate(splits):
            end = split_points[split_idx]
            split_indices = indices[start:end]
            
            for idx in split_indices:
                src = images[idx]
                dst = splits_dir / split_name / class_name / src.name
                shutil.copy2(src, dst)
            
            start = end
    
    print(f"✓ Splits created: {splits_dir}\n")


def run_pipeline(image_folder: str, output_dir: str, class_names: List[str]):
    """Run complete pipeline."""
    print(f"\n{'='*70}")
    print("SIMPLE ROBUST AUTO-ANNOTATION PIPELINE")
    print(f"{'='*70}")
    print(f"Input: {image_folder}")
    print(f"Output: {output_dir}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"{'='*70}\n")
    
    # Extract embeddings
    image_embeddings, image_paths, model, device = extract_embeddings(image_folder)
    
    # Generate text embeddings
    text_embeddings = generate_text_embeddings(class_names, model, device)
    
    # Annotate with simple refinement
    labels, confidences, similarities = annotate_with_simple_refinement(
        image_embeddings,
        text_embeddings,
        class_names
    )
    
    # Organize output
    organize_output(
        image_paths,
        labels,
        confidences,
        similarities,
        class_names,
        output_dir
    )
    
    print(f"{'='*70}")
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")
    print(f"Your annotated dataset is ready at: {output_dir}\n")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("=" * 70)
        print("SIMPLE ROBUST AUTO-ANNOTATION PIPELINE")
        print("=" * 70)
        print("\nUsage:")
        print("  python auto_pipeline_simple.py <image_folder> <output_dir> <class1> <class2> ...")
        print("\nExample:")
        print("  python auto_pipeline_simple.py data/images3 output 0 1 2 3 4 5 6 7 8 9")
        print("\nHow it works:")
        print("  1. Extracts CLIP embeddings")
        print("  2. Generates text embeddings (with special handling for digits)")
        print("  3. Simple refinement (3 iterations, 80% text / 20% visual)")
        print("  4. Hard assignment to classes")
        print("  5. Organizes into folders + splits")
        print()
        sys.exit(1)
    
    image_folder = sys.argv[1]
    output_dir = sys.argv[2]
    class_names = sys.argv[3:]
    
    if not Path(image_folder).exists():
        print(f"❌ Error: Image folder not found: {image_folder}")
        sys.exit(1)
    
    try:
        run_pipeline(image_folder, output_dir, class_names)
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
