"""
Complete End-to-End Auto-Annotation Pipeline

Single command to go from raw images to organized dataset.
Handles everything automatically: embedding extraction, clustering, and organization.

Usage:
    python auto_pipeline.py <image_folder> <output_dir> <class1> <class2> <class3> ...

Example:
    python auto_pipeline.py data/images annotated_output dog cat tiger lion giraffe
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
import tempfile
from sklearn.cluster import DBSCAN
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False


def extract_embeddings_inline(image_folder: str, batch_size: int = 32):
    """Extract CLIP embeddings from images."""
    print(f"\n{'='*70}")
    print("STEP 1: EXTRACTING IMAGE EMBEDDINGS")
    print(f"{'='*70}\n")
    
    # Load CLIP
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Loading CLIP model (device: {device})...")
    model, preprocess = clip.load("ViT-B/32", device=device)
    model.eval()
    print("✓ Model loaded\n")
    
    # Find images
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
    
    # Extract embeddings
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


def generate_text_embeddings_inline(class_names: List[str], model, device):
    """Generate text embeddings for class names."""
    print(f"{'='*70}")
    print("STEP 2: GENERATING CLASS TEXT EMBEDDINGS")
    print(f"{'='*70}\n")
    
    # Check if all class names are digits (0-9)
    is_digit_dataset = all(name.strip() in '0123456789' for name in class_names)
    
    if is_digit_dataset:
        # Special prompts for digit recognition
        digit_names = ['zero', 'one', 'two', 'three', 'four', 'five', 'six', 'seven', 'eight', 'nine']
        prompts = []
        for name in class_names:
            digit_idx = int(name.strip())
            # Use multiple descriptions and average them for better accuracy
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
        # Standard prompts for object/animal classification
        prompts = [[f"a photo of a {name}"] for name in class_names]
        print(f"Classes: {', '.join(class_names)}\n")
    
    # Generate embeddings for each class (averaging multiple prompts if provided)
    text_embeddings_list = []
    
    with torch.no_grad():
        for class_prompts in prompts:
            if isinstance(class_prompts, list):
                # Multiple prompts per class - average them
                class_features = []
                for prompt in class_prompts:
                    text_tokens = clip.tokenize([prompt]).to(device)
                    features = model.encode_text(text_tokens)
                    features = features / features.norm(dim=-1, keepdim=True)
                    class_features.append(features)
                # Average the embeddings
                avg_features = torch.stack(class_features).mean(dim=0)
                text_embeddings_list.append(avg_features.cpu().numpy())
            else:
                # Single prompt
                text_tokens = clip.tokenize([class_prompts]).to(device)
                features = model.encode_text(text_tokens)
                features = features / features.norm(dim=-1, keepdim=True)
                text_embeddings_list.append(features.cpu().numpy())
    
    text_embeddings = np.vstack(text_embeddings_list)
    
    print("✓ Generated text embeddings\n")
    return text_embeddings


def annotate_with_refinement(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    class_names: List[str],
    num_iterations: int = 10,
    temperature: float = 0.05
):
    """Iteratively refine class assignments using clustering + learning."""
    print(f"{'='*70}")
    print("STEP 3: CLUSTERING-BASED ITERATIVE REFINEMENT")
    print(f"{'='*70}\n")
    
    # Normalize
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    
    num_classes = len(class_names)
    num_images = len(image_embeddings)
    
    print(f"Dataset size: {num_images} images")
    print(f"Number of classes: {num_classes}")
    print(f"Iterations: {num_iterations}\n")
    
    # Use clustering to find visual groups
    print("Step 3a: Finding visual clusters...")
    from sklearn.cluster import KMeans
    
    # More clusters than classes for finer granularity
    num_clusters = max(num_classes * 3, 20)
    num_clusters = min(num_clusters, num_images // 2)
    
    kmeans = KMeans(n_clusters=num_clusters, random_state=42, n_init=10)
    cluster_ids = kmeans.fit_predict(image_embeddings)
    cluster_centroids = kmeans.cluster_centers_
    cluster_centroids = cluster_centroids / np.linalg.norm(cluster_centroids, axis=1, keepdims=True)
    
    print(f"  Created {num_clusters} visual clusters\n")
    
    # Initialize prototypes from text embeddings
    prototypes = text_embeddings.copy()
    
    # Iterative refinement at cluster level
    print("Step 3b: Iterative cluster-to-class alignment...")
    
    for iteration in range(num_iterations):
        # Compute cluster-to-class similarities
        cluster_similarities = cluster_centroids @ prototypes.T  # [K, C]
        
        # Soft assignment of clusters to classes
        logits = cluster_similarities / temperature
        logits = logits - np.max(logits, axis=1, keepdims=True)
        exp_logits = np.exp(logits)
        cluster_to_class_probs = exp_logits / np.sum(exp_logits, axis=1, keepdims=True)
        
        # Update prototypes based on cluster assignments
        new_prototypes = np.zeros_like(prototypes)
        prototype_weights = np.zeros(num_classes)
        
        for k in range(num_clusters):
            cluster_mask = (cluster_ids == k)
            cluster_images = image_embeddings[cluster_mask]
            
            if len(cluster_images) == 0:
                continue
            
            # Distribute cluster's contribution across classes
            for c in range(num_classes):
                weight = cluster_to_class_probs[k, c] * len(cluster_images)
                new_prototypes[c] += weight * cluster_centroids[k]
                prototype_weights[c] += weight
        
        # Normalize and blend with text embeddings
        for c in range(num_classes):
            if prototype_weights[c] > 0:
                new_prototypes[c] /= prototype_weights[c]
                # Keep text embedding influence (30% text, 70% visual)
                prototypes[c] = 0.3 * text_embeddings[c] + 0.7 * new_prototypes[c]
                prototypes[c] = prototypes[c] / np.linalg.norm(prototypes[c])
        
        # Monitor progress
        image_similarities = image_embeddings @ prototypes.T
        labels = np.argmax(image_similarities, axis=1)
        confidences = np.max(image_similarities, axis=1)
        
        unique, counts = np.unique(labels, return_counts=True)
        
        print(f"  Iteration {iteration + 1}/{num_iterations}: "
              f"mean conf={np.mean(confidences):.3f}, "
              f"filled classes={len(unique)}/{num_classes}")
    
    # Final assignment
    similarities = image_embeddings @ prototypes.T
    labels = np.argmax(similarities, axis=1)
    confidences = np.max(similarities, axis=1)
    
    print(f"\\nFinal Class Distribution:")
    for class_idx, class_name in enumerate(class_names):
        count = np.sum(labels == class_idx)
        percentage = 100 * count / len(labels) if len(labels) > 0 else 0
        avg_conf = np.mean(confidences[labels == class_idx]) if count > 0 else 0
        print(f"  {class_name:15} {count:4} images ({percentage:5.1f}%) - avg confidence: {avg_conf:.3f}")
    
    print(f"\\nOverall Statistics:")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print(f"  Std confidence:  {np.std(confidences):.3f}")
    print(f"  Min confidence:  {np.min(confidences):.3f}")
    print(f"  Max confidence:  {np.max(confidences):.3f}\\n")
    
    return labels, confidences, similarities


def annotate_images_inline(
    image_embeddings: np.ndarray,
    text_embeddings: np.ndarray,
    class_names: List[str],
    temperature: float = 0.05
):
    """Directly assign images to classes using CLIP similarity."""
    print(f"{'='*70}")
    print("STEP 3: DIRECT CLASSIFICATION (SMALL DATASET)")
    print(f"{'='*70}\n")
    
    # Normalize
    image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
    
    # Compute similarities
    similarities = image_embeddings @ text_embeddings.T  # [N, C]
    
    # Assign labels
    labels = np.argmax(similarities, axis=1)
    confidences = np.max(similarities, axis=1)
    
    print(f"Dataset size: {len(image_embeddings)} images")
    print(f"Number of classes: {len(class_names)}\n")
    
    print("Class Distribution:")
    for class_idx, class_name in enumerate(class_names):
        count = np.sum(labels == class_idx)
        percentage = 100 * count / len(labels)
        avg_conf = np.mean(confidences[labels == class_idx]) if count > 0 else 0
        print(f"  {class_name:15} {count:3} images ({percentage:5.1f}%) - avg confidence: {avg_conf:.3f}")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print(f"  Min confidence:  {np.min(confidences):.3f}")
    print(f"  Max confidence:  {np.max(confidences):.3f}\n")
    
    return labels, confidences, similarities


def organize_output_inline(
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
        
        # Handle duplicates
        counter = 1
        while dst.exists():
            dst = output_path / class_name / f"{src.stem}_{counter}{src.suffix}"
            counter += 1
        
        shutil.copy2(src, dst)
        organized_paths[class_name].append(str(dst.relative_to(output_path)))
    
    print("✓ Images organized\n")
    
    # Save metadata
    metadata = {
        'method': 'direct_clip_annotation',
        'class_names': class_names,
        'num_images': len(image_paths),
        'class_distribution': {name: int(np.sum(labels == i)) for i, name in enumerate(class_names)},
        'organized_paths': organized_paths,
        'image_annotations': [
            {
                'original_path': img_path,
                'class_index': int(label),
                'class_name': class_names[label],
                'confidence': float(conf),
                'all_similarities': {class_names[j]: float(similarities[i, j]) for j in range(len(class_names))}
            }
            for i, (img_path, label, conf) in enumerate(zip(image_paths, labels, confidences))
        ],
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
    
    # Split images
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
    """Run complete end-to-end pipeline."""
    print(f"\n{'='*70}")
    print("AUTO-ANNOTATION COMPLETE PIPELINE")
    print(f"{'='*70}")
    print(f"Input: {image_folder}")
    print(f"Output: {output_dir}")
    print(f"Classes: {', '.join(class_names)}")
    print(f"{'='*70}\n")
    
    # Step 1: Extract embeddings
    image_embeddings, image_paths, model, device = extract_embeddings_inline(image_folder)
    
    dataset_size = len(image_embeddings)
    
    # Step 2: Generate text embeddings
    text_embeddings = generate_text_embeddings_inline(class_names, model, device)
    
    # Step 3: Adaptive annotation based on dataset size
    if dataset_size < 100:
        # Small dataset: Direct CLIP classification
        print(f"Dataset size ({dataset_size}) < 100: Using direct CLIP classification\n")
        labels, confidences, similarities = annotate_images_inline(
            image_embeddings,
            text_embeddings,
            class_names,
            temperature=0.05
        )
    else:
        # Large dataset: Use iterative refinement with clustering
        print(f"Dataset size ({dataset_size}) >= 100: Using clustering + iterative refinement\\n")
        labels, confidences, similarities = annotate_with_refinement(
            image_embeddings,
            text_embeddings,
            class_names,
            num_iterations=10,
            temperature=0.05
        )
    
    # Step 4: Organize output
    organize_output_inline(
        image_paths,
        labels,
        confidences,
        similarities,
        class_names,
        output_dir
    )
    
    # Final summary
    print(f"{'='*70}")
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print(f"{'='*70}\n")
    print(f"Your annotated dataset is ready at: {output_dir}")
    print(f"\nOutput structure:")
    print(f"  {output_dir}/")
    for class_name in class_names:
        print(f"    ├── {class_name}/")
    print(f"    ├── metadata.json")
    print(f"    ├── report.txt")
    print(f"    └── splits/")
    print(f"          ├── train/")
    print(f"          ├── val/")
    print(f"          └── test/")
    print(f"\nNext steps:")
    print(f"  1. Review results: cat {output_dir}/report.txt")
    print(f"  2. Check images in class folders")
    print(f"  3. Use splits/ for training\n")


if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("=" * 70)
        print("AUTO-ANNOTATION COMPLETE PIPELINE")
        print("=" * 70)
        print("\nUsage:")
        print("  python auto_pipeline.py <image_folder> <output_dir> <class1> <class2> ...")
        print("\nExample:")
        print("  python auto_pipeline.py data/images annotated_output dog cat tiger lion giraffe")
        print("\nWhat this does:")
        print("  1. Extracts CLIP embeddings from all images")
        print("  2. Generates text embeddings for your classes")
        print("  3. Assigns images to classes using similarity")
        print("  4. Organizes images into class folders")
        print("  5. Creates train/val/test splits")
        print("\nRequirements:")
        print("  pip install git+https://github.com/openai/CLIP.git torch torchvision pillow")
        print()
        sys.exit(1)
    
    image_folder = sys.argv[1]
    output_dir = sys.argv[2]
    class_names = sys.argv[3:]
    
    # Verify input folder exists
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
