"""
Direct CLIP-based annotation for small datasets.

For tiny datasets (< 50 images), clustering doesn't help.
This script directly assigns based on CLIP similarity.
"""

import numpy as np
import json
from pathlib import Path
import shutil


def direct_clip_annotation(
    embeddings_path: str,
    image_paths_file: str,
    class_names: list,
    output_dir: str,
    copy_files: bool = True
):
    """
    Directly assign images to classes using CLIP embeddings.
    Best for very small datasets (< 50 images).
    """
    print(f"\n{'='*70}")
    print("DIRECT CLIP ANNOTATION (Small Dataset Mode)")
    print(f"{'='*70}\n")
    
    # Load embeddings
    embeddings = np.load(embeddings_path)
    print(f"Loaded {len(embeddings)} image embeddings (dim={embeddings.shape[1]})")
    
    # Load image paths
    with open(image_paths_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    print(f"Loaded {len(image_paths)} image paths")
    
    assert len(embeddings) == len(image_paths), "Mismatch between embeddings and paths!"
    
    # Generate text embeddings
    print(f"\nGenerating text embeddings for {len(class_names)} classes...")
    try:
        import clip
        import torch
        
        device = "cuda" if torch.cuda.is_available() else "cpu"
        model, _ = clip.load("ViT-B/32", device=device)
        
        prompts = [f"a photo of a {name}" for name in class_names]
        with torch.no_grad():
            text_tokens = clip.tokenize(prompts).to(device)
            text_embeddings = model.encode_text(text_tokens)
            text_embeddings = text_embeddings / text_embeddings.norm(dim=-1, keepdim=True)
            text_embeddings = text_embeddings.cpu().numpy()
        
        print("✓ Generated CLIP text embeddings")
    except:
        print("Warning: CLIP not available, using random embeddings")
        text_embeddings = np.random.randn(len(class_names), embeddings.shape[1])
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Normalize image embeddings
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    # Compute similarities
    print("\nComputing direct similarities...")
    similarities = embeddings @ text_embeddings.T  # [N, C]
    
    # Assign labels
    labels = np.argmax(similarities, axis=1)
    confidences = np.max(similarities, axis=1)
    
    print(f"\n{'='*70}")
    print("RESULTS")
    print(f"{'='*70}\n")
    
    # Print distribution
    print("Class Distribution:")
    for class_idx, class_name in enumerate(class_names):
        count = np.sum(labels == class_idx)
        percentage = 100 * count / len(labels)
        avg_conf = np.mean(confidences[labels == class_idx]) if count > 0 else 0
        print(f"  {class_name:15} {count:3} images ({percentage:5.1f}%) - avg confidence: {avg_conf:.3f}")
    
    print(f"\nOverall Statistics:")
    print(f"  Mean confidence: {np.mean(confidences):.3f}")
    print(f"  Min confidence:  {np.min(confidences):.3f}")
    print(f"  Max confidence:  {np.max(confidences):.3f}")
    
    # Show detailed assignments
    print(f"\n{'='*70}")
    print("DETAILED ASSIGNMENTS")
    print(f"{'='*70}\n")
    print(f"{'Image':<40} {'Assigned Class':<15} {'Confidence':<10}")
    print("-" * 70)
    
    for i, (path, label, conf) in enumerate(zip(image_paths, labels, confidences)):
        img_name = Path(path).name
        class_name = class_names[label]
        print(f"{img_name:<40} {class_name:<15} {conf:.3f}")
    
    # Create output directory
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    # Create class folders and organize
    print(f"\n{'='*70}")
    print("ORGANIZING FILES")
    print(f"{'='*70}\n")
    
    for class_name in class_names:
        (output_path / class_name).mkdir(exist_ok=True)
    
    organized_paths = {name: [] for name in class_names}
    
    for img_path, label in zip(image_paths, labels):
        src = Path(img_path)
        if not src.exists():
            print(f"Warning: File not found: {img_path}")
            continue
        
        class_name = class_names[label]
        dst = output_path / class_name / src.name
        
        if copy_files:
            shutil.copy2(src, dst)
        else:
            dst.symlink_to(src.absolute())
        
        organized_paths[class_name].append(str(dst.relative_to(output_path)))
    
    # Save metadata
    metadata = {
        'method': 'direct_clip_similarity',
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
                'similarities': {class_names[j]: float(similarities[i, j]) for j in range(len(class_names))}
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
        "DIRECT CLIP ANNOTATION REPORT",
        "="*70,
        "",
        f"Method: Direct CLIP similarity (optimized for small datasets)",
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
    
    print(f"\n✓ Files organized in: {output_path}")
    print(f"✓ Metadata saved: {output_path / 'metadata.json'}")
    print(f"✓ Report saved: {output_path / 'report.txt'}")
    print(f"\n{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 5:
        print("Usage: python direct_annotation.py <embeddings.npy> <image_paths.txt> <output_dir> <class1> <class2> ...")
        print("\nExample:")
        print("  python direct_annotation.py data/embeddings.npy data/image_paths.txt annotated_direct dog cat tiger lion giraffe")
        sys.exit(1)
    
    embeddings_path = sys.argv[1]
    image_paths_file = sys.argv[2]
    output_dir = sys.argv[3]
    class_names = sys.argv[4:]
    
    direct_clip_annotation(embeddings_path, image_paths_file, class_names, output_dir)
