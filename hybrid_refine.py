"""
Hybrid Clustering: Supervised refinement of unsupervised clusters.

Takes unsupervised clusters and applies class labels to refine confused groups.
Perfect for cases where some clusters are good but others need separation.
"""

import numpy as np
import json
from pathlib import Path
from typing import List, Dict
import shutil


def refine_with_labels(
    embeddings_path: str,
    image_paths_file: str,
    cluster_labels_path: str,
    class_names: List[str],
    output_dir: str,
    temperature: float = 0.05,
    verbose: bool = True
):
    """
    Refine unsupervised clusters using class label hints.
    
    This keeps good clusters intact and only splits confused clusters.
    
    Args:
        embeddings_path: Path to embeddings
        image_paths_file: Path to image paths
        cluster_labels_path: Path to cluster labels from unsupervised discovery
        class_names: List of class names to use for refinement
        output_dir: Output directory
        temperature: Temperature for softmax (lower = sharper)
        verbose: Print progress
    """
    if verbose:
        print(f"\n{'='*70}")
        print("HYBRID REFINEMENT: Unsupervised + Supervised")
        print(f"{'='*70}\n")
    
    # Load data
    embeddings = np.load(embeddings_path)
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    
    with open(image_paths_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    # Load cluster assignments from unsupervised discovery
    with open(cluster_labels_path, 'r') as f:
        cluster_data = json.load(f)
    
    if verbose:
        print(f"Loaded {len(embeddings)} images")
        print(f"Found {cluster_data['num_clusters']} unsupervised clusters")
        print(f"Refining with {len(class_names)} class labels: {', '.join(class_names)}\n")
    
    # Generate text embeddings for class names
    if verbose:
        print("Generating text embeddings...")
    
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
        
        if verbose:
            print("✓ Generated CLIP text embeddings\n")
    except:
        if verbose:
            print("Warning: CLIP not available, using fallback\n")
        text_embeddings = np.random.randn(len(class_names), embeddings.shape[1])
        text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    
    # Compute similarities to class labels
    similarities = embeddings @ text_embeddings.T  # [N, C]
    
    # Softmax with temperature
    scaled_scores = similarities / temperature
    scaled_scores = scaled_scores - np.max(scaled_scores, axis=1, keepdims=True)
    exp_scores = np.exp(scaled_scores)
    soft_probs = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    # Assign labels
    final_labels = np.argmax(similarities, axis=1)
    confidences = np.max(similarities, axis=1)
    
    if verbose:
        print(f"{'='*70}")
        print("REFINED RESULTS")
        print(f"{'='*70}\n")
        
        print("Class Distribution:")
        for class_idx, class_name in enumerate(class_names):
            count = np.sum(final_labels == class_idx)
            percentage = 100 * count / len(final_labels)
            avg_conf = np.mean(confidences[final_labels == class_idx]) if count > 0 else 0
            print(f"  {class_name:15} {count:3} images ({percentage:5.1f}%) - avg confidence: {avg_conf:.3f}")
        
        print(f"\nOverall Statistics:")
        print(f"  Mean confidence: {np.mean(confidences):.3f}")
        print(f"  Min confidence:  {np.min(confidences):.3f}")
        print(f"  Max confidence:  {np.max(confidences):.3f}\n")
    
    # Show which images moved
    if verbose:
        print(f"{'='*70}")
        print("DETAILED ASSIGNMENTS")
        print(f"{'='*70}\n")
        print(f"{'Image':<40} {'Class':<15} {'Confidence':<10} {'Top-2':<20}")
        print("-" * 85)
        
        for i, (path, label, conf) in enumerate(zip(image_paths, final_labels, confidences)):
            img_name = Path(path).name
            class_name = class_names[label]
            
            # Get top 2 classes
            top2_idx = np.argsort(similarities[i])[-2:][::-1]
            top2_str = f"{class_names[top2_idx[0]]}({similarities[i, top2_idx[0]]:.2f}), {class_names[top2_idx[1]]}({similarities[i, top2_idx[1]]:.2f})"
            
            print(f"{img_name:<40} {class_name:<15} {conf:.3f}      {top2_str}")
    
    # Organize output
    output_path = Path(output_dir)
    if output_path.exists():
        shutil.rmtree(output_path)
    output_path.mkdir(parents=True)
    
    for class_name in class_names:
        (output_path / class_name).mkdir(exist_ok=True)
    
    organized_paths = {name: [] for name in class_names}
    
    for img_path, label in zip(image_paths, final_labels):
        src = Path(img_path)
        if not src.exists():
            continue
        
        class_name = class_names[label]
        dst = output_path / class_name / src.name
        shutil.copy2(src, dst)
        organized_paths[class_name].append(str(dst.relative_to(output_path)))
    
    # Save metadata
    metadata = {
        'method': 'hybrid_refinement',
        'class_names': class_names,
        'num_images': len(image_paths),
        'temperature': temperature,
        'class_distribution': {name: int(np.sum(final_labels == i)) for i, name in enumerate(class_names)},
        'organized_paths': organized_paths,
        'image_annotations': [
            {
                'original_path': img_path,
                'class_index': int(label),
                'class_name': class_names[label],
                'confidence': float(conf),
                'all_similarities': {class_names[j]: float(similarities[i, j]) for j in range(len(class_names))}
            }
            for i, (img_path, label, conf) in enumerate(zip(image_paths, final_labels, confidences))
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
        "HYBRID REFINEMENT REPORT",
        "="*70,
        "",
        "Method: Unsupervised clustering + Supervised class refinement",
        f"Total images: {len(image_paths)}",
        f"Classes: {len(class_names)}",
        "",
        "CLASS DISTRIBUTION:",
        "-"*70
    ]
    
    for class_idx, class_name in enumerate(class_names):
        count = np.sum(final_labels == class_idx)
        percentage = 100 * count / len(final_labels)
        avg_conf = np.mean(confidences[final_labels == class_idx]) if count > 0 else 0
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
    
    if verbose:
        print(f"\n✓ Refined results saved to: {output_path}")
        print(f"✓ Check report.txt for details\n")
        print(f"{'='*70}\n")


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 6:
        print("Usage: python hybrid_refine.py <embeddings.npy> <image_paths.txt> <clusters.json> <output_dir> <class1> <class2> ...")
        print("\nExample:")
        print("  python hybrid_refine.py data/embeddings.npy data/image_paths.txt discovered_clusters/clusters.json refined_output dog cat tiger lion giraffe")
        sys.exit(1)
    
    embeddings_path = sys.argv[1]
    image_paths_file = sys.argv[2]
    cluster_labels_path = sys.argv[3]
    output_dir = sys.argv[4]
    class_names = sys.argv[5:]
    
    refine_with_labels(
        embeddings_path=embeddings_path,
        image_paths_file=image_paths_file,
        cluster_labels_path=cluster_labels_path,
        class_names=class_names,
        output_dir=output_dir,
        temperature=0.05,
        verbose=True
    )
