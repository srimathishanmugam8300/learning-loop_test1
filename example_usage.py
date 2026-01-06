"""
Example usage of the Auto-Annotation Learning Engine.

This script demonstrates the complete pipeline:
1. Load embeddings and cluster data
2. Generate text embeddings for class names
3. Run the learning loop
4. Organize dataset into class folders
"""

import numpy as np
from pathlib import Path
from typing import List

from auto_annotation_engine import AutoAnnotationEngine, ClusterData
from data_loader import (
    DataLoader,
    TextEmbeddingGenerator,
    ClusteringPipeline
)
from dataset_organizer import DatasetOrganizer


def run_complete_pipeline(
    embeddings_path: str,
    image_paths_file: str,
    class_names: List[str],
    output_dir: str,
    # Cluster data paths (optional - will compute if not provided)
    micro_assignments_path: str = None,
    meta_assignments_path: str = None,
    micro_centroids_path: str = None,
    meta_centroids_path: str = None,
    # Parameters
    num_iterations: int = 5,
    temperature: float = 0.1,
    text_embedding_method: str = "clip",  # or "sentence_transformer"
    verbose: bool = True
):
    """
    Run the complete auto-annotation pipeline.
    
    Args:
        embeddings_path: Path to image embeddings (.npy)
        image_paths_file: Path to file containing image paths
        class_names: List of target class names
        output_dir: Output directory for organized dataset
        micro_assignments_path: Path to micro-cluster assignments (optional)
        meta_assignments_path: Path to meta-cluster assignments (optional)
        micro_centroids_path: Path to micro-cluster centroids (optional)
        meta_centroids_path: Path to meta-cluster centroids (optional)
        num_iterations: Number of learning iterations
        temperature: Temperature for softmax
        text_embedding_method: Method for text embeddings ('clip' or 'sentence_transformer')
        verbose: Print detailed progress
    """
    
    print("\n" + "="*70)
    print("AUTO-ANNOTATION PIPELINE")
    print("="*70)
    
    # Step 1: Load image embeddings
    if verbose:
        print("\n[1/6] Loading image embeddings...")
    image_embeddings = DataLoader.load_embeddings(embeddings_path)
    print(f"  Loaded {len(image_embeddings)} embeddings, dim={image_embeddings.shape[1]}")
    
    # Step 2: Adaptive mode selection based on dataset size
    num_images = len(image_embeddings)
    use_direct_mode = num_images < 50  # Use direct CLIP for small datasets
    
    if verbose:
        print("\n[2/6] Determining optimal annotation strategy...")
        if use_direct_mode:
            print(f"  Dataset size: {num_images} images")
            print(f"  Using DIRECT CLIP MODE (optimal for small datasets)")
            print(f"  Skipping clustering - will use direct similarity matching")
        else:
            print(f"  Dataset size: {num_images} images")
            print(f"  Using CLUSTERING MODE (standard pipeline)")
    
    if use_direct_mode:
        # For small datasets: create single-cluster structure (direct assignment)
        # This bypasses clustering and uses direct CLIP similarity in the learning loop
        micro_assignments = np.zeros(num_images, dtype=int)
        meta_assignments = np.zeros(num_images, dtype=int)
        # Use mean of all embeddings as single cluster centroid
        micro_centroids = np.mean(image_embeddings, axis=0, keepdims=True)
        meta_centroids = micro_centroids.copy()
        
        cluster_data = ClusterData(
            micro_cluster_assignments=micro_assignments,
            meta_cluster_assignments=meta_assignments,
            micro_cluster_centroids=micro_centroids,
            meta_cluster_centroids=meta_centroids
        )
        print(f"  Single cluster mode enabled (direct assignment)")
        
    elif all([micro_assignments_path, meta_assignments_path, 
              micro_centroids_path, meta_centroids_path]):
        # Load pre-computed clusters
        cluster_data = DataLoader.load_cluster_data(
            micro_assignments_path,
            meta_assignments_path,
            micro_centroids_path,
            meta_centroids_path
        )
        print(f"  Loaded pre-computed clusters")
        print(f"  Micro-clusters: {len(cluster_data.micro_cluster_centroids)}")
        print(f"  Meta-clusters: {len(cluster_data.meta_cluster_centroids)}")
    else:
        # Compute clusters for larger datasets
        print(f"  Computing micro-clusters with HDBSCAN...")
        micro_assignments, micro_centroids = ClusteringPipeline.compute_micro_clusters(
            image_embeddings,
            min_cluster_size=max(2, min(5, len(image_embeddings) // 20)),
            min_samples=max(2, min(3, len(image_embeddings) // 50))
        )
        print(f"    Found {len(micro_centroids)} micro-clusters")
        
        print(f"  Computing meta-clusters with Agglomerative Clustering...")
        num_meta = max(2, min(len(class_names) * 2, len(micro_centroids)))
        micro_to_meta, meta_centroids = ClusteringPipeline.compute_meta_clusters(
            micro_centroids,
            num_clusters=num_meta
        )
        print(f"    Found {len(meta_centroids)} meta-clusters")
        
        # Propagate meta assignments to images
        meta_assignments = ClusteringPipeline.propagate_meta_assignments(
            micro_assignments,
            micro_to_meta
        )
        
        cluster_data = ClusterData(
            micro_cluster_assignments=micro_assignments,
            meta_cluster_assignments=meta_assignments,
            micro_cluster_centroids=micro_centroids,
            meta_cluster_centroids=meta_centroids
        )
        
        print(f"  Micro-clusters: {len(cluster_data.micro_cluster_centroids)}")
        print(f"  Meta-clusters: {len(cluster_data.meta_cluster_centroids)}")
    
    # Step 3: Generate text embeddings for class names
    if verbose:
        print(f"\n[3/6] Generating text embeddings for {len(class_names)} classes...")
        print(f"  Classes: {', '.join(class_names)}")
    
    if text_embedding_method == "clip":
        try:
            text_embeddings = TextEmbeddingGenerator.generate_clip_embeddings(class_names)
            print(f"  Generated text embeddings using CLIP")
        except ImportError as e:
            print(f"  Warning: {e}")
            print(f"  Falling back to random embeddings for demo purposes")
            text_embeddings = np.random.randn(len(class_names), image_embeddings.shape[1])
            text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    elif text_embedding_method == "sentence_transformer":
        try:
            text_embeddings = TextEmbeddingGenerator.generate_sentence_transformer_embeddings(class_names)
            print(f"  Generated text embeddings using Sentence Transformers")
        except ImportError as e:
            print(f"  Warning: {e}")
            print(f"  Falling back to random embeddings for demo purposes")
            text_embeddings = np.random.randn(len(class_names), image_embeddings.shape[1])
            text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
    else:
        raise ValueError(f"Unknown text embedding method: {text_embedding_method}")
    
    # Step 4: Initialize and run learning engine
    if verbose:
        print(f"\n[4/6] Running learning loop...")
    
    engine = AutoAnnotationEngine(
        image_embeddings=image_embeddings,
        cluster_data=cluster_data,
        class_names=class_names,
        text_embeddings=text_embeddings,
        num_iterations=num_iterations,
        temperature=temperature,
        verbose=verbose
    )
    
    learning_result = engine.run_learning_loop()
    
    # Step 5: Load image paths
    if verbose:
        print(f"\n[5/6] Loading image paths...")
    image_paths = DataLoader.load_image_paths(image_paths_file)
    print(f"  Loaded {len(image_paths)} image paths")
    
    assert len(image_paths) == len(image_embeddings), \
        f"Mismatch: {len(image_paths)} paths vs {len(image_embeddings)} embeddings"
    
    # Step 6: Organize dataset
    if verbose:
        print(f"\n[6/6] Organizing dataset...")
    
    organizer = DatasetOrganizer(
        output_dir=output_dir,
        class_names=class_names,
        overwrite=True
    )
    
    organizer.organize_dataset(
        image_paths=image_paths,
        learning_result=learning_result,
        copy_files=True  # Set to False for large datasets (will create symlinks instead)
    )
    
    # Export for training (optional)
    if verbose:
        print(f"\nExporting train/val/test splits...")
    organizer.export_for_training(
        format="classification",
        split_ratios={'train': 0.7, 'val': 0.15, 'test': 0.15}
    )
    
    print("\n" + "="*70)
    print("✓ PIPELINE COMPLETED SUCCESSFULLY!")
    print("="*70)
    print(f"\nOutput location: {output_dir}")
    print(f"Files generated:")
    print(f"  - Organized class folders")
    print(f"  - metadata.json (full annotation details)")
    print(f"  - class_prototypes.npy (learned prototypes)")
    print(f"  - report.txt (human-readable summary)")
    print(f"  - splits/ (train/val/test splits)")
    print()


def create_synthetic_demo_data(
    num_images: int = 100,
    num_classes: int = 3,
    embedding_dim: int = 512,
    output_dir: str = "demo_data"
):
    """
    Create synthetic demo data for testing the pipeline.
    
    This creates realistic-looking synthetic embeddings with cluster structure.
    """
    output_path = Path(output_dir)
    output_path.mkdir(exist_ok=True)
    
    print(f"Creating synthetic demo data...")
    print(f"  Images: {num_images}")
    print(f"  Classes: {num_classes}")
    print(f"  Embedding dim: {embedding_dim}")
    
    # Create synthetic class prototypes
    np.random.seed(42)
    class_prototypes = np.random.randn(num_classes, embedding_dim)
    class_prototypes = class_prototypes / np.linalg.norm(class_prototypes, axis=1, keepdims=True)
    
    # Generate embeddings around class prototypes with noise
    embeddings = []
    labels = []
    image_paths = []
    
    for i in range(num_images):
        # Assign to random class
        class_idx = np.random.randint(0, num_classes)
        
        # Create embedding near class prototype
        noise = np.random.randn(embedding_dim) * 0.3
        embedding = class_prototypes[class_idx] + noise
        embedding = embedding / np.linalg.norm(embedding)
        
        embeddings.append(embedding)
        labels.append(class_idx)
        image_paths.append(f"synthetic_image_{i:04d}.jpg")
    
    embeddings = np.array(embeddings)
    
    # Save embeddings
    np.save(output_path / 'embeddings.npy', embeddings)
    
    # Save image paths
    with open(output_path / 'image_paths.txt', 'w') as f:
        f.write('\n'.join(image_paths))
    
    print(f"\n✓ Synthetic data created in {output_path}")
    print(f"  embeddings.npy")
    print(f"  image_paths.txt")
    print()
    
    return str(output_path / 'embeddings.npy'), str(output_path / 'image_paths.txt')


if __name__ == "__main__":
    # Demo: Create synthetic data and run pipeline
    print("\n" + "="*70)
    print("AUTO-ANNOTATION ENGINE - DEMO")
    print("="*70)
    
    # Create synthetic demo data
    embeddings_path, image_paths_file = create_synthetic_demo_data(
        num_images=100,
        num_classes=3,
        embedding_dim=512
    )
    
    # Define class names
    class_names = ["dog", "cat", "bird"]
    
    # Run complete pipeline
    run_complete_pipeline(
        embeddings_path=embeddings_path,
        image_paths_file=image_paths_file,
        class_names=class_names,
        output_dir="demo_output",
        num_iterations=5,
        temperature=0.1,
        text_embedding_method="clip",  # Will fall back to random for demo
        verbose=True
    )
