"""
Unsupervised Image Clustering and Prototype Discovery

Discovers natural groupings in unlabeled datasets without predefined classes.
Works well with small, imbalanced datasets by preserving fine-grained distinctions.

Key principles:
- Discovers clusters from data structure, not forced classes
- Allows small clusters (even size 1-2)
- Prefers separation over merging
- Identifies representative prototypes per cluster
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Tuple, Optional
import json
import shutil


class PrototypeDiscovery:
    """
    Unsupervised clustering and prototype discovery for small datasets.
    
    Discovers natural groupings and representative prototypes without
    requiring predefined class names.
    """
    
    def __init__(
        self,
        embeddings: np.ndarray,
        image_paths: List[str],
        min_cluster_size: int = 1,
        allow_singletons: bool = True,
        separation_threshold: float = 0.7,
        verbose: bool = True
    ):
        """
        Initialize prototype discovery.
        
        Args:
            embeddings: [N, D] image embeddings
            image_paths: List of image paths
            min_cluster_size: Minimum cluster size (1 = allow singletons)
            allow_singletons: Allow clusters of size 1
            separation_threshold: Similarity threshold for separation (lower = more clusters)
            verbose: Print progress
        """
        self.embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        self.image_paths = image_paths
        self.min_cluster_size = min_cluster_size
        self.allow_singletons = allow_singletons
        self.separation_threshold = separation_threshold
        self.verbose = verbose
        
        self.num_images = len(embeddings)
        self.clusters = None
        self.prototypes = None
        self.cluster_labels = None
    
    def discover_clusters(self) -> Dict:
        """
        Discover natural clusters in the dataset.
        
        Uses aggressive hierarchical clustering that preserves fine-grained distinctions.
        
        Returns:
            Dictionary with cluster information
        """
        if self.verbose:
            print(f"\n{'='*70}")
            print("UNSUPERVISED PROTOTYPE DISCOVERY")
            print(f"{'='*70}\n")
            print(f"Dataset: {self.num_images} images")
            print(f"Strategy: Fine-grained clustering (preserves small groups)")
            print(f"Min cluster size: {self.min_cluster_size}")
            print(f"Separation threshold: {self.separation_threshold}")
            print()
        
        # Compute pairwise similarities
        similarity_matrix = self.embeddings @ self.embeddings.T
        
        # Use hierarchical clustering with aggressive separation
        from sklearn.cluster import AgglomerativeClustering
        
        # Determine number of clusters adaptively
        # For small datasets, allow more clusters (fine-grained)
        if self.num_images < 30:
            # Aggressive: aim for ~60-70% as many clusters as images
            n_clusters = max(2, int(self.num_images * 0.6))
        else:
            # For larger datasets, use square root heuristic
            n_clusters = max(2, int(np.sqrt(self.num_images)))
        
        # Convert similarity to distance
        distance_matrix = 1 - similarity_matrix
        
        clusterer = AgglomerativeClustering(
            n_clusters=n_clusters,
            metric='precomputed',
            linkage='average'
        )
        
        self.cluster_labels = clusterer.fit_predict(distance_matrix)
        
        if self.verbose:
            print(f"Initial clustering: Found {len(np.unique(self.cluster_labels))} clusters")
        
        # Refine: split clusters with high variance
        self.cluster_labels = self._refine_clusters(similarity_matrix)
        
        unique_clusters = np.unique(self.cluster_labels)
        
        if self.verbose:
            print(f"After refinement: {len(unique_clusters)} clusters\n")
        
        # Build cluster information
        self.clusters = {}
        self.prototypes = {}
        
        for cluster_id in unique_clusters:
            cluster_mask = self.cluster_labels == cluster_id
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = self.embeddings[cluster_mask]
            cluster_paths = [self.image_paths[i] for i in cluster_indices]
            
            # Find prototype (most central image)
            centroid = np.mean(cluster_embeddings, axis=0)
            centroid = centroid / np.linalg.norm(centroid)
            
            similarities_to_centroid = cluster_embeddings @ centroid
            prototype_idx_local = np.argmax(similarities_to_centroid)
            prototype_idx_global = cluster_indices[prototype_idx_local]
            
            # Compute intra-cluster cohesion
            pairwise_sims = cluster_embeddings @ cluster_embeddings.T
            cohesion = np.mean(pairwise_sims)
            
            self.clusters[int(cluster_id)] = {
                'size': len(cluster_indices),
                'indices': cluster_indices.tolist(),
                'paths': cluster_paths,
                'prototype_idx': int(prototype_idx_global),
                'prototype_path': self.image_paths[prototype_idx_global],
                'centroid': centroid,
                'cohesion': float(cohesion)
            }
            
            self.prototypes[int(cluster_id)] = {
                'image_path': self.image_paths[prototype_idx_global],
                'embedding': self.embeddings[prototype_idx_global]
            }
        
        # Generate cluster descriptions
        self._generate_descriptions()
        
        # Print summary
        self._print_summary()
        
        return {
            'num_clusters': len(unique_clusters),
            'clusters': self.clusters,
            'prototypes': self.prototypes,
            'cluster_labels': self.cluster_labels.tolist()
        }
    
    def _refine_clusters(self, similarity_matrix: np.ndarray) -> np.ndarray:
        """
        Refine clusters by splitting those with high variance.
        
        Args:
            similarity_matrix: [N, N] similarity matrix
            
        Returns:
            Refined cluster labels
        """
        labels = self.cluster_labels.copy()
        next_label = np.max(labels) + 1
        
        unique_clusters = np.unique(labels)
        
        for cluster_id in unique_clusters:
            cluster_mask = labels == cluster_id
            cluster_size = np.sum(cluster_mask)
            
            if cluster_size <= 2:
                continue  # Don't split tiny clusters
            
            cluster_indices = np.where(cluster_mask)[0]
            cluster_embeddings = self.embeddings[cluster_mask]
            
            # Compute intra-cluster similarities
            cluster_sim_matrix = cluster_embeddings @ cluster_embeddings.T
            avg_sim = np.mean(cluster_sim_matrix)
            
            # If similarity is low, split the cluster
            if avg_sim < self.separation_threshold:
                # Re-cluster within this cluster
                from sklearn.cluster import AgglomerativeClustering
                
                n_sub = min(3, cluster_size // 2 + 1)
                distance_sub = 1 - cluster_sim_matrix
                
                sub_clusterer = AgglomerativeClustering(
                    n_clusters=n_sub,
                    metric='precomputed',
                    linkage='average'
                )
                
                sub_labels = sub_clusterer.fit_predict(distance_sub)
                
                # Assign new labels
                for sub_id in np.unique(sub_labels):
                    if sub_id == 0:
                        continue  # Keep first sub-cluster with original ID
                    
                    sub_mask = sub_labels == sub_id
                    global_indices = cluster_indices[sub_mask]
                    labels[global_indices] = next_label
                    next_label += 1
        
        return labels
    
    def _generate_descriptions(self):
        """Generate semantic descriptions for each cluster."""
        # Compute inter-cluster distances to find distinctive features
        cluster_ids = sorted(self.clusters.keys())
        
        for cluster_id in cluster_ids:
            cluster_info = self.clusters[cluster_id]
            centroid = cluster_info['centroid']
            
            # Find most different cluster
            max_distance = 0
            most_different = None
            
            for other_id in cluster_ids:
                if other_id == cluster_id:
                    continue
                
                other_centroid = self.clusters[other_id]['centroid']
                distance = 1 - (centroid @ other_centroid)
                
                if distance > max_distance:
                    max_distance = distance
                    most_different = other_id
            
            # Generate description
            size = cluster_info['size']
            cohesion = cluster_info['cohesion']
            
            if size == 1:
                desc = "Unique/distinct image - no close matches in dataset"
            elif size == 2:
                desc = "Small cluster (pair) - visually similar duo"
            elif cohesion > 0.85:
                desc = f"Highly cohesive group (similarity: {cohesion:.2f}) - tight visual consistency"
            elif cohesion > 0.75:
                desc = f"Moderate cohesion (similarity: {cohesion:.2f}) - related visual features"
            else:
                desc = f"Loose grouping (similarity: {cohesion:.2f}) - diverse but related"
            
            if most_different is not None:
                desc += f" | Most different from Cluster {most_different}"
            
            cluster_info['description'] = desc
    
    def _print_summary(self):
        """Print cluster summary."""
        print(f"{'='*70}")
        print("DISCOVERED CLUSTERS")
        print(f"{'='*70}\n")
        
        cluster_ids = sorted(self.clusters.keys())
        
        for cluster_id in cluster_ids:
            info = self.clusters[cluster_id]
            print(f"Cluster {cluster_id}:")
            print(f"  Size: {info['size']} images")
            print(f"  Cohesion: {info['cohesion']:.3f}")
            print(f"  Prototype: {Path(info['prototype_path']).name}")
            print(f"  Description: {info['description']}")
            print(f"  Members: {', '.join([Path(p).name for p in info['paths'][:5]])}")
            if len(info['paths']) > 5:
                print(f"            ... and {len(info['paths']) - 5} more")
            print()
        
        # Statistics
        sizes = [self.clusters[cid]['size'] for cid in cluster_ids]
        print(f"{'='*70}")
        print("STATISTICS")
        print(f"{'='*70}")
        print(f"Total clusters: {len(cluster_ids)}")
        print(f"Cluster sizes: min={min(sizes)}, max={max(sizes)}, mean={np.mean(sizes):.1f}")
        print(f"Singletons: {sum(1 for s in sizes if s == 1)}")
        print(f"Small (2-3): {sum(1 for s in sizes if 2 <= s <= 3)}")
        print(f"Medium (4-10): {sum(1 for s in sizes if 4 <= s <= 10)}")
        print(f"Large (>10): {sum(1 for s in sizes if s > 10)}")
        print(f"\n{'='*70}\n")
    
    def save_results(self, output_dir: str, copy_files: bool = True):
        """
        Save discovered clusters to directory.
        
        Args:
            output_dir: Output directory
            copy_files: Whether to copy image files
        """
        output_path = Path(output_dir)
        if output_path.exists():
            shutil.rmtree(output_path)
        output_path.mkdir(parents=True)
        
        # Create cluster folders
        for cluster_id, info in self.clusters.items():
            cluster_dir = output_path / f"cluster_{cluster_id}"
            cluster_dir.mkdir()
            
            # Copy/link images
            for img_path in info['paths']:
                src = Path(img_path)
                if not src.exists():
                    continue
                
                dst = cluster_dir / src.name
                if copy_files:
                    shutil.copy2(src, dst)
                else:
                    dst.symlink_to(src.absolute())
            
            # Mark prototype
            proto_src = Path(info['prototype_path'])
            if proto_src.exists():
                proto_dst = cluster_dir / f"PROTOTYPE_{proto_src.name}"
                if copy_files:
                    shutil.copy2(proto_src, proto_dst)
        
        # Save metadata
        metadata = {
            'method': 'unsupervised_prototype_discovery',
            'num_images': self.num_images,
            'num_clusters': len(self.clusters),
            'min_cluster_size': self.min_cluster_size,
            'separation_threshold': self.separation_threshold,
            'clusters': {
                str(cid): {
                    'size': info['size'],
                    'cohesion': info['cohesion'],
                    'description': info['description'],
                    'prototype': Path(info['prototype_path']).name,
                    'members': [Path(p).name for p in info['paths']]
                }
                for cid, info in self.clusters.items()
            }
        }
        
        with open(output_path / 'clusters.json', 'w') as f:
            json.dump(metadata, f, indent=2)
        
        # Save report
        report_lines = [
            "="*70,
            "UNSUPERVISED PROTOTYPE DISCOVERY REPORT",
            "="*70,
            "",
            f"Total images: {self.num_images}",
            f"Discovered clusters: {len(self.clusters)}",
            f"Method: Fine-grained hierarchical clustering",
            "",
            "="*70,
            "CLUSTER DETAILS",
            "="*70,
            ""
        ]
        
        for cluster_id in sorted(self.clusters.keys()):
            info = self.clusters[cluster_id]
            report_lines.extend([
                f"Cluster {cluster_id}:",
                f"  Size: {info['size']} images",
                f"  Cohesion: {info['cohesion']:.3f}",
                f"  Prototype: {Path(info['prototype_path']).name}",
                f"  Description: {info['description']}",
                ""
            ])
        
        with open(output_path / 'report.txt', 'w') as f:
            f.write('\n'.join(report_lines))
        
        print(f"✓ Results saved to: {output_path}")
        print(f"✓ Clusters organized in cluster_* folders")
        print(f"✓ Prototypes marked with PROTOTYPE_ prefix")


def discover_prototypes(
    embeddings_path: str,
    image_paths_file: str,
    output_dir: str,
    min_cluster_size: int = 1,
    separation_threshold: float = 0.7,
    verbose: bool = True
):
    """
    Discover natural clusters and prototypes from unlabeled images.
    
    Args:
        embeddings_path: Path to embeddings (.npy)
        image_paths_file: Path to image paths file
        output_dir: Output directory
        min_cluster_size: Minimum cluster size (1 = allow singletons)
        separation_threshold: Lower = more clusters (0.6-0.8 recommended)
        verbose: Print progress
    """
    # Load data
    embeddings = np.load(embeddings_path)
    with open(image_paths_file, 'r') as f:
        image_paths = [line.strip() for line in f if line.strip()]
    
    # Discover clusters
    discoverer = PrototypeDiscovery(
        embeddings=embeddings,
        image_paths=image_paths,
        min_cluster_size=min_cluster_size,
        allow_singletons=True,
        separation_threshold=separation_threshold,
        verbose=verbose
    )
    
    result = discoverer.discover_clusters()
    
    # Save results
    discoverer.save_results(output_dir)
    
    return result


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 4:
        print("Usage: python unsupervised_discovery.py <embeddings.npy> <image_paths.txt> <output_dir> [separation_threshold]")
        print("\nExample:")
        print("  python unsupervised_discovery.py data/embeddings.npy data/image_paths.txt discovered_clusters 0.7")
        print("\nSeparation threshold:")
        print("  0.6 = More clusters (fine-grained)")
        print("  0.7 = Balanced (recommended)")
        print("  0.8 = Fewer clusters (coarse)")
        sys.exit(1)
    
    embeddings_path = sys.argv[1]
    image_paths_file = sys.argv[2]
    output_dir = sys.argv[3]
    separation_threshold = float(sys.argv[4]) if len(sys.argv) > 4 else 0.7
    
    discover_prototypes(
        embeddings_path=embeddings_path,
        image_paths_file=image_paths_file,
        output_dir=output_dir,
        separation_threshold=separation_threshold
    )
