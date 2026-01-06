"""
Data loading and preprocessing utilities for the Auto-Annotation Engine.
"""

import numpy as np
import json
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
from auto_annotation_engine import ClusterData


class DataLoader:
    """
    Load and prepare data for the Auto-Annotation Engine.
    
    Supports various input formats:
    - NumPy arrays (.npy)
    - JSON files
    - CSV files
    """
    
    @staticmethod
    def load_embeddings(filepath: Union[str, Path]) -> np.ndarray:
        """
        Load image embeddings from file.
        
        Args:
            filepath: Path to embeddings file (.npy or .npz)
            
        Returns:
            [N, D] numpy array of embeddings
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.npy':
            embeddings = np.load(filepath)
        elif filepath.suffix == '.npz':
            data = np.load(filepath)
            # Try common keys
            if 'embeddings' in data:
                embeddings = data['embeddings']
            elif 'features' in data:
                embeddings = data['features']
            else:
                # Use first array
                embeddings = data[list(data.keys())[0]]
        else:
            raise ValueError(f"Unsupported file format: {filepath.suffix}")
        
        return embeddings
    
    @staticmethod
    def load_cluster_data(
        micro_assignments_path: Union[str, Path],
        meta_assignments_path: Union[str, Path],
        micro_centroids_path: Union[str, Path],
        meta_centroids_path: Union[str, Path]
    ) -> ClusterData:
        """
        Load cluster assignments and centroids.
        
        Args:
            micro_assignments_path: Path to micro-cluster assignments
            meta_assignments_path: Path to meta-cluster assignments
            micro_centroids_path: Path to micro-cluster centroids
            meta_centroids_path: Path to meta-cluster centroids
            
        Returns:
            ClusterData object
        """
        micro_assignments = np.load(micro_assignments_path)
        meta_assignments = np.load(meta_assignments_path)
        micro_centroids = np.load(micro_centroids_path)
        meta_centroids = np.load(meta_centroids_path)
        
        return ClusterData(
            micro_cluster_assignments=micro_assignments,
            meta_cluster_assignments=meta_assignments,
            micro_cluster_centroids=micro_centroids,
            meta_cluster_centroids=meta_centroids
        )
    
    @staticmethod
    def load_image_paths(filepath: Union[str, Path]) -> List[str]:
        """
        Load list of image paths.
        
        Args:
            filepath: Path to file containing image paths (one per line)
            
        Returns:
            List of image paths
        """
        filepath = Path(filepath)
        
        if filepath.suffix == '.json':
            with open(filepath, 'r') as f:
                data = json.load(f)
                if isinstance(data, list):
                    return data
                elif 'image_paths' in data:
                    return data['image_paths']
                elif 'paths' in data:
                    return data['paths']
        else:
            # Text file, one path per line
            with open(filepath, 'r') as f:
                return [line.strip() for line in f if line.strip()]
    
    @staticmethod
    def save_embeddings(embeddings: np.ndarray, filepath: Union[str, Path]):
        """Save embeddings to file."""
        np.save(filepath, embeddings)
    
    @staticmethod
    def save_cluster_data(cluster_data: ClusterData, output_dir: Union[str, Path]):
        """Save cluster data to directory."""
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        
        np.save(output_dir / 'micro_assignments.npy', cluster_data.micro_cluster_assignments)
        np.save(output_dir / 'meta_assignments.npy', cluster_data.meta_cluster_assignments)
        np.save(output_dir / 'micro_centroids.npy', cluster_data.micro_cluster_centroids)
        np.save(output_dir / 'meta_centroids.npy', cluster_data.meta_cluster_centroids)


class TextEmbeddingGenerator:
    """
    Generate text embeddings for class names.
    
    Supports multiple embedding methods:
    - CLIP text encoder
    - Sentence transformers
    - Custom embeddings
    """
    
    @staticmethod
    def generate_clip_embeddings(
        class_names: List[str],
        model_name: str = "ViT-B/32"
    ) -> np.ndarray:
        """
        Generate text embeddings using CLIP.
        
        Args:
            class_names: List of class names
            model_name: CLIP model name
            
        Returns:
            [C, D] text embeddings
        """
        try:
            import clip
            import torch
            
            device = "cuda" if torch.cuda.is_available() else "cpu"
            model, _ = clip.load(model_name, device=device)
            
            # Create prompts
            prompts = [f"a photo of a {name}" for name in class_names]
            
            # Tokenize and encode
            with torch.no_grad():
                text_tokens = clip.tokenize(prompts).to(device)
                text_features = model.encode_text(text_tokens)
                text_features = text_features / text_features.norm(dim=-1, keepdim=True)
            
            return text_features.cpu().numpy()
        
        except ImportError:
            raise ImportError(
                "CLIP not installed. Install with: pip install git+https://github.com/openai/CLIP.git"
            )
    
    @staticmethod
    def generate_sentence_transformer_embeddings(
        class_names: List[str],
        model_name: str = "all-MiniLM-L6-v2"
    ) -> np.ndarray:
        """
        Generate text embeddings using Sentence Transformers.
        
        Args:
            class_names: List of class names
            model_name: Model name
            
        Returns:
            [C, D] text embeddings
        """
        try:
            from sentence_transformers import SentenceTransformer
            
            model = SentenceTransformer(model_name)
            
            # Create prompts
            prompts = [f"a photo of a {name}" for name in class_names]
            
            # Encode
            embeddings = model.encode(prompts, convert_to_numpy=True)
            
            # Normalize
            embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
            
            return embeddings
        
        except ImportError:
            raise ImportError(
                "Sentence Transformers not installed. Install with: pip install sentence-transformers"
            )
    
    @staticmethod
    def load_custom_embeddings(filepath: Union[str, Path]) -> np.ndarray:
        """
        Load pre-computed text embeddings.
        
        Args:
            filepath: Path to text embeddings file
            
        Returns:
            [C, D] text embeddings
        """
        return np.load(filepath)


class ClusteringPipeline:
    """
    Perform clustering on embeddings if not already provided.
    """
    
    @staticmethod
    def compute_micro_clusters(
        embeddings: np.ndarray,
        min_cluster_size: int = 5,
        min_samples: int = 3
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute micro-clusters using HDBSCAN.
        
        Args:
            embeddings: [N, D] embeddings
            min_cluster_size: Minimum cluster size
            min_samples: Minimum samples
            
        Returns:
            assignments: [N] cluster assignments (-1 for noise)
            centroids: [M, D] cluster centroids
        """
        try:
            import hdbscan
            
            clusterer = hdbscan.HDBSCAN(
                min_cluster_size=min_cluster_size,
                min_samples=min_samples,
                metric='euclidean'
            )
            
            assignments = clusterer.fit_predict(embeddings)
            
            # Compute centroids (excluding noise points)
            unique_clusters = np.unique(assignments)
            unique_clusters = unique_clusters[unique_clusters >= 0]
            
            centroids = []
            for cluster_id in unique_clusters:
                mask = assignments == cluster_id
                centroid = np.mean(embeddings[mask], axis=0)
                centroids.append(centroid)
            
            centroids = np.array(centroids)
            
            # Assign noise points to nearest cluster
            if -1 in assignments:
                noise_mask = assignments == -1
                noise_embeddings = embeddings[noise_mask]
                
                if len(centroids) > 0:
                    # Assign to nearest cluster
                    similarities = noise_embeddings @ centroids.T
                    nearest_clusters = unique_clusters[np.argmax(similarities, axis=1)]
                    assignments[noise_mask] = nearest_clusters
                else:
                    # All points are noise, create one cluster
                    assignments[:] = 0
                    centroids = np.mean(embeddings, axis=0, keepdims=True)
            
            return assignments, centroids
        
        except ImportError:
            raise ImportError(
                "HDBSCAN not installed. Install with: pip install hdbscan"
            )
    
    @staticmethod
    def compute_meta_clusters(
        micro_centroids: np.ndarray,
        num_clusters: Optional[int] = None
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Compute meta-clusters using Agglomerative Clustering.
        
        Args:
            micro_centroids: [M, D] micro-cluster centroids
            num_clusters: Number of meta-clusters (if None, auto-determine)
            
        Returns:
            assignments: [M] meta-cluster assignments
            centroids: [K, D] meta-cluster centroids
        """
        try:
            from sklearn.cluster import AgglomerativeClustering
            
            if num_clusters is None:
                # Auto-determine: use sqrt(M) as heuristic
                num_clusters = max(2, int(np.sqrt(len(micro_centroids))))
            
            # Ensure num_clusters doesn't exceed number of micro-clusters
            num_clusters = min(num_clusters, len(micro_centroids))
            
            clusterer = AgglomerativeClustering(
                n_clusters=num_clusters,
                linkage='average'
            )
            
            assignments = clusterer.fit_predict(micro_centroids)
            
            # Compute centroids
            unique_clusters = np.unique(assignments)
            centroids = []
            for cluster_id in unique_clusters:
                mask = assignments == cluster_id
                centroid = np.mean(micro_centroids[mask], axis=0)
                centroids.append(centroid)
            
            centroids = np.array(centroids)
            
            return assignments, centroids
        
        except ImportError:
            raise ImportError(
                "scikit-learn not installed. Install with: pip install scikit-learn"
            )
    
    @staticmethod
    def propagate_meta_assignments(
        micro_assignments: np.ndarray,
        micro_to_meta: np.ndarray
    ) -> np.ndarray:
        """
        Propagate meta-cluster assignments to images.
        
        Args:
            micro_assignments: [N] micro-cluster per image
            micro_to_meta: [M] meta-cluster per micro-cluster
            
        Returns:
            [N] meta-cluster per image
        """
        return micro_to_meta[micro_assignments]
