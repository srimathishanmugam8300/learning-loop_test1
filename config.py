"""
Configuration settings for the Auto-Annotation Engine.
"""

from dataclasses import dataclass, field
from typing import List, Optional, Dict


@dataclass
class AutoAnnotationConfig:
    """Configuration for auto-annotation pipeline."""
    
    # Learning parameters
    num_iterations: int = 5
    """Number of learning loop iterations (3-7 recommended)"""
    
    temperature: float = 0.1
    """Temperature for softmax (lower = sharper, 0.05-0.5)"""
    
    # Clustering parameters
    min_cluster_size: Optional[int] = None
    """Minimum micro-cluster size (auto if None: dataset_size // 20)"""
    
    min_samples: Optional[int] = None
    """HDBSCAN min_samples (auto if None: dataset_size // 50)"""
    
    num_meta_clusters: Optional[int] = None
    """Number of meta-clusters (auto if None: sqrt(num_micro_clusters))"""
    
    # Text embedding settings
    text_embedding_method: str = "clip"
    """Method for text embeddings: 'clip', 'sentence_transformer', or 'custom'"""
    
    clip_model_name: str = "ViT-B/32"
    """CLIP model name if using CLIP"""
    
    sentence_transformer_model: str = "all-MiniLM-L6-v2"
    """Sentence transformer model if using that method"""
    
    # Output settings
    copy_files: bool = True
    """Whether to copy image files (False creates symlinks)"""
    
    create_splits: bool = True
    """Whether to create train/val/test splits"""
    
    split_ratios: Dict[str, float] = field(default_factory=lambda: {
        'train': 0.7,
        'val': 0.15,
        'test': 0.15
    })
    """Train/val/test split ratios"""
    
    export_format: str = "classification"
    """Export format: 'classification', 'yolo', 'coco'"""
    
    # Verbosity
    verbose: bool = True
    """Print detailed progress information"""
    
    # Advanced options
    enable_image_level_refinement: bool = True
    """Enable image-level refinement when clusters are too coarse"""
    
    normalize_embeddings: bool = True
    """Ensure embeddings are L2-normalized"""
    
    random_seed: int = 42
    """Random seed for reproducibility"""
    
    def validate(self):
        """Validate configuration parameters."""
        assert self.num_iterations >= 1, "num_iterations must be >= 1"
        assert self.temperature > 0, "temperature must be > 0"
        assert self.text_embedding_method in ["clip", "sentence_transformer", "custom"]
        assert self.export_format in ["classification", "yolo", "coco"]
        
        # Validate split ratios
        if self.create_splits:
            total_ratio = sum(self.split_ratios.values())
            assert abs(total_ratio - 1.0) < 0.01, "Split ratios must sum to 1.0"


# Preset configurations for common scenarios

SMALL_DATASET_CONFIG = AutoAnnotationConfig(
    num_iterations=3,
    temperature=0.15,
    min_cluster_size=3,
    min_samples=2,
    enable_image_level_refinement=True
)
"""Optimized for datasets with < 100 images"""

MEDIUM_DATASET_CONFIG = AutoAnnotationConfig(
    num_iterations=5,
    temperature=0.1,
    min_cluster_size=None,  # Auto
    min_samples=None,  # Auto
)
"""Optimized for datasets with 100-10,000 images"""

LARGE_DATASET_CONFIG = AutoAnnotationConfig(
    num_iterations=7,
    temperature=0.1,
    min_cluster_size=None,  # Auto
    min_samples=None,  # Auto
    copy_files=False,  # Use symlinks for speed
    enable_image_level_refinement=False
)
"""Optimized for datasets with > 10,000 images"""

HIGH_PRECISION_CONFIG = AutoAnnotationConfig(
    num_iterations=7,
    temperature=0.05,  # Sharper assignments
    min_cluster_size=None,
    min_samples=None
)
"""For applications requiring high confidence assignments"""

EXPLORATORY_CONFIG = AutoAnnotationConfig(
    num_iterations=5,
    temperature=0.3,  # Softer assignments
    min_cluster_size=None,
    min_samples=None
)
"""For exploring dataset structure with softer assignments"""


def get_config_for_dataset_size(num_images: int) -> AutoAnnotationConfig:
    """
    Get recommended configuration based on dataset size.
    
    Args:
        num_images: Number of images in dataset
        
    Returns:
        Recommended AutoAnnotationConfig
    """
    if num_images < 100:
        return SMALL_DATASET_CONFIG
    elif num_images < 10000:
        return MEDIUM_DATASET_CONFIG
    else:
        return LARGE_DATASET_CONFIG


def load_config_from_dict(config_dict: Dict) -> AutoAnnotationConfig:
    """
    Load configuration from dictionary.
    
    Args:
        config_dict: Dictionary with config parameters
        
    Returns:
        AutoAnnotationConfig instance
    """
    return AutoAnnotationConfig(**config_dict)


def save_config_to_dict(config: AutoAnnotationConfig) -> Dict:
    """
    Save configuration to dictionary.
    
    Args:
        config: AutoAnnotationConfig instance
        
    Returns:
        Dictionary representation
    """
    from dataclasses import asdict
    return asdict(config)
