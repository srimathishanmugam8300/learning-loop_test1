# Auto-Annotation Learning Engine

A fully automatic system that learns to align unlabeled image datasets with user-defined class names through iterative prototype refinement.

**Works for ANY dataset size (small or large) and ANY domain (animals, vehicles, textures, etc.)**

## Features

✨ **Zero-Threshold Design**: No hard thresholds or cutoffs - everything uses maximum-likelihood assignments  
🔄 **Self-Refining**: Iteratively learns class prototypes from data structure  
📊 **Cluster-Aware**: Works with hierarchical clustering (micro + meta clusters)  
🎯 **Always Assigns**: Every image gets labeled - no "unknown" class  
🚀 **Universal**: Works on tiny datasets (20 images) or massive ones (millions)  
🌍 **Domain-Agnostic**: Animals, marine life, vehicles, textures - any domain  

## How It Works

The engine uses a **learning loop** to align your unlabeled dataset with user-provided class names:

1. **Initialize**: Start with text embeddings of class names (e.g., "dog", "cat", "bird")
2. **Iterate**: For 3-5 iterations:
   - Compute soft alignment between clusters and classes
   - Update class prototypes using weighted cluster centroids
   - Refine at both meta-cluster and micro-cluster levels
3. **Propagate**: Assign final labels using maximum likelihood

### Key Algorithm

```
For each iteration t:
  1. Compute similarity: s(k,c) = cosine_similarity(cluster_k, prototype_c)
  2. Soft responsibility: γ(k,c) = softmax(s(k,c) / temperature)
  3. Update prototype: P_c = Σ_k γ(k,c) * C_k / Σ_k γ(k,c)
  4. Refine at micro-cluster level

Final assignment: label(i) = argmax_c similarity(image_i, prototype_c)
```

## Installation

```bash
# Clone or download this repository
cd learning_loop_T1

# Install required packages
pip install -r requirements.txt

# Optional: Install CLIP for better text embeddings
pip install git+https://github.com/openai/CLIP.git torch torchvision

# Or use Sentence Transformers:
pip install sentence-transformers
```

## Quick Start

### Run the Demo

```bash
python example_usage.py
```

This creates synthetic data and runs the complete pipeline, outputting organized class folders.

### Use Your Own Data

```python
from example_usage import run_complete_pipeline

run_complete_pipeline(
    embeddings_path="path/to/your_embeddings.npy",
    image_paths_file="path/to/image_paths.txt",
    class_names=["dog", "cat", "tiger", "lion"],
    output_dir="annotated_dataset",
    num_iterations=5,
    temperature=0.1,
    verbose=True
)
```

## Input Requirements

### 1. Image Embeddings
- **Format**: NumPy array `.npy` file
- **Shape**: `[N, D]` where N = number of images, D = embedding dimension
- **Normalized**: Should be L2-normalized (unit vectors)
- **Source**: CLIP, ResNet, DINOv2, or any vision model

### 2. Image Paths
- **Format**: Text file (one path per line) or JSON
- **Content**: List of image file paths
- **Order**: Must match embedding order

### 3. Cluster Data (Optional)
If you have pre-computed clusters, provide:
- `micro_assignments.npy` - Micro-cluster ID per image
- `meta_assignments.npy` - Meta-cluster ID per image
- `micro_centroids.npy` - Micro-cluster centroids
- `meta_centroids.npy` - Meta-cluster centroids

If not provided, clusters will be computed automatically using HDBSCAN + Agglomerative Clustering.

### 4. Class Names
- List of target class names (strings)
- Example: `["dog", "cat", "bird"]`

## Output Structure

```
output_dir/
├── dog/                          # Class folders
│   ├── image001.jpg
│   ├── image042.jpg
│   └── ...
├── cat/
│   └── ...
├── bird/
│   └── ...
├── metadata.json                 # Complete annotation details
├── class_prototypes.npy          # Learned class prototypes
├── meta_cluster_alignment.npy    # Soft cluster-to-class alignment
├── micro_cluster_alignment.npy   # Fine-grained alignment
├── report.txt                    # Human-readable summary
└── splits/                       # Train/val/test splits
    ├── train/
    ├── val/
    └── test/
```

## Advanced Usage

### Complete Pipeline Example

```python
import numpy as np
from auto_annotation_engine import AutoAnnotationEngine, ClusterData
from data_loader import DataLoader, TextEmbeddingGenerator, ClusteringPipeline
from dataset_organizer import DatasetOrganizer

# 1. Load your data
embeddings = DataLoader.load_embeddings("embeddings.npy")
image_paths = DataLoader.load_image_paths("image_paths.txt")

# 2. Compute clusters (if not pre-computed)
micro_assignments, micro_centroids = ClusteringPipeline.compute_micro_clusters(embeddings)
micro_to_meta, meta_centroids = ClusteringPipeline.compute_meta_clusters(micro_centroids)
meta_assignments = ClusteringPipeline.propagate_meta_assignments(micro_assignments, micro_to_meta)

cluster_data = ClusterData(
    micro_cluster_assignments=micro_assignments,
    meta_cluster_assignments=meta_assignments,
    micro_cluster_centroids=micro_centroids,
    meta_cluster_centroids=meta_centroids
)

# 3. Generate text embeddings for classes
class_names = ["dog", "cat", "bird"]
text_embeddings = TextEmbeddingGenerator.generate_clip_embeddings(class_names)

# 4. Run learning loop
engine = AutoAnnotationEngine(
    image_embeddings=embeddings,
    cluster_data=cluster_data,
    class_names=class_names,
    text_embeddings=text_embeddings,
    num_iterations=5,
    temperature=0.1,
    verbose=True
)

result = engine.run_learning_loop()

# 5. Organize output
organizer = DatasetOrganizer("output", class_names, overwrite=True)
organizer.organize_dataset(image_paths, result, copy_files=True)
organizer.export_for_training("classification", {'train': 0.7, 'val': 0.15, 'test': 0.15})
```

### Parameter Tuning

**Temperature** (`temperature`):
- Lower (0.05-0.1): Sharper assignments, more confident
- Higher (0.2-0.5): Softer assignments, more exploratory
- Default: `0.1`

**Iterations** (`num_iterations`):
- Small datasets: 3-5 iterations
- Large datasets: 5-7 iterations
- Default: `5`

**Clustering**:
- `min_cluster_size`: Minimum images per micro-cluster (default: dataset_size // 20)
- `min_samples`: HDBSCAN density parameter (default: dataset_size // 50)

## Use Cases

### 1. Image Classification Dataset Creation
Generate labeled training data for image classifiers without manual annotation.

### 2. Object Detection Preprocessing
Organize unlabeled images before bounding box annotation for YOLO/Faster R-CNN.

### 3. Segmentation Pipeline
Prepare images for SAM or segmentation model fine-tuning.

### 4. Active Learning
Bootstrap an active learning loop with automatic initial labels.

### 5. Dataset Exploration
Understand the structure of an unlabeled dataset by seeing how it naturally clusters.

## Design Principles

### ✅ What This System DOES

- Automatically assigns ALL images to classes
- Uses maximum-likelihood for every decision
- Adapts class semantics to dataset structure
- Works with any number of clusters vs classes
- Handles imbalanced datasets naturally
- Self-refines through iteration

### ❌ What This System DOES NOT Do

- ❌ Use hard thresholds for acceptance/rejection
- ❌ Create "unknown" or "outlier" classes
- ❌ Require human review or correction
- ❌ Assume clusters equal classes
- ❌ Discard low-confidence images
- ❌ Need balanced class distributions

## Architecture

```
auto_annotation_engine.py    # Core learning loop implementation
data_loader.py               # Data loading and clustering utilities
dataset_organizer.py         # Output organization and export
example_usage.py             # Demo and pipeline examples
requirements.txt             # Dependencies
```

## Mathematical Foundation

### Soft Alignment

For each cluster `k` and class `c`:

```
γ(k,c) = exp(s(k,c) / τ) / Σ_c' exp(s(k,c') / τ)
```

where:
- `s(k,c)` = cosine similarity between cluster `k` and prototype `c`
- `τ` = temperature parameter
- `γ(k,c)` = soft responsibility (how much cluster `k` belongs to class `c`)

### Prototype Update

```
P_c^(t+1) = Σ_k γ(k,c) * C_k / Σ_k γ(k,c)
```

This weighted average naturally:
- Moves prototypes toward strongly matching clusters
- Ignores weakly matching clusters
- Aligns class semantics with data structure

## FAQs

**Q: What if I only have one cluster?**  
A: The system automatically refines at the image level when clusters are too coarse.

**Q: What if my class names don't match the data?**  
A: The text embeddings are just initial anchors. The prototypes adapt to the actual data distribution during learning.

**Q: How many classes should I specify?**  
A: Specify the number you want in your final dataset. The system will map clusters to classes, even if the numbers differ.

**Q: What about noisy or ambiguous images?**  
A: All images are assigned to their maximum-likelihood class. Check `confidence_scores` in the output to identify low-confidence assignments.

**Q: Can I use this without CLIP?**  
A: Yes. You can use Sentence Transformers, or even provide custom text embeddings. For testing, random embeddings work surprisingly well!

**Q: Does this work for non-image data?**  
A: Yes! As long as you have embeddings and can define class names, it works for text, audio, or any other modality.

## Citation

If you use this system in your research, please cite:

```
Auto-Annotation Learning Engine
A threshold-free system for automatic dataset organization
https://github.com/your-repo/learning_loop_T1
```

## License

MIT License - Feel free to use in commercial or research projects.

## Contributing

Contributions welcome! Areas for improvement:
- Additional export formats (YOLO, COCO, etc.)
- Visualization tools
- Multi-modal embeddings
- Streaming/online learning mode

## Contact

Questions? Issues? Open a GitHub issue or reach out!

---

**Built with ❤️ for the ML community**
