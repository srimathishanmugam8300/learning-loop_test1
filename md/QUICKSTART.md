# Quick Start Guide - Auto-Annotation Learning Engine

## 🚀 5-Minute Quick Start

### Option 1: Run the Demo (Fastest)

```bash
# Install dependencies
pip install numpy scikit-learn hdbscan

# Run demo with synthetic data
python main.py --demo

# Output will be in demo_output/
```

### Option 2: Use Your Own Data

```bash
# 1. Prepare your data files:
#    - embeddings.npy: [N, D] numpy array of image embeddings
#    - image_paths.txt: List of image paths (one per line)

# 2. Run the engine
python main.py \
  --embeddings path/to/embeddings.npy \
  --image-paths path/to/image_paths.txt \
  --classes dog cat bird tiger \
  --output my_annotated_dataset

# 3. Check results
cat my_annotated_dataset/report.txt
```

## 📋 Step-by-Step Tutorial

### Step 1: Install Dependencies

```bash
# Required
pip install numpy scikit-learn hdbscan

# Optional (for better text embeddings)
pip install git+https://github.com/openai/CLIP.git torch torchvision
# OR
pip install sentence-transformers
```

### Step 2: Prepare Your Data

You need two files:

**embeddings.npy** - Image embeddings from any vision model:
```python
import numpy as np

# Example: Load embeddings you extracted from CLIP, ResNet, etc.
embeddings = np.load('your_embeddings.npy')
print(embeddings.shape)  # Should be [N, D] where N=num_images, D=embedding_dim
```

**image_paths.txt** - List of image paths:
```
/path/to/image001.jpg
/path/to/image002.jpg
/path/to/image003.jpg
...
```

### Step 3: Run the Pipeline

**Simple command:**
```bash
python main.py \
  --embeddings embeddings.npy \
  --image-paths image_paths.txt \
  --classes dog cat bird \
  --output annotated_dataset
```

**With more control:**
```bash
python main.py \
  --embeddings embeddings.npy \
  --image-paths image_paths.txt \
  --classes dog cat bird tiger lion \
  --output annotated_dataset \
  --iterations 7 \
  --temperature 0.05 \
  --text-embedding-method clip
```

**For small datasets (<100 images):**
```bash
python main.py \
  --embeddings embeddings.npy \
  --image-paths image_paths.txt \
  --classes class1 class2 class3 \
  --preset small \
  --output results
```

**For large datasets (>10,000 images):**
```bash
python main.py \
  --embeddings embeddings.npy \
  --image-paths image_paths.txt \
  --classes class1 class2 class3 \
  --preset large \
  --output results
```

### Step 4: Review Results

```bash
# Check the summary
cat annotated_dataset/report.txt

# View metadata
cat annotated_dataset/metadata.json

# See organized files
ls annotated_dataset/
# Output:
#   dog/
#   cat/
#   bird/
#   metadata.json
#   report.txt
#   class_prototypes.npy
#   splits/
```

## 💡 Common Use Cases

### Use Case 1: You Have Pre-extracted Embeddings

```python
from example_usage import run_complete_pipeline

run_complete_pipeline(
    embeddings_path="clip_embeddings.npy",
    image_paths_file="images.txt",
    class_names=["car", "truck", "bus", "motorcycle"],
    output_dir="vehicles_dataset",
    num_iterations=5,
    temperature=0.1
)
```

### Use Case 2: You Have Pre-computed Clusters

```bash
python main.py \
  --embeddings embeddings.npy \
  --image-paths image_paths.txt \
  --classes dog cat bird \
  --micro-assignments micro_clusters.npy \
  --meta-assignments meta_clusters.npy \
  --micro-centroids micro_centroids.npy \
  --meta-centroids meta_centroids.npy \
  --output results
```

### Use Case 3: Python API for Custom Pipeline

```python
import numpy as np
from auto_annotation_engine import AutoAnnotationEngine, ClusterData
from data_loader import TextEmbeddingGenerator, ClusteringPipeline
from dataset_organizer import DatasetOrganizer

# Load your data
embeddings = np.load('embeddings.npy')
image_paths = open('image_paths.txt').read().splitlines()

# Compute clusters
micro_assignments, micro_centroids = ClusteringPipeline.compute_micro_clusters(embeddings)
micro_to_meta, meta_centroids = ClusteringPipeline.compute_meta_clusters(micro_centroids)
meta_assignments = ClusteringPipeline.propagate_meta_assignments(micro_assignments, micro_to_meta)

cluster_data = ClusterData(
    micro_cluster_assignments=micro_assignments,
    meta_cluster_assignments=meta_assignments,
    micro_cluster_centroids=micro_centroids,
    meta_cluster_centroids=meta_centroids
)

# Generate text embeddings
class_names = ["dog", "cat", "bird"]
text_embeddings = TextEmbeddingGenerator.generate_clip_embeddings(class_names)

# Run learning loop
engine = AutoAnnotationEngine(
    image_embeddings=embeddings,
    cluster_data=cluster_data,
    class_names=class_names,
    text_embeddings=text_embeddings,
    num_iterations=5,
    temperature=0.1
)

result = engine.run_learning_loop()

# Organize output
organizer = DatasetOrganizer("output", class_names, overwrite=True)
organizer.organize_dataset(image_paths, result, copy_files=True)
organizer.export_for_training("classification")
```

## 🔧 Parameter Tuning

### Temperature
- **0.05**: Very sharp assignments (high confidence)
- **0.1**: Balanced (recommended default)
- **0.3**: Softer assignments (exploratory)

### Iterations
- **3**: Fast, for well-separated data
- **5**: Balanced (recommended default)
- **7**: More refinement, for complex data

### Dataset Size Presets
- **--preset small**: Optimized for <100 images
- **--preset medium**: Optimized for 100-10,000 images
- **--preset large**: Optimized for >10,000 images

## 📊 Understanding Output

### Directory Structure
```
output/
├── class_1/              # Images assigned to class_1
│   ├── img001.jpg
│   └── img003.jpg
├── class_2/              # Images assigned to class_2
│   └── img002.jpg
├── metadata.json         # Full annotation details
├── report.txt           # Human-readable summary
├── class_prototypes.npy # Learned class representations
└── splits/              # Train/val/test splits
    ├── train/
    ├── val/
    └── test/
```

### Metadata JSON Structure
```json
{
  "class_names": ["dog", "cat", "bird"],
  "num_images": 100,
  "class_distribution": {
    "dog": 35,
    "cat": 40,
    "bird": 25
  },
  "image_annotations": [
    {
      "original_path": "img001.jpg",
      "class_name": "dog",
      "confidence": 0.85
    }
  ],
  "statistics": {
    "mean_confidence": 0.78,
    "std_confidence": 0.12
  }
}
```

## 🎯 Next Steps

1. **Review low-confidence assignments**
   ```python
   import json
   metadata = json.load(open('output/metadata.json'))
   low_conf = [ann for ann in metadata['image_annotations'] if ann['confidence'] < 0.5]
   ```

2. **Use for training**
   ```bash
   # The output is ready for PyTorch, TensorFlow, etc.
   # Use splits/train/, splits/val/, splits/test/
   ```

3. **Active learning loop**
   - Use low-confidence images for manual review
   - Add corrected labels
   - Re-run with updated embeddings

## ❓ Troubleshooting

**Q: "ModuleNotFoundError: No module named 'hdbscan'"**
```bash
pip install hdbscan
```

**Q: Getting random-looking results?**
- Check that embeddings are meaningful (use CLIP, DINOv2, etc.)
- Try different temperature values
- Increase iterations

**Q: All images going to one class?**
- Your classes might be too similar
- Try more specific class names
- Check if embeddings capture the differences

**Q: Low confidence scores?**
- Normal for ambiguous data
- Try sharper temperature (0.05)
- Check if class names match your data

## 📚 Additional Resources

- Full documentation: See README.md
- API reference: See docstrings in source files
- Examples: See example_usage.py
- Tests: See test_engine.py

---

**Ready to annotate your dataset? Start with `python main.py --demo`!**
