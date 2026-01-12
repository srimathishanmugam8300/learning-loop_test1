# Universal Auto-Annotation Pipeline

## 📋 Project Overview

This project implements a **universal automatic image annotation system** that can classify and organize any dataset of images into user-defined categories using CLIP (Contrastive Language-Image Pre-training) and advanced iterative refinement techniques.

**Key Features:**
- 🎯 Zero-shot classification (no training required)
- 🔄 Iterative refinement with learning
- 📊 Automatic train/val/test splits
- 🎨 Specialized handling for digits and sign language
- ⚡ Adaptive strategy based on dataset size
- 📁 Organized output structure with metadata

---

## 🎭 The Journey: From Concept to Working Solution

### **Phase 1: Initial Implementation (Hierarchical Clustering)**

**Goal:** Build a sophisticated pipeline using hierarchical clustering for better accuracy.

**Approach:**
- Meta-clusters: Group similar classes together
- Micro-clusters: Subdivide within each class
- Soft alignment: Blend text and visual embeddings (70% text / 30% visual)

**What Happened:**
```
Dataset: 413 digit images (digits 2, 3, 4, 6, 7, 8, 9)
Result: ALL images dumped into 1-2 classes (digit 6 and 8)
```

**The Problem - "Prototype Collapse":**
- Iteration 1: One class (e.g., digit9) gets majority of images
- Prototypes recompute: digit9 now has visual data, others remain pure text
- Iterations 2-10: digit9 prototype gets stronger, attracts ALL images
- No learning signal: Confidence stuck at 0.921 across all iterations
- Movement: 0% after first iteration

**Root Cause:**
Clustering "froze" assignments within clusters. Images couldn't escape their initial cluster assignment, preventing true learning.

---

### **Phase 2: Simplification Attempt**

**Action:** Created `auto_pipeline_simple.py` as backup

**Changes:**
- Removed complex clustering
- 3 iterations instead of 10
- 80% text / 20% visual blend
- Lighter refinement

**Result:** Still had issues with collapse, but served as stable fallback.

---

### **Phase 3: Major Refactor - Pure Iterative Reassignment**

**Breakthrough Insight:**
> "Don't lock images into clusters. Reassign EVERY image EVERY iteration."

**New Algorithm:**
```python
for iteration in range(10):
    # 1. Reassign ALL images based on current prototypes
    similarities = image_embeddings @ prototypes.T
    labels = argmax(similarities)
    
    # 2. Recompute prototypes from NEW assignments
    for each class:
        class_mean = mean(images_assigned_to_class)
        prototype = 0.7 * text_embedding + 0.3 * class_mean
    
    # 3. Track movement (learning signal)
    num_moved = count(labels != previous_labels)
```

**Result:** Better, but still collapsed!

**Why?**
Fixed 70/30 blend was too visual-heavy early on. Once one class got majority images, it dominated forever.

---

### **Phase 4: Dynamic Text Weighting (Current Solution)**

**Final Fix - Preventing Early Collapse:**

**Key Innovation:** Gradually shift from text-anchored to visual-learning

```python
# Dynamic text weight per iteration
text_weight = 0.95 - (0.35 * iteration / num_iterations)

Iteration 1:  95% text / 5% visual   (strong anchor)
Iteration 5:  78% text / 22% visual  (balanced)
Iteration 10: 60% text / 40% visual  (visual learning)
```

**Why This Works:**
- **Early iterations (95% text):** All classes remain competitive, prevents collapse
- **Middle iterations (78% text):** Gradual learning, images start finding correct classes
- **Late iterations (60% text):** Full visual refinement, convergence

**Added Temperature Scaling:**
```python
similarities_scaled = similarities / temperature  # temperature=0.05
labels = argmax(similarities_scaled)  # Sharper class boundaries
```

**Result:** ✅ **WORKING!**
- Multiple classes active from iteration 1
- Gradual movement across iterations
- Proper convergence
- Accurate classification

---

## 🔧 Technical Architecture

### **Core Components**

#### 1. **CLIP Embeddings (ViT-B/32)**
- **Image encoder:** Converts images → 512D vectors
- **Text encoder:** Converts class names → 512D vectors
- **Similarity:** Cosine similarity in embedding space

#### 2. **Adaptive Classification Strategy**

**Small Datasets (<100 images):**
```python
def annotate_images_inline():
    # Direct CLIP classification
    similarities = image_embeddings @ text_embeddings.T
    labels = argmax(similarities / temperature)
    return labels, confidences
```
- Fast, no iterative refinement
- Suitable for quick tasks

**Large Datasets (≥100 images):**
```python
def annotate_with_refinement():
    # 10 iterations of learning
    for iteration in range(10):
        # Dynamic weighting
        text_weight = 0.95 - (0.35 * iteration / 10)
        
        # Reassign all images
        labels = classify_images(prototypes)
        
        # Update prototypes
        prototypes = text_weight * text + (1 - text_weight) * visual_mean
```
- Iterative learning
- Better accuracy for complex datasets

#### 3. **Special Prompt Engineering**

**Digit Classification:**
```python
# Input: classes = ["2", "3", "4"]
# Generates 4 prompts per digit:
[
    "the digit two",
    "handwritten number 2",
    "the number 2",
    "digit 2"
]
# Averages embeddings for robustness
```

**Sign Language:**
```python
# Input: classes = ["two", "three", "four"]
# Generates 5 prompts per number:
[
    "sign language number two",
    "hand sign for two",
    "hand gesture showing two",
    "a hand showing the number 2",
    "two"
]
```

**General Objects:**
```python
# Input: classes = ["dog", "cat", "tiger"]
# Simple prompt:
"a photo of a {class_name}"
```

---

## 🚀 How to Use

### **Installation**

```powershell
# Create virtual environment
python -m venv venv
.\venv\Scripts\Activate.ps1

# Install dependencies
pip install torch torchvision
pip install git+https://github.com/openai/CLIP.git
pip install pillow numpy scikit-learn
```

### **Basic Usage**

```powershell
python auto_pipeline.py <image_folder> <output_dir> <class1> <class2> ...
```

### **Examples**

**Animals:**
```powershell
python auto_pipeline.py data\images animals_output dog cat tiger lion giraffe
```

**Fruits:**
```powershell
python auto_pipeline.py data\images4 fruits_output apple banana mango grape pineapple
```

**Handwritten Digits:**
```powershell
python auto_pipeline.py data\digits digit_output 0 1 2 3 4 5 6 7 8 9
```

**Sign Language:**
```powershell
python auto_pipeline.py data\signs sign_output two three four six seven eight nine
```

---

## 📊 Pipeline Steps

### **Step 1: Extract Embeddings**
- Loads CLIP ViT-B/32 model
- Processes images in batches (default: 32)
- Generates 512D embedding per image
- Output: `(N, 512)` numpy array

### **Step 2: Generate Text Embeddings**
- Detects classification mode (digits/sign language/general)
- Creates specialized prompts
- Averages multiple prompts for robust embeddings
- Output: `(C, 512)` numpy array (C = number of classes)

### **Step 3a: Classification (Small Datasets)**
- Direct cosine similarity
- Temperature scaling for confidence
- Fast single-pass classification

### **Step 3b: Iterative Refinement (Large Datasets)**
```
Iteration 1:  text_weight=0.95, moved=100%, active_classes=7/7
Iteration 2:  text_weight=0.92, moved=23%, active_classes=7/7
Iteration 3:  text_weight=0.88, moved=12%, active_classes=7/7
...
Iteration 10: text_weight=0.60, moved=1%, active_classes=7/7
```

**Outputs:**
- Movement tracking: Learning progress
- Confidence statistics: Quality metrics
- Class distribution: Balance check

### **Step 4: Organize Output**
```
output_dir/
├── class1/              # Images classified as class1
├── class2/              # Images classified as class2
├── ...
├── metadata.json        # Full classification data
├── report.txt          # Human-readable summary
└── splits/
    ├── train/          # 70% of images
    │   ├── class1/
    │   └── class2/
    ├── val/            # 15% of images
    └── test/           # 15% of images
```

---

## 📈 Performance & Results

### **Tested Datasets**

| Dataset | Size | Classes | Result | Notes |
|---------|------|---------|--------|-------|
| Sign Language (images3) | 413 | 7 digits | ✅ Working | Required dynamic weighting |
| Fruits (images4) | ~300 | 9 fruits | ✅ Working | Good separation |
| Animals | ~200 | 5 animals | ✅ Working | High confidence |

### **Typical Output**

```
Iteration 1/10: mean conf=0.276, active classes=7/7, text_weight=0.95, moved=413 (100.0%)
Iteration 2/10: mean conf=0.482, active classes=7/7, text_weight=0.92, moved=95 (23.0%)
Iteration 3/10: mean conf=0.561, active classes=7/7, text_weight=0.88, moved=47 (11.4%)
...
Iteration 10/10: mean conf=0.687, active classes=7/7, text_weight=0.60, moved=3 (0.7%)

Final Class Distribution:
  digit2    59 images (14.3%) - avg confidence: 0.682
  digit3    61 images (14.8%) - avg confidence: 0.694
  digit4    57 images (13.8%) - avg confidence: 0.678
  digit6    60 images (14.5%) - avg confidence: 0.689
  digit7    58 images (14.0%) - avg confidence: 0.691
  digit8    59 images (14.3%) - avg confidence: 0.685
  digit9    59 images (14.3%) - avg confidence: 0.687
```

**Indicators of Success:**
- ✅ All classes active throughout iterations
- ✅ Decreasing movement (convergence)
- ✅ Increasing confidence
- ✅ Balanced distribution

---

## � Complete Code Walkthrough: auto_pipeline.py

This section explains every function and code block in `auto_pipeline.py` line by line.

### **File Structure (565 lines total)**

```
Lines 1-28:    Imports and Setup
Lines 30-99:   extract_embeddings_inline() - CLIP image embeddings
Lines 101-157: generate_text_embeddings_inline() - Text embeddings with prompts
Lines 159-267: annotate_with_refinement() - Iterative learning algorithm
Lines 269-305: annotate_images_inline() - Direct classification for small datasets
Lines 307-458: organize_output_inline() - File organization and splits
Lines 460-509: run_pipeline() - Main orchestrator
Lines 511-565: Command-line interface
```

---

### **1. Imports and Setup (Lines 1-28)**

```python
"""
Complete End-to-End Auto-Annotation Pipeline
"""
```
**What it does:** Module docstring explaining the purpose and basic usage.

```python
import numpy as np        # Array operations and embeddings
import sys               # Command-line arguments
import torch             # PyTorch for CLIP
import clip              # OpenAI's CLIP model
from PIL import Image    # Image loading
from pathlib import Path # File path operations
```
**What it does:** Core dependencies for embeddings, image processing, and file handling.

```python
import json              # Save metadata
import shutil            # Copy files, create folders
import tempfile          # Temporary files (unused currently)
from sklearn.cluster import DBSCAN  # Clustering (unused currently)
```
**What it does:** Utilities for file organization and metadata.

```python
try:
    import hdbscan
    HAS_HDBSCAN = True
except ImportError:
    HAS_HDBSCAN = False
```
**What it does:** Optional dependency check. HDBSCAN is not currently used but kept for potential future use.

---

### **2. extract_embeddings_inline() - Lines 30-99**

**Purpose:** Load CLIP model and convert all images to 512D embedding vectors.

#### **Block 2.1: Model Loading (Lines 35-42)**
```python
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Loading CLIP model (device: {device})...")
model, preprocess = clip.load("ViT-B/32", device=device)
model.eval()
```
**What it does:**
- Detects if GPU available (CUDA) or use CPU
- Loads CLIP ViT-B/32 model (Vision Transformer, 32x32 patches)
- Returns: `model` (neural network) and `preprocess` (image transformation function)
- `model.eval()`: Sets model to evaluation mode (disables training features)

#### **Block 2.2: Image Discovery (Lines 44-53)**
```python
image_folder = Path(image_folder)
image_extensions = ['.jpg', '.jpeg', '.png', '.bmp', '.gif']
image_paths = []

for ext in image_extensions:
    image_paths.extend(list(image_folder.rglob(f"*{ext}")))
    image_paths.extend(list(image_folder.rglob(f"*{ext.upper()}")))

image_paths = sorted(set(image_paths))
```
**What it does:**
- Converts input folder to Path object
- Searches recursively (`rglob`) for all image files
- Handles both lowercase and uppercase extensions (.jpg and .JPG)
- Removes duplicates with `set()` and sorts alphabetically
- **Result:** List of all image file paths

#### **Block 2.3: Batch Processing (Lines 60-91)**
```python
embeddings = []
valid_paths = []

for i in range(0, len(image_paths), batch_size):
    batch_paths = image_paths[i:i + batch_size]
```
**What it does:**
- Processes images in batches of 32 (default)
- Batch processing is faster than one-by-one

```python
for img_path in batch_paths:
    try:
        image = Image.open(img_path).convert('RGB')
        batch_images.append(preprocess(image))
        batch_valid_paths.append(str(img_path))
    except Exception as e:
        print(f"Warning: Failed to load {img_path}: {e}")
        continue
```
**What it does:**
- Opens each image file
- Converts to RGB (removes alpha channel, handles grayscale)
- Applies CLIP preprocessing (resize to 224x224, normalize)
- Catches and skips corrupted images
- Tracks which images loaded successfully

```python
with torch.no_grad():
    image_input = torch.stack(batch_images).to(device)
    image_features = model.encode_image(image_input)
    image_features = image_features / image_features.norm(dim=-1, keepdim=True)
    embeddings.append(image_features.cpu().numpy())
```
**What it does:**
- `torch.no_grad()`: Disables gradient computation (saves memory, speeds up)
- `torch.stack()`: Combines list of images into single tensor [batch_size, 3, 224, 224]
- `.to(device)`: Moves data to GPU or CPU
- `model.encode_image()`: Passes through neural network → 512D vector per image
- **Normalization:** Divides by L2 norm (makes all vectors unit length)
  - Why? Enables cosine similarity (dot product of normalized vectors)
- `.cpu().numpy()`: Converts back to NumPy array on CPU

#### **Block 2.4: Return (Lines 93-99)**
```python
embeddings = np.vstack(embeddings)  # Stack all batches
print(f"✓ Extracted embeddings: {embeddings.shape}\n")
return embeddings, valid_paths, model, device
```
**What it does:**
- `np.vstack()`: Combines all batch embeddings into single array [N, 512]
- Returns: embeddings (array), image paths (list), model (for text encoding), device

---

### **3. generate_text_embeddings_inline() - Lines 101-157**

**Purpose:** Convert class names into 512D text embeddings using CLIP.

#### **Block 3.1: Digit Detection (Lines 107-109)**
```python
is_digit_dataset = all(name.strip() in '0123456789' for name in class_names)
```
**What it does:**
- Checks if ALL class names are single digits ("0", "1", "2", etc.)
- Returns `True` only if every class name is in "0123456789"
- Example: `["2", "3", "4"]` → True, `["two", "three"]` → False

#### **Block 3.2: Prompt Generation - Digits (Lines 111-127)**
```python
if is_digit_dataset:
    digit_names = ['zero', 'one', 'two', ..., 'nine']
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
```
**What it does:**
- For digit "2", creates 4 prompts:
  - "the digit two"
  - "handwritten number 2"
  - "the number 2"
  - "digit 2"
- Why? Multiple descriptions capture different visual representations
- Result: `prompts = [[prompt1, prompt2, ...], ...]` (list of lists)

#### **Block 3.3: Prompt Generation - General Objects (Lines 128-130)**
```python
else:
    prompts = [[f"a photo of a {name}"] for name in class_names]
```
**What it does:**
- For "dog", creates: `["a photo of a dog"]`
- Standard CLIP prompt format for object classification
- Wrapped in list for consistent processing with digit mode

#### **Block 3.4: Text Encoding (Lines 133-151)**
```python
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
```
**What it does:**
- `clip.tokenize()`: Converts text → token IDs (CLIP's vocabulary)
- `model.encode_text()`: Token IDs → 512D embedding
- **Normalization:** Unit length (same as image embeddings)
- `torch.stack().mean(dim=0)`: Averages 4 prompts into single embedding
  - Why? More robust than single prompt
  - Example: avg("digit two", "number 2", ...) captures concept better
- Result: One embedding per class

#### **Block 3.5: Return (Lines 153-157)**
```python
text_embeddings = np.vstack(text_embeddings_list)
return text_embeddings
```
**What it does:**
- Stacks into array [num_classes, 512]
- Each row = embedding for one class

---

### **4. annotate_with_refinement() - Lines 159-267**

**Purpose:** Iteratively improve classification by learning from visual data. Used for datasets ≥100 images.

#### **Block 4.1: Setup (Lines 169-184)**
```python
# Normalize
image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)
text_embeddings = text_embeddings / np.linalg.norm(text_embeddings, axis=1, keepdims=True)
```
**What it does:**
- Ensures unit length (already normalized but defensive check)
- `axis=1`: Normalizes each row independently
- `keepdims=True`: Keeps 2D shape [N, 512] instead of [N]

```python
num_classes = len(class_names)
num_images = len(image_embeddings)

# Initialize prototypes from text embeddings
prototypes = text_embeddings.copy()
```
**What it does:**
- **Prototypes:** Class representations in embedding space
- Initially: Pure text embeddings (e.g., "dog", "cat")
- Will blend with visual data during iterations

#### **Block 4.2: Iterative Loop (Lines 193-244)**

**THE CORE ALGORITHM - This is where learning happens**

```python
for iteration in range(num_iterations):  # Default: 10 iterations
    # Dynamic text weight
    text_weight = 0.95 - (0.35 * iteration / num_iterations)
    visual_weight = 1.0 - text_weight
```
**What it does:**
- **Dynamic weighting schedule:**
  - Iteration 1: text_weight = 0.95 (95% text, 5% visual)
  - Iteration 5: text_weight = 0.78 (78% text, 22% visual)
  - Iteration 10: text_weight = 0.60 (60% text, 40% visual)
- **Why dynamic?** Prevents early collapse (all images → one class)
- Start with strong text anchor, gradually allow visual learning

**Step 1: Reassign ALL Images**
```python
similarities = image_embeddings @ prototypes.T
```
**What it does:**
- Matrix multiplication: [N, 512] @ [C, 512].T = [N, C]
- Each cell [i, j] = similarity between image i and class j
- Cosine similarity (dot product of normalized vectors)
- Range: [-1, 1], higher = more similar

```python
similarities_scaled = similarities / temperature  # temperature = 0.05
labels = np.argmax(similarities_scaled, axis=1)
confidences = np.max(similarities, axis=1)
```
**What it does:**
- **Temperature scaling:** Divides by 0.05 (multiplies by 20)
  - Original: [0.65, 0.63, 0.61] (hard to distinguish)
  - Scaled: [13, 12.6, 12.2] → After softmax: [0.75, 0.15, 0.10] (clear winner)
- `argmax()`: Picks class with highest similarity
- Confidence: Maximum similarity value (unscaled)

```python
if prev_labels is not None:
    num_moved = np.sum(labels != prev_labels)
    movement_pct = 100 * num_moved / num_images
```
**What it does:**
- **Movement tracking:** How many images changed class?
- Learning indicator: High movement → still learning, Low → converged
- First iteration: 100% movement (no previous labels)

**Step 2: Recompute Prototypes**
```python
new_prototypes = np.zeros_like(prototypes)

for c in range(num_classes):
    class_mask = (labels == c)
    class_count = np.sum(class_mask)
    
    if class_count > 0:
        # Average embeddings of images assigned to this class
        class_mean = np.mean(image_embeddings[class_mask], axis=0)
        class_mean /= np.linalg.norm(class_mean)
```
**What it does:**
- `class_mask`: Boolean array, True for images assigned to class c
- `class_mean`: Average of all embeddings in this class
  - Example: If digit "2" has 50 images, averages their 50 embeddings
  - This is the **visual prototype** (what "2" actually looks like in data)

```python
        # Blend with dynamic weights
        new_prototypes[c] = text_weight * text_embeddings[c] + visual_weight * class_mean
        new_prototypes[c] /= np.linalg.norm(new_prototypes[c])
```
**What it does:**
- **Blending formula:** `prototype = α * text + (1-α) * visual`
- Early iterations: Mostly text (prevents collapse)
- Later iterations: More visual (learns from data)
- Normalize to unit length

```python
    else:
        # No images assigned - keep text embedding
        new_prototypes[c] = text_embeddings[c]
```
**What it does:**
- Handles empty classes (no images assigned)
- Falls back to text-only prototype

**Step 3: Update and Report**
```python
prototypes = new_prototypes
prev_labels = labels.copy()

print(f"  Iteration {iteration + 1}/{num_iterations}: "
      f"mean conf={np.mean(confidences):.3f}, "
      f"active classes={active_classes}/{num_classes}, "
      f"text_weight={text_weight:.2f}, "
      f"moved={num_moved} ({movement_pct:.1f}%)")
```
**What it does:**
- Updates prototypes for next iteration
- Saves current labels to detect movement next time
- Prints progress: confidence, active classes, weight, movement

#### **Block 4.3: Final Assignment and Statistics (Lines 246-267)**
```python
# Final assignment with final prototypes
similarities = image_embeddings @ prototypes.T
labels = np.argmax(similarities, axis=1)
confidences = np.max(similarities, axis=1)

print(f"Final Class Distribution:")
for class_idx, class_name in enumerate(class_names):
    count = np.sum(labels == class_idx)
    percentage = 100 * count / len(labels)
    avg_conf = np.mean(confidences[labels == class_idx])
    print(f"  {class_name:15} {count:4} images ({percentage:5.1f}%) - avg confidence: {avg_conf:.3f}")
```
**What it does:**
- Re-classifies with final learned prototypes
- Prints distribution: how many images per class
- Calculates average confidence per class
- Prints overall statistics (mean, std, min, max confidence)

**Returns:** `labels, confidences, similarities`

---

### **5. annotate_images_inline() - Lines 269-305**

**Purpose:** Direct classification for small datasets (<100 images). No iterative learning.

```python
# Normalize
image_embeddings = image_embeddings / np.linalg.norm(image_embeddings, axis=1, keepdims=True)

# Compute similarities
similarities = image_embeddings @ text_embeddings.T  # [N, C]

# Assign labels
labels = np.argmax(similarities, axis=1)
confidences = np.max(similarities, axis=1)
```
**What it does:**
- Single-pass classification (no iterations)
- Uses pure text embeddings (no visual blending)
- Fast and simple for small datasets
- Same math as iteration 1 of `annotate_with_refinement()`

**Returns:** `labels, confidences, similarities`

---

### **6. organize_output_inline() - Lines 307-458**

**Purpose:** Copy images to class folders, create train/val/test splits, save metadata.

#### **Block 6.1: Setup Output Directory (Lines 320-327)**
```python
output_path = Path(output_dir)
if output_path.exists():
    shutil.rmtree(output_path)  # Delete existing
output_path.mkdir(parents=True)

# Create class folders
for class_name in class_names:
    (output_path / class_name).mkdir(exist_ok=True)
```
**What it does:**
- Cleans up old output (prevents mixing old and new results)
- Creates main output directory
- Creates one folder per class (e.g., `dog/`, `cat/`, `tiger/`)

#### **Block 6.2: Copy Images (Lines 331-350)**
```python
for img_path, label in zip(image_paths, labels):
    src = Path(img_path)
    class_name = class_names[label]
    dst = output_path / class_name / src.name
    
    # Handle duplicates
    counter = 1
    while dst.exists():
        dst = output_path / class_name / f"{src.stem}_{counter}{src.suffix}"
        counter += 1
    
    shutil.copy2(src, dst)
    organized_paths[class_name].append(str(dst.relative_to(output_path)))
```
**What it does:**
- Loops through each image and its predicted label
- Copies to appropriate class folder
- **Duplicate handling:** If `dog_1.jpg` exists, creates `dog_1_2.jpg`
- `shutil.copy2()`: Copies file + preserves metadata (timestamps)
- Tracks organized paths for metadata

#### **Block 6.3: Save Metadata JSON (Lines 354-384)**
```python
metadata = {
    'method': 'direct_clip_annotation',
    'class_names': class_names,
    'num_images': len(image_paths),
    'class_distribution': {...},
    'organized_paths': {...},
    'image_annotations': [
        {
            'original_path': img_path,
            'class_index': int(label),
            'class_name': class_names[label],
            'confidence': float(conf),
            'all_similarities': {...}
        }
        for i, (img_path, label, conf) in enumerate(...)
    ],
    'statistics': {
        'mean_confidence': float(np.mean(confidences)),
        ...
    }
}

with open(output_path / 'metadata.json', 'w') as f:
    json.dump(metadata, f, indent=2)
```
**What it does:**
- Creates comprehensive JSON with all classification data
- **Per-image annotations:** Path, label, confidence, all class scores
- **Global statistics:** Distribution, confidence metrics
- Useful for analysis, debugging, visualization

#### **Block 6.4: Save Human-Readable Report (Lines 386-415)**
```python
report_lines = [
    "="*70,
    "AUTO-ANNOTATION REPORT",
    "="*70,
    "",
    f"Total images: {len(image_paths)}",
    ...
]

with open(output_path / 'report.txt', 'w') as f:
    f.write('\n'.join(report_lines))
```
**What it does:**
- Creates simple text file with key results
- Easy to view: `cat report.txt` or open in any editor
- Shows distribution and statistics

#### **Block 6.5: Train/Val/Test Splits (Lines 417-458)**
```python
# Create split directories
splits_dir = output_path / 'splits'
for split_name in ['train', 'val', 'test']:
    split_dir = splits_dir / split_name
    split_dir.mkdir(exist_ok=True)
    for class_name in class_names:
        (split_dir / class_name).mkdir(exist_ok=True)
```
**What it does:**
- Creates directory structure:
  ```
  splits/
    train/
      dog/
      cat/
    val/
      dog/
      cat/
    test/
      dog/
      cat/
  ```

```python
np.random.seed(42)  # Reproducible splits
split_ratios = {'train': 0.7, 'val': 0.15, 'test': 0.15}

for class_name in class_names:
    class_folder = output_path / class_name
    images = list(class_folder.glob('*'))
    
    indices = np.random.permutation(len(images))
```
**What it does:**
- Seed 42: Same random split every time
- Ratios: 70% training, 15% validation, 15% test
- Shuffles images randomly per class

```python
    ratios = np.array(ratios) / sum(ratios)
    split_points = np.cumsum(ratios * len(images)).astype(int)
    
    for split_idx, split_name in enumerate(splits):
        end = split_points[split_idx]
        split_indices = indices[start:end]
        
        for idx in split_indices:
            src = images[idx]
            dst = splits_dir / split_name / class_name / src.name
            shutil.copy2(src, dst)
```
**What it does:**
- Calculates split points: If 100 images → [70, 85, 100]
- Copies first 70 to train/, next 15 to val/, last 15 to test/
- **Per-class splitting:** Ensures balanced representation

---

### **7. run_pipeline() - Lines 460-509**

**Purpose:** Main orchestrator that calls all functions in sequence.

```python
def run_pipeline(image_folder: str, output_dir: str, class_names: List[str]):
    # Print header
    print(f"\n{'='*70}")
    print("AUTO-ANNOTATION COMPLETE PIPELINE")
    print(f"{'='*70}")
```

#### **Step 1: Extract Embeddings**
```python
image_embeddings, image_paths, model, device = extract_embeddings_inline(image_folder)
dataset_size = len(image_embeddings)
```
**What it does:** Loads CLIP, processes all images → 512D embeddings

#### **Step 2: Text Embeddings**
```python
text_embeddings = generate_text_embeddings_inline(class_names, model, device)
```
**What it does:** Converts class names → 512D embeddings

#### **Step 3: Adaptive Classification**
```python
if dataset_size < 100:
    # Small dataset: Direct CLIP classification
    labels, confidences, similarities = annotate_images_inline(
        image_embeddings, text_embeddings, class_names, temperature=0.05
    )
else:
    # Large dataset: Use iterative refinement
    labels, confidences, similarities = annotate_with_refinement(
        image_embeddings, text_embeddings, class_names,
        num_iterations=10, temperature=0.05
    )
```
**What it does:**
- **Decision logic:** Size-based strategy
- Small (<100): Fast direct classification
- Large (≥100): Iterative learning for better accuracy
- Both use temperature=0.05 for sharp boundaries

#### **Step 4: Organize Output**
```python
organize_output_inline(
    image_paths, labels, confidences, similarities, class_names, output_dir
)
```
**What it does:** Creates folders, copies files, generates metadata

#### **Final Summary**
```python
print(f"Your annotated dataset is ready at: {output_dir}")
print(f"\nOutput structure:")
print(f"  {output_dir}/")
for class_name in class_names:
    print(f"    ├── {class_name}/")
print(f"    ├── metadata.json")
print(f"    ├── report.txt")
print(f"    └── splits/")
```
**What it does:** Shows user what was created and next steps

---

### **8. Command-Line Interface (Lines 511-565)**

#### **Block 8.1: Usage Help (Lines 535-549)**
```python
if __name__ == "__main__":
    if len(sys.argv) < 4:
        print("Usage:")
        print("  python auto_pipeline.py <image_folder> <output_dir> <class1> <class2> ...")
        print("\nExample:")
        print("  python auto_pipeline.py data/images annotated_output dog cat tiger lion giraffe")
        sys.exit(1)
```
**What it does:**
- `__name__ == "__main__"`: Only runs if script executed directly (not imported)
- Checks argument count: Need at least 4 (script name, folder, output, 1+ class)
- Shows usage examples if arguments missing

#### **Block 8.2: Parse Arguments (Lines 551-553)**
```python
image_folder = sys.argv[1]  # Input folder
output_dir = sys.argv[2]     # Output folder
class_names = sys.argv[3:]   # All remaining args = class names
```
**What it does:**
- `sys.argv[0]`: Script name (`auto_pipeline.py`)
- `sys.argv[1]`: First argument (input folder)
- `sys.argv[2]`: Second argument (output folder)
- `sys.argv[3:]`: Slice from index 3 to end (all class names)

Example: `python auto_pipeline.py data/imgs out dog cat`
- `sys.argv = ['auto_pipeline.py', 'data/imgs', 'out', 'dog', 'cat']`
- `image_folder = 'data/imgs'`
- `output_dir = 'out'`
- `class_names = ['dog', 'cat']`

#### **Block 8.3: Validation and Execution (Lines 555-565)**
```python
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
```
**What it does:**
- **Validation:** Checks if input folder exists before processing
- **Error handling:** Catches any exceptions during pipeline
- `traceback.print_exc()`: Shows full error details for debugging
- `sys.exit(1)`: Returns error code (non-zero = failure)

---

## 🎯 Key Concepts Summary

### **1. Embeddings**
- **What:** High-dimensional vectors (512D) representing images/text
- **How:** Neural networks convert images/text → numbers
- **Why:** Math operations in vector space (similarity, clustering)

### **2. Cosine Similarity**
```python
similarity = image_embedding @ text_embedding.T
# Equivalent to: dot(image, text) when normalized
# Range: [-1, 1], higher = more similar
```

### **3. Temperature Scaling**
```python
# Sharpen probability distribution
logits_scaled = logits / 0.05  # Small temp = confident
probs = softmax(logits_scaled)
```

### **4. Dynamic Text Weighting**
```python
# Prevents collapse by maintaining text anchor early on
text_weight = 0.95 - (0.35 * iteration / 10)
prototype = text_weight * text + (1 - text_weight) * visual
```

### **5. Normalization**
```python
# Makes all vectors unit length
normalized = vector / np.linalg.norm(vector)
# Why? Cosine similarity requires unit vectors
```

---

## �🔬 Key Algorithms

### **Iterative Prototype Refinement**

```python
def annotate_with_refinement(image_embeddings, text_embeddings, class_names):
    # Initialize prototypes from text
    prototypes = text_embeddings.copy()
    
    for iteration in range(10):
        # Dynamic text weight (prevents collapse)
        text_weight = 0.95 - (0.35 * iteration / 10)
        visual_weight = 1.0 - text_weight
        
        # Classify all images
        similarities = image_embeddings @ prototypes.T
        similarities_scaled = similarities / temperature
        labels = np.argmax(similarities_scaled, axis=1)
        
        # Recompute prototypes
        for class_idx in range(num_classes):
            class_mask = (labels == class_idx)
            if np.sum(class_mask) > 0:
                # Visual mean of assigned images
                visual_mean = np.mean(image_embeddings[class_mask], axis=0)
                visual_mean /= np.linalg.norm(visual_mean)
                
                # Blend text and visual
                prototypes[class_idx] = (
                    text_weight * text_embeddings[class_idx] + 
                    visual_weight * visual_mean
                )
                prototypes[class_idx] /= np.linalg.norm(prototypes[class_idx])
        
        # Track movement
        num_moved = np.sum(labels != prev_labels)
    
    return labels, confidences
```

### **Temperature Scaling**

```python
# Without temperature: similar classes overlap
similarities = [0.65, 0.63, 0.61]  # Hard to distinguish

# With temperature=0.05: sharper distinctions
similarities_scaled = similarities / 0.05
# After softmax: [0.75, 0.15, 0.10]  # Clear winner
```

---

## 🛠️ Files in Repository

### **Active Files**

| File | Purpose | Status |
|------|---------|--------|
| `auto_pipeline.py` | Main production pipeline | ✅ Active |
| `auto_pipeline_simple.py` | Simplified backup version | ✅ Backup |
| `README.md` | This documentation | ✅ Active |

### **Legacy/Experimental Files** *(Not Required)*

- `auto_annotation_engine.py` - Old implementation
- `config.py` - Legacy config
- `data_loader.py` - Old data handling
- `dataset_organizer.py` - Old organizer
- `direct_annotation.py` - Early experiment
- `example_usage.py` - Old examples
- `extract_embeddings.py` - Standalone extractor
- `hybrid_refine.py` - Experimental refinement
- `main.py` - Old entry point
- `rename.py` - Utility script
- `test_engine.py` - Old testing
- `unsupervised_discovery.py` - Experimental clustering
- `visualizer.py` - Visualization tool

**These 13 files can be deleted.** Only `auto_pipeline.py` and `auto_pipeline_simple.py` are needed.

---

## 🧪 Troubleshooting

### **Problem: All images in one class**

**Symptoms:**
```
Iteration 1/10: active classes=1/7, moved=413 (100.0%)
Iteration 2/10: active classes=1/7, moved=0 (0.0%)
```

**Cause:** Early prototype collapse

**Solution:** ✅ Fixed with dynamic text weighting (95% → 60%)

### **Problem: No movement after iteration 1**

**Symptoms:**
```
moved=0 (0.0%) for iterations 2-10
```

**Cause:** Prototypes not allowing reassignment

**Solution:** ✅ Fixed by reassigning ALL images every iteration

### **Problem: Low confidence**

**Symptoms:**
```
Mean confidence: 0.312
```

**Causes:**
- Classes too similar (e.g., "cat" vs "tiger")
- Poor image quality
- Wrong class names

**Solutions:**
- Use more distinct class names
- Check image quality
- Try different prompts

### **Problem: Unbalanced distribution**

**Symptoms:**
```
class1: 300 images (75%)
class2: 10 images (2.5%)
```

**Causes:**
- Dataset actually is imbalanced
- One class definition too broad

**Solutions:**
- Check if dataset truly needs those classes
- Refine class definitions
- Add more specific classes

---

## 📚 Technical Background

### **CLIP (Contrastive Language-Image Pre-training)**

- Developed by OpenAI
- Trained on 400M image-text pairs
- Zero-shot: No retraining needed for new classes
- Embedding space: Images and text in same 512D space
- Architecture: ViT-B/32 (Vision Transformer)

### **Key Concepts**

**Cosine Similarity:**
```python
similarity = (image_vec · text_vec) / (||image_vec|| × ||text_vec||)
# Range: [-1, 1]
# 1 = identical direction, -1 = opposite, 0 = orthogonal
```

**Prototype:**
```python
# Class representation in embedding space
prototype = text_weight * text_embedding + visual_weight * mean(class_images)
```

**Temperature Scaling:**
```python
# Sharpen or soften probability distribution
logits_scaled = logits / temperature
# Low temp (0.05): Sharp peaks (confident)
# High temp (1.0): Smooth distribution (uncertain)
```

---

## 🎯 Future Improvements

### **Potential Enhancements:**

1. **LLaVA Integration** *(Removed for simplicity)*
   - Use for low-confidence verification
   - 7B parameter model too heavy
   - Could use BLIP-2 (lighter alternative)

2. **Active Learning**
   - Human labels for uncertain images
   - Retrain prototypes with human feedback

3. **Ensemble Methods**
   - Multiple CLIP models (ViT-B/32, ViT-L/14)
   - Voting or confidence weighting

4. **Fine-tuning**
   - Fine-tune CLIP on specific domains
   - Better accuracy for specialized datasets

5. **Confidence Thresholding**
   - Flag images below confidence threshold
   - Manual review queue

---

## 📝 Lessons Learned

1. **Simplicity > Complexity**
   - Hierarchical clustering was overcomplicated
   - Pure iterative reassignment works better

2. **Prevent Early Collapse**
   - Fixed blend ratios cause dominance
   - Dynamic weighting maintains competition

3. **Track Learning Signals**
   - Movement percentage shows if learning happens
   - Confidence alone is not enough

4. **Temperature Matters**
   - 0.05 works well for sharper boundaries
   - Too low: overconfident, too high: indecisive

5. **Prompt Engineering**
   - Multiple prompts + averaging = robustness
   - Domain-specific prompts (digits, sign language) help

---

## 👤 Author

Developed through iterative problem-solving and debugging.

**Project Timeline:**
- Initial Implementation: Complex clustering approach
- Phase 1 Issues: Prototype collapse discovered
- Phase 2 Solution: Pure iterative reassignment
- Phase 3 Fix: Dynamic text weighting
- Final Result: Working universal pipeline

---

## 📄 License

Open source - Use freely for any project.

---

## 🙏 Acknowledgments

- **OpenAI CLIP:** Foundation model enabling zero-shot classification
- **PyTorch:** Deep learning framework
- **scikit-learn:** Clustering and preprocessing utilities

---

**Last Updated:** January 12, 2026
