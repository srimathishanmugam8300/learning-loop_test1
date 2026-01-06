"""
Output organizer for the Auto-Annotation Engine.

Organizes labeled images into class folders and generates metadata files.
"""

import json
import shutil
import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from auto_annotation_engine import LearningResult


class DatasetOrganizer:
    """
    Organize annotated dataset into class folders.
    
    Creates structure:
    output_dir/
        class_1/
            image1.jpg
            image2.jpg
        class_2/
            image3.jpg
        metadata.json
        prototypes.npy
        alignment_matrix.npy
        confidence_scores.json
    """
    
    def __init__(
        self,
        output_dir: Union[str, Path],
        class_names: List[str],
        overwrite: bool = False
    ):
        """
        Initialize organizer.
        
        Args:
            output_dir: Output directory path
            class_names: List of class names
            overwrite: Whether to overwrite existing directory
        """
        self.output_dir = Path(output_dir)
        self.class_names = class_names
        self.overwrite = overwrite
        
        # Create output directory
        if self.output_dir.exists() and not overwrite:
            raise ValueError(
                f"Output directory {output_dir} already exists. "
                "Set overwrite=True to replace it."
            )
        
        if self.output_dir.exists() and overwrite:
            shutil.rmtree(self.output_dir)
        
        self.output_dir.mkdir(parents=True, exist_ok=True)
    
    def organize_dataset(
        self,
        image_paths: List[str],
        learning_result: LearningResult,
        copy_files: bool = True,
        create_symlinks: bool = False
    ):
        """
        Organize images into class folders.
        
        Args:
            image_paths: List of source image paths
            learning_result: Results from learning loop
            copy_files: Whether to copy image files (default: True)
            create_symlinks: Create symlinks instead of copying (default: False)
        """
        assert len(image_paths) == len(learning_result.image_labels)
        
        # Create class folders
        class_folders = {}
        for class_idx, class_name in enumerate(self.class_names):
            class_folder = self.output_dir / class_name
            class_folder.mkdir(exist_ok=True)
            class_folders[class_idx] = class_folder
        
        # Organize images
        organized_paths = {class_name: [] for class_name in self.class_names}
        
        for img_idx, img_path in enumerate(image_paths):
            src_path = Path(img_path)
            if not src_path.exists():
                print(f"Warning: Image not found: {img_path}")
                continue
            
            # Get assigned class
            class_idx = learning_result.image_labels[img_idx]
            class_name = self.class_names[class_idx]
            class_folder = class_folders[class_idx]
            
            # Determine destination path
            dest_path = class_folder / src_path.name
            
            # Handle duplicate names
            counter = 1
            while dest_path.exists():
                stem = src_path.stem
                suffix = src_path.suffix
                dest_path = class_folder / f"{stem}_{counter}{suffix}"
                counter += 1
            
            # Copy or symlink
            if create_symlinks:
                dest_path.symlink_to(src_path.absolute())
            elif copy_files:
                shutil.copy2(src_path, dest_path)
            
            organized_paths[class_name].append(str(dest_path.relative_to(self.output_dir)))
        
        # Save metadata
        self._save_metadata(
            image_paths=image_paths,
            learning_result=learning_result,
            organized_paths=organized_paths
        )
        
        # Save prototypes
        np.save(
            self.output_dir / 'class_prototypes.npy',
            learning_result.class_prototypes
        )
        
        # Save alignment matrices
        np.save(
            self.output_dir / 'meta_cluster_alignment.npy',
            learning_result.meta_cluster_alignment
        )
        np.save(
            self.output_dir / 'micro_cluster_alignment.npy',
            learning_result.micro_cluster_alignment
        )
        
        # Generate summary report
        self._generate_report(learning_result, organized_paths)
        
        print(f"\n✓ Dataset organized successfully!")
        print(f"  Output directory: {self.output_dir}")
        print(f"  Total images: {len(image_paths)}")
        print(f"  Classes: {len(self.class_names)}")
    
    def _save_metadata(
        self,
        image_paths: List[str],
        learning_result: LearningResult,
        organized_paths: Dict[str, List[str]]
    ):
        """Save comprehensive metadata."""
        metadata = {
            'class_names': self.class_names,
            'num_classes': len(self.class_names),
            'num_images': len(image_paths),
            'class_distribution': {
                class_name: len(paths)
                for class_name, paths in organized_paths.items()
            },
            'organized_paths': organized_paths,
            'image_annotations': []
        }
        
        # Per-image annotations
        for img_idx, img_path in enumerate(image_paths):
            class_idx = learning_result.image_labels[img_idx]
            class_name = self.class_names[class_idx]
            confidence = float(learning_result.confidence_scores[img_idx])
            
            metadata['image_annotations'].append({
                'original_path': img_path,
                'class_index': int(class_idx),
                'class_name': class_name,
                'confidence': confidence
            })
        
        # Statistics
        metadata['statistics'] = {
            'mean_confidence': float(np.mean(learning_result.confidence_scores)),
            'std_confidence': float(np.std(learning_result.confidence_scores)),
            'min_confidence': float(np.min(learning_result.confidence_scores)),
            'max_confidence': float(np.max(learning_result.confidence_scores))
        }
        
        # Iteration history
        metadata['learning_history'] = learning_result.iteration_history
        
        # Save to file
        with open(self.output_dir / 'metadata.json', 'w') as f:
            json.dump(metadata, f, indent=2)
    
    def _generate_report(
        self,
        learning_result: LearningResult,
        organized_paths: Dict[str, List[str]]
    ):
        """Generate human-readable report."""
        report_lines = []
        report_lines.append("=" * 70)
        report_lines.append("AUTO-ANNOTATION RESULTS REPORT")
        report_lines.append("=" * 70)
        report_lines.append("")
        
        # Class distribution
        report_lines.append("CLASS DISTRIBUTION:")
        report_lines.append("-" * 70)
        total = len(learning_result.image_labels)
        for class_name in self.class_names:
            count = len(organized_paths[class_name])
            percentage = 100 * count / total
            report_lines.append(f"  {class_name:<30} {count:>6} images ({percentage:>5.1f}%)")
        report_lines.append("")
        
        # Confidence statistics
        report_lines.append("CONFIDENCE STATISTICS:")
        report_lines.append("-" * 70)
        for class_idx, class_name in enumerate(self.class_names):
            mask = learning_result.image_labels == class_idx
            if np.sum(mask) > 0:
                class_confidences = learning_result.confidence_scores[mask]
                report_lines.append(
                    f"  {class_name:<30} "
                    f"mean: {np.mean(class_confidences):.3f}, "
                    f"std: {np.std(class_confidences):.3f}, "
                    f"min: {np.min(class_confidences):.3f}"
                )
        report_lines.append("")
        
        # Overall statistics
        report_lines.append("OVERALL STATISTICS:")
        report_lines.append("-" * 70)
        report_lines.append(f"  Total images: {total}")
        report_lines.append(f"  Number of classes: {len(self.class_names)}")
        report_lines.append(f"  Mean confidence: {np.mean(learning_result.confidence_scores):.3f}")
        report_lines.append(f"  Std confidence: {np.std(learning_result.confidence_scores):.3f}")
        report_lines.append("")
        
        # Learning convergence
        report_lines.append("LEARNING CONVERGENCE:")
        report_lines.append("-" * 70)
        for iteration_info in learning_result.iteration_history:
            it = iteration_info['iteration']
            change = iteration_info['prototype_change']
            report_lines.append(f"  Iteration {it}: prototype change = {change:.6f}")
        report_lines.append("")
        
        report_lines.append("=" * 70)
        
        # Save report
        report_text = "\n".join(report_lines)
        with open(self.output_dir / 'report.txt', 'w') as f:
            f.write(report_text)
        
        # Print to console
        print(report_text)
    
    def export_for_training(
        self,
        format: str = "classification",
        split_ratios: Optional[Dict[str, float]] = None
    ):
        """
        Export dataset in format suitable for training.
        
        Args:
            format: Export format ('classification', 'yolo', 'coco')
            split_ratios: Train/val/test split ratios
        """
        if format == "classification":
            self._export_classification_format(split_ratios)
        elif format == "yolo":
            self._export_yolo_format(split_ratios)
        elif format == "coco":
            self._export_coco_format(split_ratios)
        else:
            raise ValueError(f"Unsupported format: {format}")
    
    def _export_classification_format(
        self,
        split_ratios: Optional[Dict[str, float]] = None
    ):
        """
        Export in standard image classification format.
        
        Format:
        dataset/
            train/
                class1/
                class2/
            val/
                class1/
                class2/
            test/
                class1/
                class2/
        """
        if split_ratios is None:
            split_ratios = {'train': 0.8, 'val': 0.1, 'test': 0.1}
        
        # Create split directories
        splits_dir = self.output_dir / 'splits'
        splits_dir.mkdir(exist_ok=True)
        
        for split_name in split_ratios.keys():
            split_dir = splits_dir / split_name
            split_dir.mkdir(exist_ok=True)
            
            for class_name in self.class_names:
                (split_dir / class_name).mkdir(exist_ok=True)
        
        # Split images
        np.random.seed(42)
        split_mapping = {split: [] for split in split_ratios.keys()}
        
        for class_name in self.class_names:
            class_folder = self.output_dir / class_name
            images = list(class_folder.glob('*'))
            
            if len(images) == 0:
                continue
            
            # Shuffle
            indices = np.random.permutation(len(images))
            
            # Compute split points
            splits = list(split_ratios.keys())
            ratios = [split_ratios[s] for s in splits]
            ratios = np.array(ratios) / sum(ratios)  # Normalize
            
            split_points = np.cumsum(ratios * len(images)).astype(int)
            
            # Assign to splits
            start = 0
            for split_idx, split_name in enumerate(splits):
                end = split_points[split_idx]
                split_indices = indices[start:end]
                
                for idx in split_indices:
                    src = images[idx]
                    dst = splits_dir / split_name / class_name / src.name
                    shutil.copy2(src, dst)
                    split_mapping[split_name].append({
                        'image': str(src.relative_to(self.output_dir)),
                        'class': class_name
                    })
                
                start = end
        
        # Save split metadata
        with open(splits_dir / 'split_info.json', 'w') as f:
            json.dump(split_mapping, f, indent=2)
        
        print(f"\n✓ Exported classification dataset to {splits_dir}")
        for split_name, items in split_mapping.items():
            print(f"  {split_name}: {len(items)} images")
    
    def _export_yolo_format(self, split_ratios: Optional[Dict[str, float]]):
        """Export in YOLO format (placeholder for future implementation)."""
        print("YOLO format export not yet implemented.")
    
    def _export_coco_format(self, split_ratios: Optional[Dict[str, float]]):
        """Export in COCO format (placeholder for future implementation)."""
        print("COCO format export not yet implemented.")
