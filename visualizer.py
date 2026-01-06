"""
Visualization utilities for analyzing auto-annotation results.

Optional module - requires matplotlib and seaborn.
"""

import json
import numpy as np
from pathlib import Path
from typing import Optional, List

try:
    import matplotlib.pyplot as plt
    import seaborn as sns
    VISUALIZATION_AVAILABLE = True
except ImportError:
    VISUALIZATION_AVAILABLE = False
    print("Warning: matplotlib/seaborn not installed. Visualization unavailable.")


class ResultsVisualizer:
    """
    Visualize auto-annotation results and learning progress.
    """
    
    def __init__(self, output_dir: str):
        """
        Initialize visualizer.
        
        Args:
            output_dir: Directory containing auto-annotation results
        """
        if not VISUALIZATION_AVAILABLE:
            raise ImportError(
                "Visualization requires matplotlib and seaborn. "
                "Install with: pip install matplotlib seaborn"
            )
        
        self.output_dir = Path(output_dir)
        
        # Load metadata
        with open(self.output_dir / 'metadata.json', 'r') as f:
            self.metadata = json.load(f)
        
        # Load prototypes
        self.prototypes = np.load(self.output_dir / 'class_prototypes.npy')
        
        # Load alignments
        self.meta_alignment = np.load(self.output_dir / 'meta_cluster_alignment.npy')
        self.micro_alignment = np.load(self.output_dir / 'micro_cluster_alignment.npy')
        
        # Set style
        sns.set_style("whitegrid")
        plt.rcParams['figure.figsize'] = (12, 6)
    
    def plot_class_distribution(self, save_path: Optional[str] = None):
        """Plot class distribution bar chart."""
        class_names = self.metadata['class_names']
        distribution = self.metadata['class_distribution']
        
        counts = [distribution[name] for name in class_names]
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(class_names, counts, color=sns.color_palette("husl", len(class_names)))
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Number of Images', fontsize=12)
        plt.title('Class Distribution', fontsize=14, fontweight='bold')
        plt.xticks(rotation=45, ha='right')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{int(height)}',
                    ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_confidence_distribution(self, save_path: Optional[str] = None):
        """Plot confidence score distribution."""
        confidences = [ann['confidence'] for ann in self.metadata['image_annotations']]
        class_names = self.metadata['class_names']
        
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
        
        # Overall histogram
        ax1.hist(confidences, bins=30, color='skyblue', edgecolor='black', alpha=0.7)
        ax1.axvline(np.mean(confidences), color='red', linestyle='--', 
                   label=f'Mean: {np.mean(confidences):.3f}')
        ax1.set_xlabel('Confidence Score', fontsize=12)
        ax1.set_ylabel('Frequency', fontsize=12)
        ax1.set_title('Overall Confidence Distribution', fontsize=13, fontweight='bold')
        ax1.legend()
        ax1.grid(alpha=0.3)
        
        # Per-class box plot
        class_confidences = {name: [] for name in class_names}
        for ann in self.metadata['image_annotations']:
            class_confidences[ann['class_name']].append(ann['confidence'])
        
        data_for_boxplot = [class_confidences[name] for name in class_names]
        bp = ax2.boxplot(data_for_boxplot, labels=class_names, patch_artist=True)
        
        # Color boxes
        colors = sns.color_palette("husl", len(class_names))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
        
        ax2.set_xlabel('Class', fontsize=12)
        ax2.set_ylabel('Confidence Score', fontsize=12)
        ax2.set_title('Confidence by Class', fontsize=13, fontweight='bold')
        ax2.tick_params(axis='x', rotation=45)
        ax2.grid(alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_learning_convergence(self, save_path: Optional[str] = None):
        """Plot learning convergence over iterations."""
        history = self.metadata['learning_history']
        
        iterations = [h['iteration'] for h in history]
        changes = [h['prototype_change'] for h in history]
        
        plt.figure(figsize=(10, 6))
        plt.plot(iterations, changes, marker='o', linewidth=2, markersize=8, color='#2E86AB')
        plt.xlabel('Iteration', fontsize=12)
        plt.ylabel('Prototype Change (L2 norm)', fontsize=12)
        plt.title('Learning Convergence', fontsize=14, fontweight='bold')
        plt.grid(alpha=0.3)
        plt.xticks(iterations)
        
        # Add exponential trend if applicable
        if len(changes) > 2:
            z = np.polyfit(iterations, np.log(np.array(changes) + 1e-10), 1)
            p = np.poly1d(z)
            plt.plot(iterations, np.exp(p(iterations)), '--', color='red', 
                    alpha=0.5, label='Exponential trend')
            plt.legend()
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_cluster_alignment_heatmap(
        self, 
        level: str = 'meta',
        save_path: Optional[str] = None
    ):
        """
        Plot heatmap of cluster-to-class alignment.
        
        Args:
            level: 'meta' or 'micro' cluster level
            save_path: Path to save figure
        """
        if level == 'meta':
            alignment = self.meta_alignment
            title = 'Meta-Cluster to Class Alignment'
        else:
            alignment = self.micro_alignment
            title = 'Micro-Cluster to Class Alignment'
        
        class_names = self.metadata['class_names']
        
        plt.figure(figsize=(10, max(6, len(alignment) * 0.3)))
        
        sns.heatmap(
            alignment,
            xticklabels=class_names,
            yticklabels=[f"Cluster {i}" for i in range(len(alignment))],
            annot=True if len(alignment) < 20 else False,
            fmt='.2f',
            cmap='YlOrRd',
            cbar_kws={'label': 'Soft Assignment Probability'}
        )
        
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Cluster', fontsize=12)
        plt.title(title, fontsize=14, fontweight='bold')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def plot_confidence_vs_class(self, save_path: Optional[str] = None):
        """Plot confidence scores vs class assignments."""
        class_names = self.metadata['class_names']
        
        # Prepare data
        data = {name: [] for name in class_names}
        for ann in self.metadata['image_annotations']:
            data[ann['class_name']].append(ann['confidence'])
        
        # Create violin plot
        plt.figure(figsize=(12, 6))
        
        positions = range(len(class_names))
        parts = plt.violinplot(
            [data[name] for name in class_names],
            positions=positions,
            showmeans=True,
            showmedians=True
        )
        
        # Color violins
        colors = sns.color_palette("husl", len(class_names))
        for pc, color in zip(parts['bodies'], colors):
            pc.set_facecolor(color)
            pc.set_alpha(0.7)
        
        plt.xlabel('Class', fontsize=12)
        plt.ylabel('Confidence Score', fontsize=12)
        plt.title('Confidence Distribution by Class (Violin Plot)', fontsize=14, fontweight='bold')
        plt.xticks(positions, class_names, rotation=45, ha='right')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
    
    def create_summary_report(self, output_path: Optional[str] = None):
        """Create a comprehensive visualization report."""
        if output_path is None:
            output_path = self.output_dir / 'visualizations'
        
        output_path = Path(output_path)
        output_path.mkdir(exist_ok=True)
        
        print("Generating visualization report...")
        
        # Generate all plots
        print("  [1/5] Class distribution...")
        self.plot_class_distribution(output_path / 'class_distribution.png')
        plt.close()
        
        print("  [2/5] Confidence distribution...")
        self.plot_confidence_distribution(output_path / 'confidence_distribution.png')
        plt.close()
        
        print("  [3/5] Learning convergence...")
        self.plot_learning_convergence(output_path / 'learning_convergence.png')
        plt.close()
        
        print("  [4/5] Meta-cluster alignment...")
        self.plot_cluster_alignment_heatmap('meta', output_path / 'meta_cluster_alignment.png')
        plt.close()
        
        print("  [5/5] Confidence by class...")
        self.plot_confidence_vs_class(output_path / 'confidence_by_class.png')
        plt.close()
        
        print(f"\n✓ Visualization report saved to: {output_path}")
        print(f"  Generated {len(list(output_path.glob('*.png')))} plots")


def visualize_results(output_dir: str):
    """
    Convenience function to generate all visualizations.
    
    Args:
        output_dir: Directory containing auto-annotation results
    """
    visualizer = ResultsVisualizer(output_dir)
    visualizer.create_summary_report()


if __name__ == "__main__":
    import sys
    
    if len(sys.argv) < 2:
        print("Usage: python visualizer.py <output_directory>")
        print("\nExample:")
        print("  python visualizer.py demo_output")
        sys.exit(1)
    
    output_dir = sys.argv[1]
    
    if not Path(output_dir).exists():
        print(f"Error: Directory not found: {output_dir}")
        sys.exit(1)
    
    visualize_results(output_dir)
