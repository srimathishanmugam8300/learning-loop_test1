"""
Main entry point for the Auto-Annotation Learning Engine.

Provides a simple command-line interface for running the pipeline.
"""

import argparse
import json
import sys
from pathlib import Path

from example_usage import run_complete_pipeline, create_synthetic_demo_data
from config import (
    AutoAnnotationConfig,
    get_config_for_dataset_size,
    SMALL_DATASET_CONFIG,
    MEDIUM_DATASET_CONFIG,
    LARGE_DATASET_CONFIG
)


def main():
    parser = argparse.ArgumentParser(
        description="Auto-Annotation Learning Engine - Automatic dataset organization",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Run with demo data
  python main.py --demo
  
  # Run with your data
  python main.py \\
    --embeddings data/embeddings.npy \\
    --image-paths data/image_paths.txt \\
    --classes dog cat tiger lion giraffe \\
    --output annotated_dataset
  
  # Use pre-computed clusters
  python main.py \\
    --embeddings data/embeddings.npy \\
    --image-paths data/image_paths.txt \\
    --classes dog cat tiger lion giraffe \\
    --micro-assignments data/micro_assignments.npy \\
    --meta-assignments data/meta_assignments.npy \\
    --micro-centroids data/micro_centroids.npy \\
    --meta-centroids data/meta_centroids.npy \\
    --output annotated_dataset
        """
    )
    
    # Data inputs
    parser.add_argument(
        "--embeddings",
        type=str,
        help="Path to image embeddings (.npy file)"
    )
    parser.add_argument(
        "--image-paths",
        type=str,
        help="Path to file containing image paths"
    )
    parser.add_argument(
        "--classes",
        nargs="+",
        help="List of class names (space-separated)"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="output",
        help="Output directory (default: output)"
    )
    
    # Optional cluster data
    parser.add_argument(
        "--micro-assignments",
        type=str,
        help="Path to micro-cluster assignments (.npy)"
    )
    parser.add_argument(
        "--meta-assignments",
        type=str,
        help="Path to meta-cluster assignments (.npy)"
    )
    parser.add_argument(
        "--micro-centroids",
        type=str,
        help="Path to micro-cluster centroids (.npy)"
    )
    parser.add_argument(
        "--meta-centroids",
        type=str,
        help="Path to meta-cluster centroids (.npy)"
    )
    
    # Learning parameters
    parser.add_argument(
        "--iterations",
        type=int,
        default=5,
        help="Number of learning iterations (default: 5)"
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="Temperature for softmax (default: 0.1)"
    )
    parser.add_argument(
        "--text-embedding-method",
        choices=["clip", "sentence_transformer"],
        default="clip",
        help="Method for generating text embeddings (default: clip)"
    )
    
    # Demo mode
    parser.add_argument(
        "--demo",
        action="store_true",
        help="Run with synthetic demo data"
    )
    parser.add_argument(
        "--demo-images",
        type=int,
        default=100,
        help="Number of images in demo (default: 100)"
    )
    parser.add_argument(
        "--demo-classes",
        type=int,
        default=3,
        help="Number of classes in demo (default: 3)"
    )
    
    # Other options
    parser.add_argument(
        "--preset",
        choices=["small", "medium", "large"],
        help="Use preset configuration for dataset size"
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress detailed output"
    )
    
    args = parser.parse_args()
    
    # Demo mode
    if args.demo:
        print("\n" + "="*70)
        print("RUNNING IN DEMO MODE")
        print("="*70)
        print(f"Creating synthetic dataset with {args.demo_images} images...")
        print()
        
        embeddings_path, image_paths_file = create_synthetic_demo_data(
            num_images=args.demo_images,
            num_classes=args.demo_classes,
            embedding_dim=512,
            output_dir="demo_data"
        )
        
        class_names = [f"class_{i}" for i in range(args.demo_classes)]
        output_dir = "demo_output"
        
    else:
        # Validate required arguments
        if not args.embeddings or not args.image_paths or not args.classes:
            parser.error(
                "When not using --demo, you must provide: "
                "--embeddings, --image-paths, and --classes"
            )
        
        embeddings_path = args.embeddings
        image_paths_file = args.image_paths
        class_names = args.classes
        output_dir = args.output
        
        # Verify files exist
        if not Path(embeddings_path).exists():
            print(f"Error: Embeddings file not found: {embeddings_path}")
            sys.exit(1)
        if not Path(image_paths_file).exists():
            print(f"Error: Image paths file not found: {image_paths_file}")
            sys.exit(1)
    
    # Apply preset if specified
    if args.preset:
        if args.preset == "small":
            config = SMALL_DATASET_CONFIG
        elif args.preset == "medium":
            config = MEDIUM_DATASET_CONFIG
        else:  # large
            config = LARGE_DATASET_CONFIG
        
        iterations = config.num_iterations
        temperature = config.temperature
        print(f"\nUsing preset configuration: {args.preset.upper()}_DATASET_CONFIG")
    else:
        iterations = args.iterations
        temperature = args.temperature
    
    # Run pipeline
    try:
        run_complete_pipeline(
            embeddings_path=embeddings_path,
            image_paths_file=image_paths_file,
            class_names=class_names,
            output_dir=output_dir,
            micro_assignments_path=args.micro_assignments,
            meta_assignments_path=args.meta_assignments,
            micro_centroids_path=args.micro_centroids,
            meta_centroids_path=args.meta_centroids,
            num_iterations=iterations,
            temperature=temperature,
            text_embedding_method=args.text_embedding_method,
            verbose=not args.quiet
        )
        
        print("\n" + "="*70)
        print("SUCCESS!")
        print("="*70)
        print(f"\nYour annotated dataset is ready at: {output_dir}")
        print("\nNext steps:")
        print(f"  1. Review the report: {output_dir}/report.txt")
        print(f"  2. Check metadata: {output_dir}/metadata.json")
        print(f"  3. Use the organized dataset in {output_dir}/splits/")
        print()
        
    except Exception as e:
        print(f"\n❌ Error: {e}")
        if not args.quiet:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
