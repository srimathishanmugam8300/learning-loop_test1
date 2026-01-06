"""
Unit tests for the Auto-Annotation Learning Engine.
"""

import numpy as np
import pytest
from pathlib import Path
import tempfile
import shutil

from auto_annotation_engine import AutoAnnotationEngine, ClusterData, ImageLevelRefinement
from data_loader import DataLoader, ClusteringPipeline
from dataset_organizer import DatasetOrganizer


class TestAutoAnnotationEngine:
    """Tests for the main engine."""
    
    def test_initialization(self):
        """Test engine initialization."""
        # Create synthetic data
        num_images = 50
        num_classes = 3
        embedding_dim = 128
        
        embeddings = np.random.randn(num_images, embedding_dim)
        embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
        
        # Create simple clusters
        micro_assignments = np.random.randint(0, 5, num_images)
        meta_assignments = np.random.randint(0, 2, num_images)
        micro_centroids = np.random.randn(5, embedding_dim)
        meta_centroids = np.random.randn(2, embedding_dim)
        
        cluster_data = ClusterData(
            micro_cluster_assignments=micro_assignments,
            meta_cluster_assignments=meta_assignments,
            micro_cluster_centroids=micro_centroids,
            meta_cluster_centroids=meta_centroids
        )
        
        class_names = ["class_a", "class_b", "class_c"]
        text_embeddings = np.random.randn(num_classes, embedding_dim)
        
        engine = AutoAnnotationEngine(
            image_embeddings=embeddings,
            cluster_data=cluster_data,
            class_names=class_names,
            text_embeddings=text_embeddings,
            num_iterations=3,
            temperature=0.1,
            verbose=False
        )
        
        assert engine.num_images == num_images
        assert engine.num_classes == num_classes
        assert engine.embedding_dim == embedding_dim
    
    def test_learning_loop(self):
        """Test the learning loop execution."""
        num_images = 50
        num_classes = 3
        embedding_dim = 128
        
        # Create synthetic data with clear cluster structure
        class_prototypes = np.random.randn(num_classes, embedding_dim)
        class_prototypes = class_prototypes / np.linalg.norm(class_prototypes, axis=1, keepdims=True)
        
        embeddings = []
        for i in range(num_images):
            class_idx = i % num_classes
            noise = np.random.randn(embedding_dim) * 0.2
            emb = class_prototypes[class_idx] + noise
            emb = emb / np.linalg.norm(emb)
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        # Create clusters
        micro_assignments = np.arange(num_images) % 10
        meta_assignments = np.arange(num_images) % 3
        micro_centroids = np.random.randn(10, embedding_dim)
        meta_centroids = np.random.randn(3, embedding_dim)
        
        cluster_data = ClusterData(
            micro_cluster_assignments=micro_assignments,
            meta_cluster_assignments=meta_assignments,
            micro_cluster_centroids=micro_centroids,
            meta_cluster_centroids=meta_centroids
        )
        
        class_names = ["a", "b", "c"]
        text_embeddings = class_prototypes.copy()
        
        engine = AutoAnnotationEngine(
            image_embeddings=embeddings,
            cluster_data=cluster_data,
            class_names=class_names,
            text_embeddings=text_embeddings,
            num_iterations=3,
            temperature=0.1,
            verbose=False
        )
        
        result = engine.run_learning_loop()
        
        # Verify results
        assert len(result.image_labels) == num_images
        assert len(result.confidence_scores) == num_images
        assert result.class_prototypes.shape == (num_classes, embedding_dim)
        assert len(result.iteration_history) == 3
        
        # Check all images are labeled
        assert np.all(result.image_labels >= 0)
        assert np.all(result.image_labels < num_classes)
        
        # Check confidences are in [0, 1]
        assert np.all(result.confidence_scores >= 0)
        assert np.all(result.confidence_scores <= 1)
    
    def test_cosine_similarity(self):
        """Test cosine similarity computation."""
        engine = self._create_minimal_engine()
        
        A = np.array([[1, 0, 0], [0, 1, 0]])
        B = np.array([[1, 0, 0], [0, 0, 1]])
        
        sim = engine._cosine_similarity(A, B)
        
        expected = np.array([[1, 0], [0, 0]])
        np.testing.assert_array_almost_equal(sim, expected)
    
    def test_softmax(self):
        """Test softmax computation."""
        engine = self._create_minimal_engine()
        
        scores = np.array([[1, 2, 3], [3, 2, 1]])
        probs = engine._softmax(scores, temperature=1.0)
        
        # Check probabilities sum to 1
        np.testing.assert_array_almost_equal(np.sum(probs, axis=1), [1, 1])
        
        # Check all positive
        assert np.all(probs >= 0)
    
    def _create_minimal_engine(self):
        """Create minimal engine for testing."""
        embeddings = np.eye(3)
        cluster_data = ClusterData(
            micro_cluster_assignments=np.array([0, 1, 2]),
            meta_cluster_assignments=np.array([0, 0, 1]),
            micro_cluster_centroids=np.eye(3),
            meta_cluster_centroids=np.array([[1, 0, 0], [0, 0, 1]])
        )
        class_names = ["a", "b"]
        text_embeddings = np.array([[1, 0, 0], [0, 1, 0]])
        
        return AutoAnnotationEngine(
            image_embeddings=embeddings,
            cluster_data=cluster_data,
            class_names=class_names,
            text_embeddings=text_embeddings,
            verbose=False
        )


class TestClusteringPipeline:
    """Tests for clustering utilities."""
    
    def test_micro_clustering(self):
        """Test micro-cluster computation."""
        # Create data with clear clusters
        np.random.seed(42)
        cluster_1 = np.random.randn(20, 10) + np.array([5, 0, 0, 0, 0, 0, 0, 0, 0, 0])
        cluster_2 = np.random.randn(20, 10) + np.array([0, 5, 0, 0, 0, 0, 0, 0, 0, 0])
        embeddings = np.vstack([cluster_1, cluster_2])
        
        try:
            assignments, centroids = ClusteringPipeline.compute_micro_clusters(
                embeddings,
                min_cluster_size=5,
                min_samples=3
            )
            
            assert len(assignments) == len(embeddings)
            assert len(centroids) > 0
            assert centroids.shape[1] == embeddings.shape[1]
            
        except ImportError:
            pytest.skip("HDBSCAN not installed")
    
    def test_meta_clustering(self):
        """Test meta-cluster computation."""
        np.random.seed(42)
        micro_centroids = np.random.randn(10, 20)
        
        try:
            assignments, centroids = ClusteringPipeline.compute_meta_clusters(
                micro_centroids,
                num_clusters=3
            )
            
            assert len(assignments) == len(micro_centroids)
            assert len(centroids) == 3
            assert centroids.shape[1] == micro_centroids.shape[1]
            
        except ImportError:
            pytest.skip("scikit-learn not installed")


class TestDatasetOrganizer:
    """Tests for dataset organization."""
    
    def test_organizer_creation(self):
        """Test organizer initialization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            organizer = DatasetOrganizer(
                output_dir=tmpdir,
                class_names=["dog", "cat"],
                overwrite=True
            )
            
            assert organizer.output_dir.exists()
            assert len(organizer.class_names) == 2
    
    def test_organize_dataset(self):
        """Test dataset organization."""
        with tempfile.TemporaryDirectory() as tmpdir:
            # Create fake images
            image_dir = Path(tmpdir) / "images"
            image_dir.mkdir()
            
            image_paths = []
            for i in range(10):
                img_path = image_dir / f"img_{i}.txt"
                img_path.write_text(f"fake image {i}")
                image_paths.append(str(img_path))
            
            # Create fake result
            from auto_annotation_engine import LearningResult
            
            result = LearningResult(
                image_labels=np.array([0, 0, 1, 1, 0, 1, 0, 1, 1, 0]),
                confidence_scores=np.random.rand(10),
                class_prototypes=np.random.randn(2, 128),
                meta_cluster_alignment=np.random.rand(3, 2),
                micro_cluster_alignment=np.random.rand(5, 2),
                iteration_history=[{'iteration': 1, 'prototype_change': 0.1}]
            )
            
            output_dir = Path(tmpdir) / "output"
            organizer = DatasetOrganizer(
                output_dir=output_dir,
                class_names=["dog", "cat"],
                overwrite=True
            )
            
            organizer.organize_dataset(
                image_paths=image_paths,
                learning_result=result,
                copy_files=True
            )
            
            # Check structure
            assert (output_dir / "dog").exists()
            assert (output_dir / "cat").exists()
            assert (output_dir / "metadata.json").exists()
            assert (output_dir / "report.txt").exists()
            assert (output_dir / "class_prototypes.npy").exists()


class TestImageLevelRefinement:
    """Tests for image-level refinement."""
    
    def test_refinement(self):
        """Test image-level prototype refinement."""
        num_images = 30
        num_classes = 3
        embedding_dim = 64
        
        # Create data with structure
        prototypes = np.random.randn(num_classes, embedding_dim)
        prototypes = prototypes / np.linalg.norm(prototypes, axis=1, keepdims=True)
        
        embeddings = []
        for i in range(num_images):
            class_idx = i % num_classes
            noise = np.random.randn(embedding_dim) * 0.3
            emb = prototypes[class_idx] + noise
            embeddings.append(emb)
        
        embeddings = np.array(embeddings)
        
        refined = ImageLevelRefinement.refine_at_image_level(
            image_embeddings=embeddings,
            class_prototypes=prototypes,
            temperature=0.1,
            top_k_per_class=10
        )
        
        assert refined.shape == (num_classes, embedding_dim)
        # Check normalized
        norms = np.linalg.norm(refined, axis=1)
        np.testing.assert_array_almost_equal(norms, np.ones(num_classes))


def run_integration_test():
    """Run a complete integration test."""
    print("\nRunning integration test...")
    
    from example_usage import create_synthetic_demo_data, run_complete_pipeline
    
    with tempfile.TemporaryDirectory() as tmpdir:
        # Create synthetic data
        embeddings_path, image_paths_file = create_synthetic_demo_data(
            num_images=30,
            num_classes=3,
            embedding_dim=128,
            output_dir=str(Path(tmpdir) / "data")
        )
        
        # Run pipeline
        output_dir = str(Path(tmpdir) / "output")
        
        run_complete_pipeline(
            embeddings_path=embeddings_path,
            image_paths_file=image_paths_file,
            class_names=["a", "b", "c"],
            output_dir=output_dir,
            num_iterations=3,
            temperature=0.1,
            text_embedding_method="clip",  # Will fall back to random
            verbose=False
        )
        
        # Verify output
        output_path = Path(output_dir)
        assert (output_path / "a").exists()
        assert (output_path / "b").exists()
        assert (output_path / "c").exists()
        assert (output_path / "metadata.json").exists()
        assert (output_path / "report.txt").exists()
        assert (output_path / "splits").exists()
        
        print("✓ Integration test passed!")


if __name__ == "__main__":
    # Run tests
    print("Running unit tests...")
    pytest.main([__file__, "-v"])
    
    # Run integration test
    run_integration_test()
