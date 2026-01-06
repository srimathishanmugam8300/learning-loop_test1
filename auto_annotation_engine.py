"""
Auto-Annotation Learning Engine

A fully automatic system that learns to align unlabeled image datasets
with user-defined class names through iterative prototype refinement.

Works for ANY dataset size (small or large) and ANY domain.
NO thresholds. NO human review. Maximum-likelihood assignments only.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
import json


@dataclass
class ClusterData:
    """Container for clustering information."""
    micro_cluster_assignments: np.ndarray  # [N] micro-cluster ID per image
    meta_cluster_assignments: np.ndarray   # [N] meta-cluster ID per image
    micro_cluster_centroids: np.ndarray    # [M, D] micro-cluster centroids
    meta_cluster_centroids: np.ndarray     # [K, D] meta-cluster centroids


@dataclass
class LearningResult:
    """Results from the learning loop."""
    image_labels: np.ndarray               # [N] class index per image
    confidence_scores: np.ndarray          # [N] confidence per image
    class_prototypes: np.ndarray           # [C, D] final prototypes
    meta_cluster_alignment: np.ndarray     # [K, C] soft alignment
    micro_cluster_alignment: np.ndarray    # [M, C] soft alignment
    iteration_history: List[Dict]          # tracking per iteration


class AutoAnnotationEngine:
    """
    Auto-Annotation Learning Engine
    
    Automatically organizes and labels unlabeled image datasets using
    a learning loop that aligns image embeddings with user-defined classes.
    """
    
    def __init__(
        self,
        image_embeddings: np.ndarray,
        cluster_data: ClusterData,
        class_names: List[str],
        text_embeddings: np.ndarray,
        num_iterations: int = 5,
        temperature: float = 0.1,
        verbose: bool = True
    ):
        """
        Initialize the Auto-Annotation Engine.
        
        Args:
            image_embeddings: [N, D] normalized image embeddings
            cluster_data: ClusterData object with clustering information
            class_names: List of target class names
            text_embeddings: [C, D] normalized text embeddings for each class
            num_iterations: Number of learning iterations (default: 5)
            temperature: Temperature for softmax (lower = sharper, default: 0.1)
            verbose: Print progress information
        """
        self.image_embeddings = self._normalize(image_embeddings)
        self.cluster_data = cluster_data
        self.class_names = class_names
        self.num_classes = len(class_names)
        self.num_iterations = num_iterations
        self.temperature = temperature
        self.verbose = verbose
        
        # Initialize class prototypes from text embeddings
        self.class_prototypes = self._normalize(text_embeddings)
        
        # Verify shapes
        self.num_images = len(image_embeddings)
        self.embedding_dim = image_embeddings.shape[1]
        
        assert self.class_prototypes.shape[0] == self.num_classes
        assert self.class_prototypes.shape[1] == self.embedding_dim
        
        # History tracking
        self.iteration_history = []
        
    def _normalize(self, vectors: np.ndarray) -> np.ndarray:
        """Normalize vectors to unit length."""
        norms = np.linalg.norm(vectors, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)  # Avoid division by zero
        return vectors / norms
    
    def _cosine_similarity(self, A: np.ndarray, B: np.ndarray) -> np.ndarray:
        """
        Compute cosine similarity between two sets of vectors.
        
        Args:
            A: [N, D] vectors
            B: [M, D] vectors
            
        Returns:
            [N, M] similarity matrix
        """
        return A @ B.T
    
    def _softmax(self, scores: np.ndarray, temperature: float = 1.0) -> np.ndarray:
        """
        Compute softmax with temperature scaling.
        
        Args:
            scores: [N, C] similarity scores
            temperature: Temperature parameter
            
        Returns:
            [N, C] soft responsibilities
        """
        scaled_scores = scores / temperature
        # Numerical stability
        scaled_scores = scaled_scores - np.max(scaled_scores, axis=1, keepdims=True)
        exp_scores = np.exp(scaled_scores)
        return exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
    
    def _compute_soft_alignment(
        self,
        centroids: np.ndarray,
        prototypes: np.ndarray
    ) -> np.ndarray:
        """
        Compute soft alignment between centroids and class prototypes.
        
        Args:
            centroids: [K, D] cluster centroids
            prototypes: [C, D] class prototypes
            
        Returns:
            [K, C] soft responsibilities gamma(k, c)
        """
        similarities = self._cosine_similarity(centroids, prototypes)
        responsibilities = self._softmax(similarities, self.temperature)
        return responsibilities
    
    def _update_prototypes(
        self,
        centroids: np.ndarray,
        responsibilities: np.ndarray
    ) -> np.ndarray:
        """
        Update class prototypes using weighted average of centroids.
        
        Args:
            centroids: [K, D] cluster centroids
            responsibilities: [K, C] soft responsibilities
            
        Returns:
            [C, D] updated prototypes
        """
        # P_c = Σ_k gamma(k,c) * C_k / Σ_k gamma(k,c)
        numerator = responsibilities.T @ centroids  # [C, D]
        denominator = np.sum(responsibilities, axis=0, keepdims=True).T  # [C, 1]
        denominator = np.where(denominator == 0, 1, denominator)  # Avoid division by zero
        
        new_prototypes = numerator / denominator
        return self._normalize(new_prototypes)
    
    def run_learning_loop(self) -> LearningResult:
        """
        Run the iterative learning loop to refine class prototypes.
        
        Returns:
            LearningResult with final labels and alignment information
        """
        if self.verbose:
            print(f"\n{'='*60}")
            print(f"AUTO-ANNOTATION LEARNING ENGINE")
            print(f"{'='*60}")
            print(f"Dataset size: {self.num_images} images")
            print(f"Target classes: {self.num_classes} ({', '.join(self.class_names)})")
            print(f"Meta-clusters: {len(self.cluster_data.meta_cluster_centroids)}")
            print(f"Micro-clusters: {len(self.cluster_data.micro_cluster_centroids)}")
            print(f"Iterations: {self.num_iterations}")
            print(f"Temperature: {self.temperature}")
            print(f"{'='*60}\n")
        
        # Run iterations
        for iteration in range(self.num_iterations):
            if self.verbose:
                print(f"Iteration {iteration + 1}/{self.num_iterations}")
            
            # Step 1: Meta-cluster alignment
            meta_alignment = self._compute_soft_alignment(
                self.cluster_data.meta_cluster_centroids,
                self.class_prototypes
            )
            
            # Step 2: Update prototypes from meta-clusters
            updated_prototypes = self._update_prototypes(
                self.cluster_data.meta_cluster_centroids,
                meta_alignment
            )
            
            # Step 3: Micro-cluster alignment (fine-grained refinement)
            micro_alignment = self._compute_soft_alignment(
                self.cluster_data.micro_cluster_centroids,
                updated_prototypes
            )
            
            # Step 4: Further refine prototypes from micro-clusters
            refined_prototypes = self._update_prototypes(
                self.cluster_data.micro_cluster_centroids,
                micro_alignment
            )
            
            # Track convergence
            prototype_change = np.linalg.norm(
                refined_prototypes - self.class_prototypes
            )
            
            # Update prototypes
            self.class_prototypes = refined_prototypes
            
            # Store iteration info
            iteration_info = {
                'iteration': iteration + 1,
                'prototype_change': float(prototype_change),
                'meta_alignment': meta_alignment.tolist(),
                'micro_alignment': micro_alignment.tolist()
            }
            self.iteration_history.append(iteration_info)
            
            if self.verbose:
                print(f"  Prototype change: {prototype_change:.6f}")
                print(f"  Meta-cluster dominant classes: {self._get_dominant_classes(meta_alignment)}")
                print()
        
        # Final label propagation
        if self.verbose:
            print(f"{'='*60}")
            print("FINAL LABEL PROPAGATION")
            print(f"{'='*60}\n")
        
        image_labels, confidence_scores = self._propagate_labels()
        
        # Get final alignments
        final_meta_alignment = self._compute_soft_alignment(
            self.cluster_data.meta_cluster_centroids,
            self.class_prototypes
        )
        final_micro_alignment = self._compute_soft_alignment(
            self.cluster_data.micro_cluster_centroids,
            self.class_prototypes
        )
        
        # Print summary
        if self.verbose:
            self._print_summary(image_labels, confidence_scores)
        
        return LearningResult(
            image_labels=image_labels,
            confidence_scores=confidence_scores,
            class_prototypes=self.class_prototypes,
            meta_cluster_alignment=final_meta_alignment,
            micro_cluster_alignment=final_micro_alignment,
            iteration_history=self.iteration_history
        )
    
    def _propagate_labels(self) -> Tuple[np.ndarray, np.ndarray]:
        """
        Propagate labels to all images using maximum likelihood.
        
        Returns:
            image_labels: [N] class index per image
            confidence_scores: [N] confidence per image
        """
        # Compute similarity between all images and class prototypes
        similarities = self._cosine_similarity(
            self.image_embeddings,
            self.class_prototypes
        )  # [N, C]
        
        # Maximum likelihood assignment
        image_labels = np.argmax(similarities, axis=1)
        confidence_scores = np.max(similarities, axis=1)
        
        return image_labels, confidence_scores
    
    def _get_dominant_classes(self, alignment: np.ndarray) -> str:
        """Get dominant class for each cluster."""
        dominant = np.argmax(alignment, axis=1)
        class_names_short = [name[:10] for name in self.class_names]
        return ', '.join([f"{i}→{class_names_short[c]}" for i, c in enumerate(dominant)])
    
    def _print_summary(self, labels: np.ndarray, confidences: np.ndarray):
        """Print summary statistics."""
        print("Label distribution:")
        for class_idx, class_name in enumerate(self.class_names):
            count = np.sum(labels == class_idx)
            percentage = 100 * count / len(labels)
            avg_conf = np.mean(confidences[labels == class_idx]) if count > 0 else 0
            print(f"  {class_name}: {count} images ({percentage:.1f}%) "
                  f"- avg confidence: {avg_conf:.3f}")
        
        print(f"\nOverall statistics:")
        print(f"  Mean confidence: {np.mean(confidences):.3f}")
        print(f"  Min confidence: {np.min(confidences):.3f}")
        print(f"  Max confidence: {np.max(confidences):.3f}")
        print(f"  Std confidence: {np.std(confidences):.3f}")
        print(f"\n{'='*60}\n")


class ImageLevelRefinement:
    """
    Fallback refinement for cases with very few clusters.
    
    If only one meta-cluster exists or clusters are too coarse,
    refine at the image level directly.
    """
    
    @staticmethod
    def refine_at_image_level(
        image_embeddings: np.ndarray,
        class_prototypes: np.ndarray,
        temperature: float = 0.1,
        top_k_per_class: int = 100
    ) -> np.ndarray:
        """
        Refine prototypes using top-k images per class.
        
        Args:
            image_embeddings: [N, D] image embeddings
            class_prototypes: [C, D] current prototypes
            temperature: Temperature for softmax
            top_k_per_class: Number of top images to use per class
            
        Returns:
            [C, D] refined prototypes
        """
        # Normalize
        image_embeddings = image_embeddings / np.linalg.norm(
            image_embeddings, axis=1, keepdims=True
        )
        class_prototypes = class_prototypes / np.linalg.norm(
            class_prototypes, axis=1, keepdims=True
        )
        
        # Compute similarities
        similarities = image_embeddings @ class_prototypes.T  # [N, C]
        
        # Compute soft responsibilities
        scaled_scores = similarities / temperature
        scaled_scores = scaled_scores - np.max(scaled_scores, axis=1, keepdims=True)
        exp_scores = np.exp(scaled_scores)
        responsibilities = exp_scores / np.sum(exp_scores, axis=1, keepdims=True)
        
        # Update using all images (weighted)
        new_prototypes = responsibilities.T @ image_embeddings  # [C, D]
        new_prototypes = new_prototypes / np.linalg.norm(
            new_prototypes, axis=1, keepdims=True
        )
        
        return new_prototypes
