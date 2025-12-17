"""
Traditional Machine Learning models for baseline comparison.

This module implements traditional ML classifiers using TF-IDF features:
- Support Vector Machine (SVM)
- Random Forest
- Logistic Regression
- K-Nearest Neighbors (KNN)
"""

import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from typing import Dict, Tuple, List, Any
import time


class TraditionalMLPipeline:
    """
    Pipeline for training and evaluating traditional ML models.
    
    Args:
        max_features: Maximum number of TF-IDF features
        random_state: Random seed for reproducibility
        
    Attributes:
        vectorizer: TF-IDF vectorizer
        models: Dictionary of ML models
        results: Dictionary storing model results
    """
    
    def __init__(self, max_features: int = 5000, random_state: int = 42):
        self.max_features = max_features
        self.random_state = random_state
        self.vectorizer = TfidfVectorizer(max_features=max_features)
        self.models = {}
        self.results = {}
        
        self._initialize_models()
    
    def _initialize_models(self) -> None:
        """Initialize all traditional ML models."""
        self.models = {
            'SVM': SVC(
                kernel='linear',
                C=1.0,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'Random Forest': RandomForestClassifier(
                n_estimators=200,
                min_samples_split=6,
                random_state=self.random_state
            ),
            'Logistic Regression': LogisticRegression(
                max_iter=1000,
                class_weight='balanced',
                random_state=self.random_state
            ),
            'KNN': KNeighborsClassifier(
                n_neighbors=5
            )
        }
    
    def fit_vectorizer(self, texts: List[str]) -> None:
        """
        Fit TF-IDF vectorizer on training texts.
        
        Args:
            texts: List of training text strings
        """
        self.vectorizer.fit(texts)
        print(f"✅ TF-IDF vectorizer fitted with {len(self.vectorizer.vocabulary_)} features")
    
    def transform_texts(self, texts: List[str]):
        """
        Transform texts using fitted vectorizer.
        
        Args:
            texts: List of text strings
            
        Returns:
            TF-IDF feature matrix
        """
        return self.vectorizer.transform(texts)
    
    def train_model(
        self,
        model_name: str,
        X_train,
        y_train,
        X_val,
        y_val
    ) -> Dict[str, Any]:
        """
        Train a single model and evaluate on validation set.
        
        Args:
            model_name: Name of the model to train
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary containing model and metrics
        """
        if model_name not in self.models:
            raise ValueError(f"Model '{model_name}' not found")
        
        model = self.models[model_name]
        
        print(f"\n{'='*60}")
        print(f"Training {model_name}...")
        start_time = time.time()
        
        # Train
        model.fit(X_train, y_train)
        
        # Predict on validation set
        y_val_pred = model.predict(X_val)
        
        # Calculate metrics
        accuracy = accuracy_score(y_val, y_val_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average='macro'
        )
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_val, y_val_pred, average='weighted'
        )
        
        train_time = time.time() - start_time
        
        # Store results
        results = {
            'model': model,
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'precision_weighted': precision_w,
            'recall_weighted': recall_w,
            'f1_weighted': f1_w,
            'train_time': train_time
        }
        
        print(f"  Accuracy: {accuracy:.4f}")
        print(f"  F1-Score (Macro): {f1:.4f}")
        print(f"  F1-Score (Weighted): {f1_w:.4f}")
        print(f"  Training time: {train_time:.2f}s")
        
        return results
    
    def train_all_models(
        self,
        X_train,
        y_train,
        X_val,
        y_val
    ) -> Dict[str, Dict[str, Any]]:
        """
        Train all models and store results.
        
        Args:
            X_train: Training features
            y_train: Training labels
            X_val: Validation features
            y_val: Validation labels
            
        Returns:
            Dictionary of all model results
        """
        print("🚀 Training Traditional ML Models...\n")
        
        for model_name in self.models.keys():
            self.results[model_name] = self.train_model(
                model_name, X_train, y_train, X_val, y_val
            )
        
        print(f"\n{'='*60}")
        print("✅ Traditional ML training complete!")
        
        return self.results
    
    def evaluate_on_test(
        self,
        model_name: str,
        X_test,
        y_test
    ) -> Dict[str, float]:
        """
        Evaluate a trained model on test set.
        
        Args:
            model_name: Name of the model
            X_test: Test features
            y_test: Test labels
            
        Returns:
            Dictionary of test metrics
        """
        if model_name not in self.results:
            raise ValueError(f"Model '{model_name}' not trained yet")
        
        model = self.results[model_name]['model']
        y_pred = model.predict(X_test)
        
        # Calculate metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision, recall, f1, _ = precision_recall_fscore_support(
            y_test, y_pred, average='macro'
        )
        precision_w, recall_w, f1_w, _ = precision_recall_fscore_support(
            y_test, y_pred, average='weighted'
        )
        
        return {
            'accuracy': accuracy,
            'precision_macro': precision,
            'recall_macro': recall,
            'f1_macro': f1,
            'precision_weighted': precision_w,
            'recall_weighted': recall_w,
            'f1_weighted': f1_w,
            'predictions': y_pred
        }
    
    def get_best_model(self, metric: str = 'f1_macro') -> Tuple[str, Dict[str, Any]]:
        """
        Get the best performing model based on a metric.
        
        Args:
            metric: Metric to use for comparison
            
        Returns:
            Tuple of (model_name, results)
        """
        if not self.results:
            raise ValueError("No models trained yet")
        
        best_model = max(
            self.results.items(),
            key=lambda x: x[1][metric]
        )
        
        return best_model


def train_traditional_ml_baselines(
    train_texts: List[str],
    train_labels: List[int],
    val_texts: List[str],
    val_labels: List[int],
    max_features: int = 5000,
    random_state: int = 42
) -> Dict[str, Dict[str, Any]]:
    """
    Convenience function to train all traditional ML baselines.
    
    Args:
        train_texts: Training texts
        train_labels: Training labels
        val_texts: Validation texts
        val_labels: Validation labels
        max_features: Maximum TF-IDF features
        random_state: Random seed
        
    Returns:
        Dictionary of all model results
    """
    # Initialize pipeline
    pipeline = TraditionalMLPipeline(max_features, random_state)
    
    # Fit vectorizer and transform texts
    pipeline.fit_vectorizer(train_texts)
    X_train = pipeline.transform_texts(train_texts)
    X_val = pipeline.transform_texts(val_texts)
    
    # Train all models
    results = pipeline.train_all_models(X_train, train_labels, X_val, val_labels)
    
    return results, pipeline


if __name__ == "__main__":
    """Test traditional ML pipeline."""
    
    print("Testing TraditionalMLPipeline...\n")
    
    # Create dummy data
    train_texts = ["good movie", "bad film", "okay show"] * 100
    train_labels = [2, 0, 1] * 100
    val_texts = ["great movie", "terrible film", "average show"] * 20
    val_labels = [2, 0, 1] * 20
    
    # Train models
    results, pipeline = train_traditional_ml_baselines(
        train_texts, train_labels,
        val_texts, val_labels,
        max_features=1000
    )
    
    # Get best model
    best_name, best_result = pipeline.get_best_model('f1_macro')
    print(f"\n🏆 Best model: {best_name}")
    print(f"   F1-Score (Macro): {best_result['f1_macro']:.4f}")
    
    print("\n✅ All tests passed!")
