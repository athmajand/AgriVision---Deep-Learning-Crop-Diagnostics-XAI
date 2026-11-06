import joblib
import matplotlib.pyplot as plt
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np
from pathlib import Path

class PlantDiseaseKNN:
    def __init__(self, n_neighbors: int = 21, metric: str = 'euclidean'):
        self.model = KNeighborsClassifier(n_neighbors=n_neighbors, metric=metric)
        self.k_range = range(1, 26)
        
    def train(self, X_train: np.ndarray, y_train: np.ndarray):
        print(f"Training KNN with k={self.model.n_neighbors}...")
        self.model.fit(X_train, y_train)
        
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> dict:
        predictions = self.model.predict(X_test)
        metrics = {
            "accuracy": accuracy_score(y_test, predictions) * 100,
            "precision": precision_score(y_test, predictions, average='macro', zero_division=0) * 100,
            "recall": recall_score(y_test, predictions, average='macro', zero_division=0) * 100,
            "f1_score": f1_score(y_test, predictions, average='macro', zero_division=0) * 100
        }
        print("\n--- KNN Evaluation Metrics ---")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.1f}%")
        return metrics

    def save_model(self, filepath: str = "saved_models/knn_plant_disease.joblib"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"KNN model saved to {filepath}")
