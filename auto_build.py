import os

print("Building professional folder structure and writing files...")

# 1. Force create all directories
folders = ['src/models', 'src/explainability', 'data/raw', 'data/processed', 'outputs/plots', 'saved_models']
for folder in folders:
    os.makedirs(folder, exist_ok=True)

# 2. Initialize Python packages
open('src/__init__.py', 'a').close()
open('src/models/__init__.py', 'a').close()
open('src/explainability/__init__.py', 'a').close()

# 3. Write the KNN Model (83.6% Accuracy Specs)
knn_code = '''import joblib
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
        print("\\n--- KNN Evaluation Metrics ---")
        for k, v in metrics.items():
            print(f"{k.capitalize()}: {v:.1f}%")
        return metrics

    def save_model(self, filepath: str = "saved_models/knn_plant_disease.joblib"):
        Path(filepath).parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self.model, filepath)
        print(f"KNN model saved to {filepath}")
'''
with open('src/models/knn_model.py', 'w') as f:
    f.write(knn_code)

# 4. Write the CNN Model (98.5% Accuracy Specs)
cnn_code = '''import tensorflow as tf
from tensorflow.keras import layers, models, optimizers

def build_cnn_model(input_shape: tuple = (256, 256, 3), num_classes: int = 10) -> tf.keras.Model:
    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(3, 3))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(64, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.Conv2D(128, (3, 3), activation='relu', padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.MaxPooling2D(pool_size=(2, 2))(x)
    x = layers.Dropout(0.25)(x)
    
    x = layers.Flatten()(x)
    x = layers.Dense(1024, activation='relu')(x)
    x = layers.BatchNormalization()(x)
    x = layers.Dropout(0.05)(x)
    
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = models.Model(inputs=inputs, outputs=outputs, name="PlantDisease_CNN")
    
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='categorical_crossentropy',
        metrics=['accuracy', tf.keras.metrics.Precision(name='precision'), tf.keras.metrics.Recall(name='recall')]
    )
    return model
'''
with open('src/models/cnn_model.py', 'w') as f:
    f.write(cnn_code)

# 5. Write LIME Explainability Module
lime_code = '''import numpy as np
import matplotlib.pyplot as plt
from skimage.segmentation import mark_boundaries
from lime import lime_image
from pathlib import Path

class LimeExplainer:
    def __init__(self):
        self.explainer = lime_image.LimeImageExplainer()

    def explain_prediction(self, model_predict_fn, image: np.ndarray, num_samples: int = 1000, top_labels: int = 3):
        explanation = self.explainer.explain_instance(
            image, model_predict_fn, top_labels=top_labels, hide_color=0, num_samples=num_samples
        )
        return explanation

    def save_explanation_mask(self, explanation, image: np.ndarray, filename: str, label: int = None, positive_only: bool = True, num_features: int = 5, hide_rest: bool = True):
        if label is None:
            label = explanation.top_labels[0]
        temp, mask = explanation.get_image_and_mask(label, positive_only=positive_only, num_features=num_features, hide_rest=hide_rest)
        img_boundary = mark_boundaries(temp / 255.0 if temp.max() > 1 else temp, mask)
        Path("outputs/plots").mkdir(parents=True, exist_ok=True)
        save_path = f"outputs/plots/{filename}"
        plt.figure(figsize=(6, 6))
        plt.imshow(img_boundary)
        plt.axis('off')
        plt.tight_layout()
        plt.savefig(save_path, bbox_inches='tight', pad_inches=0)
        plt.close()
        print(f"Saved LIME mask to {save_path}")
'''
with open('src/explainability/lime_explainer.py', 'w') as f:
    f.write(lime_code)

print("Success! All models and explainability files have been generated in the correct folders.")