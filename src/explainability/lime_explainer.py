import numpy as np
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
