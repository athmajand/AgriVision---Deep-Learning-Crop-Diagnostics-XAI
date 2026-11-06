import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Dict

def load_and_resize_image(image_path: str, target_size: Tuple[int, int] = (256, 256)) -> np.ndarray:
    """Loads an image from disk and resizes it to target dimensions."""
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"Unable to read image at: {image_path}")
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img_rgb, target_size, interpolation=cv2.INTER_AREA)
    return img_resized

def convert_to_lab(image_rgb: np.ndarray) -> np.ndarray:
    """Converts an RGB image to L*a*b* color space for robust color segmentation."""
    return cv2.cvtColor(image_rgb, cv2.COLOR_RGB2LAB)

def segment_leaf_kmeans(image_rgb: np.ndarray, k: int = 3) -> Tuple[np.ndarray, np.ndarray]:
    """
    Applies k-Means clustering on L*a*b* space to segment leaf tissue and diseased regions.
    Returns segmented image and cluster labels.
    """
    lab_img = convert_to_lab(image_rgb)
    pixel_values = lab_img.reshape((-1, 3))
    pixel_values = np.float32(pixel_values)
    
    # Define k-means criteria
    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.2)
    _, labels, centers = cv2.kmeans(pixel_values, k, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
    
    centers = np.uint8(centers)
    segmented_data = centers[labels.flatten()]
    segmented_lab = segmented_data.reshape(lab_img.shape)
    segmented_rgb = cv2.cvtColor(segmented_lab, cv2.COLOR_LAB2RGB)
    
    return segmented_rgb, labels.reshape(image_rgb.shape[:2])

def calculate_defect_percentage(total_leaf_mask: np.ndarray, defect_mask: np.ndarray) -> Dict[str, float]:
    """
    Calculates disease severity percentage based on leaf surface area vs. affected area.
    Formula: (A1 / A2) * 100
    """
    total_leaf_area = np.count_nonzero(total_leaf_mask)
    diseased_area = np.count_nonzero(defect_mask)
    
    if total_leaf_area == 0:
        percentage = 0.0
    else:
        percentage = (diseased_area / total_leaf_area) * 100.0
        
    return {
        "total_leaf_pixels": int(total_leaf_area),
        "diseased_pixels": int(diseased_area),
        "defect_percentage": round(percentage, 2)
    }