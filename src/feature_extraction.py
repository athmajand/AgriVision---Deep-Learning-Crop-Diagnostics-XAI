import cv2
import numpy as np
import pandas as pd
from scipy.stats import skew, kurtosis
from skimage.feature import graycomatrix, graycoprops
from pathlib import Path
from typing import Dict, List

def extract_glcm_features(
    image_gray: np.ndarray, 
    distances: List[int] = [1], 
    angles: List[float] = [0, np.pi/4, np.pi/2, 3*np.pi/4]
) -> Dict[str, float]:
    """Computes 13 statistical and GLCM texture features from a grayscale image array."""
    if len(image_gray.shape) == 3:
        image_gray = cv2.cvtColor(image_gray, cv2.COLOR_RGB2GRAY)

    glcm = graycomatrix(image_gray, distances=distances, angles=angles, levels=256, symmetric=True, normed=True)

    contrast = graycoprops(glcm, 'contrast').mean()
    correlation = graycoprops(glcm, 'correlation').mean()
    energy = graycoprops(glcm, 'energy').mean()
    homogeneity = graycoprops(glcm, 'homogeneity').mean()

    # Inverse Difference Moment (IDM)
    i_idx, j_idx = np.ogrid[:256, :256]
    idm_weights = 1.0 / (1.0 + (i_idx - j_idx) ** 2)
    idm = np.mean([np.sum(glcm[:, :, d, a] * idm_weights) for d in range(len(distances)) for a in range(len(angles))])

    pixels = image_gray.flatten().astype(np.float64)
    variance_val = np.var(pixels)

    p_norm = glcm / (np.sum(glcm) + 1e-12)
    p_nz = p_norm[p_norm > 0]

    return {
        "contrast": float(contrast),
        "correlation": float(correlation),
        "energy": float(energy),
        "homogeneity": float(homogeneity),
        "idm": float(idm),
        "mean": float(np.mean(pixels)),
        "std_dev": float(np.std(pixels)),
        "variance": float(variance_val),
        "rms": float(np.sqrt(np.mean(pixels ** 2))),
        "smoothness": float(1.0 - (1.0 / (1.0 + variance_val))),
        "skewness": float(skew(pixels)),
        "kurtosis": float(kurtosis(pixels)),
        "entropy": float(-np.sum(p_nz * np.log2(p_nz)))
    }

def batch_extract_dataset(data_dir: str, output_csv: str = "data/extracted_features/glcm_features.csv"):
    """
    Iterates through dataset folders, extracts features for every leaf image,
    and saves the tabular dataset for KNN model training.
    """
    root_path = Path(data_dir)
    records = []

    if not root_path.exists():
        print(f"Warning: Data directory {data_dir} does not exist yet. Please place your PlantVillage folders inside data/raw/")
        return

    for class_folder in root_path.iterdir():
        if class_folder.is_dir():
            label = class_folder.name
            for img_path in class_folder.glob("*.[jJ][pP][gG]"):
                img = cv2.imread(str(img_path), cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    features = extract_glcm_features(img)
                    features["filename"] = img_path.name
                    features["label"] = label
                    records.append(features)

    if records:
        df = pd.DataFrame(records)
        Path(output_csv).parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(output_csv, index=False)
        print(f"Extracted {len(df)} samples and saved feature matrix to: {output_csv}")
    else:
        print(f"No images found in {data_dir}. Ensure you have JPG files in your class folders.")