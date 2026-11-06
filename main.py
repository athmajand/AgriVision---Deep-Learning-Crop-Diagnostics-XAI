import argparse
import logging
import pandas as pd
import numpy as np
import json
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Internal modular imports
from src.utils import load_config, setup_logger, set_seeds
from src.feature_extraction import batch_extract_dataset
from src.models.knn_model import PlantDiseaseKNN
from src.models.cnn_model import build_cnn_model

# Target the newly merged universal dataset
TARGET_DATA_DIR = "data/raw/unified_dataset"

def run_preprocessing(config: dict):
    """Executes the GLCM feature extraction pipeline."""
    output_csv = f"{config['dataset']['extracted_features_dir']}/glcm_features.csv"
    logging.info(f"Starting feature extraction from {TARGET_DATA_DIR}...")
    batch_extract_dataset(data_dir=TARGET_DATA_DIR, output_csv=output_csv)
    logging.info("Preprocessing and feature extraction complete.")

def run_knn_pipeline(config: dict):
    """Loads GLCM features, splits data 75:25, and trains the KNN model."""
    feature_csv = f"{config['dataset']['extracted_features_dir']}/glcm_features.csv"
    if not Path(feature_csv).exists():
        logging.error("Feature CSV not found. Run '--mode preprocess' first.")
        return

    logging.info("Loading extracted features for KNN...")
    df = pd.read_csv(feature_csv)
    X = df.drop(columns=['filename', 'label']).values
    y_raw = df['label'].values
    
    le = LabelEncoder()
    y = le.fit_transform(y_raw)
    
    split_ratio = config['dataset'].get('test_split', 0.25)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=split_ratio, random_state=42)
    
    knn_params = config['knn_params']
    knn = PlantDiseaseKNN(n_neighbors=knn_params['n_neighbors'], metric=knn_params['metric'])
    knn.train(X_train, y_train)
    knn.evaluate(X_test, y_test)
    knn.save_model(config['paths']['knn_model_save'])

def run_cnn_pipeline(config: dict):
    """Sets up data generators, dynamically detects classes, and trains the CNN."""
    cnn_params = config['cnn_params']
    img_size = tuple(config['dataset']['image_size'])
    
    logging.info(f"Initializing CNN Image Data Generators from {TARGET_DATA_DIR}...")
    datagen = ImageDataGenerator(rescale=1./255, validation_split=config['dataset']['test_split'])
    
    train_generator = datagen.flow_from_directory(
        TARGET_DATA_DIR, target_size=img_size, batch_size=cnn_params['batch_size'],
        class_mode='categorical', subset='training'
    )
    val_generator = datagen.flow_from_directory(
        TARGET_DATA_DIR, target_size=img_size, batch_size=cnn_params['batch_size'],
        class_mode='categorical', subset='validation'
    )
    
    # --- DYNAMIC CLASS MAPPING FOR UI ---
    class_indices = train_generator.class_indices
    class_mapping = {str(v): k for k, v in class_indices.items()}
    Path("saved_models").mkdir(parents=True, exist_ok=True)
    with open("saved_models/class_mapping.json", "w") as f:
        json.dump(class_mapping, f)
        
    num_classes = len(class_indices)
    logging.info(f"Dynamically detected {num_classes} disease classes. Mapping saved to JSON.")
    
    logging.info("Building Scaled CNN Architecture...")
    model = build_cnn_model(input_shape=(img_size[0], img_size[1], 3), num_classes=num_classes)
    
    logging.info(f"Starting Training for {cnn_params['epochs']} epochs...")
    model.fit(train_generator, validation_data=val_generator, epochs=cnn_params['epochs'])
    
    save_path = config['paths']['cnn_model_save']
    model.save(save_path)
    logging.info(f"CNN Model training complete and saved to {save_path}")

def run_explain_pipeline(config: dict):
    """Generates LIME Explainability masks for a sample prediction."""
    import cv2
    from tensorflow.keras.models import load_model
    from src.explainability.lime_explainer import LimeExplainer
    
    model_path = config['paths']['cnn_model_save']
    if not Path(model_path).exists():
        logging.error("CNN model not found. Run '--mode train_cnn' first.")
        return
        
    model = load_model(model_path)
    
    raw_dir = Path(TARGET_DATA_DIR)
    sample_images = list(raw_dir.rglob("*.[jJ][pP][gG]"))
    if not sample_images:
        logging.error(f"No sample images found in {TARGET_DATA_DIR} to explain.")
        return
        
    sample_image_path = str(sample_images[0])
    img_size = tuple(config['dataset']['image_size'])
    
    img = cv2.imread(sample_image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img_resized = cv2.resize(img, img_size)
    
    def predict_fn(images):
        images_scaled = np.array(images) / 255.0
        return model.predict(images_scaled, verbose=0)
    
    logging.info(f"Generating LIME perturbations for {sample_image_path}...")
    explainer = LimeExplainer()
    xai_params = config['xai']
    
    explanation = explainer.explain_prediction(
        predict_fn, img_resized, num_samples=xai_params['num_samples'], top_labels=xai_params['top_labels']
    )
    
    explainer.save_explanation_mask(explanation, img_resized, "lime_mask1_positive.png", positive_only=True, num_features=5, hide_rest=True)
    explainer.save_explanation_mask(explanation, img_resized, "lime_mask2_all.png", positive_only=False, num_features=10, hide_rest=False)
    logging.info("XAI generation complete. Check 'outputs/plots/'.")

def main():
    parser = argparse.ArgumentParser(description="Plant Disease Recognition Enterprise Pipeline")
    parser.add_argument("--mode", type=str, required=True, choices=["preprocess", "train_knn", "train_cnn", "explain"])
    args = parser.parse_args()

    config = load_config()
    setup_logger(config['paths']['logs'])
    set_seeds(42)
    
    logging.info(f"--- Pipeline Triggered: {args.mode.upper()} ---")

    if args.mode == "preprocess":
        run_preprocessing(config)
    elif args.mode == "train_knn":
        run_knn_pipeline(config)
    elif args.mode == "train_cnn":
        run_cnn_pipeline(config)
    elif args.mode == "explain":
        run_explain_pipeline(config)

if __name__ == "__main__":
    main()