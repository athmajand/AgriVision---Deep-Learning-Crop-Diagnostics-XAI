# AgriVision: Deep Learning Crop Diagnostics & XAI

An enterprise-grade, native Windows desktop application for advanced plant pathology. This engine leverages a custom Convolutional Neural Network (CNN) to classify 49 distinct crop disease profiles and integrates Explainable AI (LIME) to provide transparent, visual reasoning for every diagnosis.

## 🧠 Core Architecture
*   **Deep Learning Engine:** Multi-block CNN trained to identify 49 classes of plant diseases across varied crop species.
*   **Explainable AI (XAI):** Integrated `lime_image` pipeline generates real-time heatmaps, isolating the exact necrotic lesions and feature vectors driving the model's predictions.
*   **Native Desktop UI:** Built with `CustomTkinter`, entirely bypassing browser dependencies (e.g., Streamlit). Features multi-threading to prevent UI freezing during heavy LIME computations and includes dynamic Light/Dark theme rendering.
*   **Dynamic Data Mapping:** Autonomous JSON router handles class mapping, eliminating hardcoded classification constraints and enabling seamless dataset scaling.

## 🚀 Key Features
*   **Browser-Free Execution:** Full native OS graphics handling.
*   **Dual-Theme Interface:** Seamless real-time switching between Light and Dark modes.
*   **Feature Isolation:** Side-by-side LIME masking (Positive Feature Isolation vs. Full Decision Contrast).

## ⚙️ Tech Stack
`Python` | `TensorFlow/Keras` | `CustomTkinter` | `OpenCV` | `Scikit-Image` | `LIME`

## 📥 Deployment
Ensure your virtual environment is active, then execute:
```cmd
python ui/app.py
