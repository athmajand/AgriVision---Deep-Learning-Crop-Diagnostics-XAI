import customtkinter as ctk
from tkinter import filedialog
from PIL import Image
import numpy as np
import cv2
import json
import os
from tensorflow.keras.models import load_model
from lime import lime_image
from skimage.segmentation import mark_boundaries
import matplotlib.pyplot as plt

# --- UI Default Styling ---
ctk.set_appearance_mode("Dark") 
ctk.set_default_color_theme("blue")

class AgriVisionApp(ctk.CTk):
    def __init__(self):
        super().__init__()
        self.title("AgriVision Enterprise - Plant Pathology Engine")
        self.geometry("1100x750")
        
        self.model_path = "saved_models/cnn_plant_disease.h5"
        self.json_path = "saved_models/class_mapping.json"
        self.plot_dir = "outputs/plots"
        os.makedirs(self.plot_dir, exist_ok=True)
        
        self.model = None
        self.classes = {}
        self.current_image_path = None
        
        self.build_ui()
        self.load_backend()

    def build_ui(self):
        # Header
        self.header = ctk.CTkLabel(self, text="AgriVision AI - Deep Learning Crop Diagnostics & XAI", font=ctk.CTkFont(size=24, weight="bold"))
        self.header.pack(pady=15)
        
        self.main_frame = ctk.CTkFrame(self, fg_color="transparent")
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=10)
        
        # --- LEFT PANEL (Controls) ---
        self.left_panel = ctk.CTkFrame(self.main_frame, width=350)
        self.left_panel.pack(side="left", fill="y", padx=10)
        
        self.upload_btn = ctk.CTkButton(self.left_panel, text="Upload Leaf Image", command=self.upload_image, height=40)
        self.upload_btn.pack(pady=20, padx=20, fill="x")
        
        # Tuple colors map to: ("Light Mode Color", "Dark Mode Color")
        self.img_display = ctk.CTkLabel(self.left_panel, text="No Image Selected", width=300, height=300, 
                                        fg_color=("gray85", "gray20"), corner_radius=10)
        self.img_display.pack(pady=10, padx=20)
        
        self.analyze_btn = ctk.CTkButton(self.left_panel, text="Run Diagnostics & XAI", command=self.run_diagnostics, height=40, state="disabled")
        self.analyze_btn.pack(pady=20, padx=20, fill="x")
        
        self.status_label = ctk.CTkLabel(self.left_panel, text="System Initializing...", text_color=("gray30", "gray60"))
        self.status_label.pack(pady=10)
        
        # Theme Toggle Dropdown
        self.theme_label = ctk.CTkLabel(self.left_panel, text="Interface Theme:")
        self.theme_label.pack(side="bottom", pady=(0, 5))
        self.theme_menu = ctk.CTkOptionMenu(self.left_panel, values=["Dark", "Light", "System"], command=self.change_theme)
        self.theme_menu.pack(side="bottom", pady=(0, 20), padx=20, fill="x")
        
        # --- RIGHT PANEL (Results) ---
        self.right_panel = ctk.CTkFrame(self.main_frame)
        self.right_panel.pack(side="right", fill="both", expand=True, padx=10)
        
        self.result_label = ctk.CTkLabel(self.right_panel, text="Diagnosis: Pending", font=ctk.CTkFont(size=20, weight="bold"))
        self.result_label.pack(pady=15)
        
        self.conf_label = ctk.CTkLabel(self.right_panel, text="Confidence Score: --%", font=ctk.CTkFont(size=16))
        self.conf_label.pack(pady=5)
        
        self.xai_frame = ctk.CTkFrame(self.right_panel, fg_color="transparent")
        self.xai_frame.pack(fill="both", expand=True, pady=15)
        
        # Mask 1 Display
        self.mask1_frame = ctk.CTkFrame(self.xai_frame, fg_color="transparent")
        self.mask1_frame.pack(side="left", expand=True, padx=5)
        self.mask1_title = ctk.CTkLabel(self.mask1_frame, text="LIME Feature Isolation")
        self.mask1_title.pack(pady=5)
        self.mask1_display = ctk.CTkLabel(self.mask1_frame, text="", width=280, height=280, fg_color=("gray85", "gray20"), corner_radius=10)
        self.mask1_display.pack()
        
        # Mask 2 Display
        self.mask2_frame = ctk.CTkFrame(self.xai_frame, fg_color="transparent")
        self.mask2_frame.pack(side="right", expand=True, padx=5)
        self.mask2_title = ctk.CTkLabel(self.mask2_frame, text="LIME Decision Contrast")
        self.mask2_title.pack(pady=5)
        self.mask2_display = ctk.CTkLabel(self.mask2_frame, text="", width=280, height=280, fg_color=("gray85", "gray20"), corner_radius=10)
        self.mask2_display.pack()

    def load_backend(self):
        try:
            self.model = load_model(self.model_path)
            with open(self.json_path, "r") as f:
                mapping = json.load(f)
            self.classes = {int(k): v for k, v in mapping.items()}
            self.status_label.configure(text=f"Online: {len(self.classes)} Disease Profiles Active", text_color=("green", "#00FF00"))
        except Exception as e:
            self.status_label.configure(text="ERROR: AI Model Offline", text_color="red")

    def upload_image(self):
        file_path = filedialog.askopenfilename(filetypes=[("Images", "*.png *.jpg *.jpeg")])
        if file_path:
            self.current_image_path = file_path
            img = Image.open(file_path)
            img_tk = ctk.CTkImage(light_image=img, dark_image=img, size=(300, 300))
            self.img_display.configure(image=img_tk, text="")
            self.analyze_btn.configure(state="normal")
            
            self.result_label.configure(text="Diagnosis: Pending", text_color=("black", "white"))
            self.conf_label.configure(text="Confidence Score: --%")
            self.mask1_display.configure(image=None, text="")
            self.mask2_display.configure(image=None, text="")

    def run_diagnostics(self):
        self.status_label.configure(text="Processing image topology...", text_color=("blue", "yellow"))
        self.update() 
        
        img_raw = cv2.imread(self.current_image_path)
        img_rgb = cv2.cvtColor(img_raw, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (256, 256))
        img_scaled = np.expand_dims(img_resized, axis=0) / 255.0
        
        predictions = self.model.predict(img_scaled)[0]
        predicted_idx = int(np.argmax(predictions))
        predicted_name = self.classes[predicted_idx].replace("_", " ")
        confidence = np.max(predictions) * 100
        
        self.result_label.configure(text=f"Diagnosis: {predicted_name}", text_color=("green", "#00FF00"))
        self.conf_label.configure(text=f"Confidence Score: {confidence:.2f}%")
        
        self.status_label.configure(text="Rendering XAI Heatmaps (Please wait...)")
        self.update()
        
        def predict_fn(images):
            return self.model.predict(np.array(images) / 255.0, verbose=0)
            
        explainer = lime_image.LimeImageExplainer()
        explanation = explainer.explain_instance(
            img_resized, predict_fn, top_labels=3, hide_color=0, num_samples=300
        )
        
        temp1, mask1 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=True, num_features=5, hide_rest=True)
        img_bound1 = mark_boundaries(temp1 / 255.0 if temp1.max() > 1 else temp1, mask1)
        mask1_path = os.path.join(self.plot_dir, "ctk_mask1.png")
        plt.imsave(mask1_path, img_bound1)
        
        temp2, mask2 = explanation.get_image_and_mask(explanation.top_labels[0], positive_only=False, num_features=10, hide_rest=False)
        img_bound2 = mark_boundaries(temp2 / 255.0 if temp2.max() > 1 else temp2, mask2)
        mask2_path = os.path.join(self.plot_dir, "ctk_mask2.png")
        plt.imsave(mask2_path, img_bound2)
        
        m1_img = Image.open(mask1_path)
        self.mask1_display.configure(image=ctk.CTkImage(m1_img, size=(280, 280)))
        
        m2_img = Image.open(mask2_path)
        self.mask2_display.configure(image=ctk.CTkImage(m2_img, size=(280, 280)))
        
        self.status_label.configure(text="Analysis Complete.", text_color=("green", "#00FF00"))

    def change_theme(self, new_theme: str):
        """Dynamically triggers the rendering engine to swap color palettes."""
        ctk.set_appearance_mode(new_theme)

if __name__ == "__main__":
    app = AgriVisionApp()
    app.mainloop()