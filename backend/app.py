from flask import Flask, render_template, request, redirect, url_for
import os
from model import predict_fire, get_rgb_from_5band_array
import numpy as np
from PIL import Image
import io
import base64
import matplotlib.pyplot as plt

# Compute absolute paths
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # project root
TEMPLATES_DIR = os.path.join(BASE_DIR, "templates")
STATIC_DIR = os.path.join(BASE_DIR, "static")

# Initialize Flask app with explicit template_folder and static_folder
app = Flask(__name__, template_folder=TEMPLATES_DIR, static_folder=STATIC_DIR)

UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

def array_to_base64(array):
    # Ensure input array is float 0-1 before conversion if needed, then scale to 0-255
    if array.dtype != np.uint8 and array.max() <= 1.0 + 1e-6:
        img_array = (array * 255).astype(np.uint8)
    else:
        img_array = array.astype(np.uint8)
        
    # Handle Grayscale (1-channel) or RGB (3-channel)
    if img_array.ndim == 3 and img_array.shape[-1] == 1:
        img_array = img_array.squeeze(-1)

    img = Image.fromarray(img_array)
    buffer = io.BytesIO()
    img.save(buffer, format="PNG")
    encoded = base64.b64encode(buffer.getvalue()).decode()
    return "data:image/png;base64," + encoded

# Function to create MBR image visualization (using a common index/band with a colormap)
def create_mbr_image(arr_hw):
    # Example MBR visualization: using the Near-Infrared band (index 3)
    if arr_hw.ndim == 3 and arr_hw.shape[2] >= 4:
        band = arr_hw[:, :, 3] 
    else:
        # Fallback 
        band = arr_hw.mean(axis=-1) if arr_hw.ndim == 3 else np.zeros((256, 256))

    # Normalize band data 
    band_norm = (band - band.min()) / (band.max() - band.min() + 1e-6)
    
    # Apply a color map (using 'magma' or similar for MBR visualization)
    cmap = plt.get_cmap('magma') 
    mbr_rgb = cmap(band_norm)[:, :, :3] # Take only RGB channels
    
    return array_to_base64(mbr_rgb)


@app.route("/", methods=["GET", "POST"])
def index():
    if request.method == "POST":
        if "file" not in request.files:
            return redirect(request.url)
        file = request.files["file"]
        if file.filename == "":
            return redirect(request.url)
        
        filepath = os.path.join(UPLOAD_FOLDER, file.filename)
        file.save(filepath)

        # Run Prediction
        arr_hw, probs_up = predict_fire(filepath)
        rgb = get_rgb_from_5band_array(arr_hw)
        
        
        # ----------------------------------------------------
        # 1. Prediction Overlay Generation (FINALIZED BLENDING)
        # ----------------------------------------------------
        
        # Use 'hot' colormap (dark red -> bright yellow/white)
        cmap = plt.get_cmap('hot') 
        
        # Use original probability for Colormap
        fire_rgba = cmap(probs_up)

        # Define Alpha (Opacity) using a squared probability (P^2). 
        # This highlights high-confidence areas (the yellow parts).
        alpha = (probs_up ** 2)[..., None] 

        # Proper Alpha Blending: Result = Base * (1 - Alpha) + Foreground * Alpha
        blended_image = (rgb * (1.0 - alpha)) + (fire_rgba[..., :3] * alpha)
        
        # Ensure final image is clipped to valid 0-1 range
        overlay_image = np.clip(blended_image, 0, 1)
        
        # ----------------------------------------------------
        
        # 2. Create MBR image 
        mbr_image = create_mbr_image(arr_hw)
        
        # Convert images to base64
        rgb_base64 = array_to_base64(rgb)
        overlay_base64 = array_to_base64(overlay_image)

        # Calculate the fraction of pixels predicted as fire
        fire_fraction = (probs_up > 0.5).sum() / probs_up.size

        return render_template("index.html",
                               rgb_image=rgb_base64,
                               mbr_image=mbr_image,
                               overlay_image=overlay_base64,
                               fire_fraction=f"{fire_fraction*100:.2f}%")
    return render_template("index.html")

if __name__ == "__main__":
    app.run(debug=True)