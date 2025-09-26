import os
import numpy as np
import cv2
import rasterio
from scipy import ndimage as ndi
from sklearn.ensemble import RandomForestClassifier
import joblib

dataset_folder = "uploads/Permafrost_Thaw_Depth_YK_1598"
model_folder = "models"
os.makedirs(model_folder, exist_ok=True)

# --- Feature extraction ---
def compute_slope(dem):
    gy, gx = np.gradient(dem)
    return np.sqrt(gx**2 + gy**2)

def compute_curvature(dem):
    return ndi.laplace(dem)

def edges_and_variance(dem, kernel=15):
    dem_norm = ((dem - dem.min()) / (np.ptp(dem)+1e-8) * 255).astype(np.uint8)
    edges = cv2.Canny(dem_norm, 50, 150).astype(np.float32)/255.0
    mean = ndi.uniform_filter(dem.astype(np.float32), size=kernel)
    mean_sq = ndi.uniform_filter(dem.astype(np.float32)**2, size=kernel)
    var = mean_sq - mean**2
    var = (var - var.min()) / (np.ptp(var)+1e-8)
    return edges, var

def make_features_labels(dem_file):
    with rasterio.open(dem_file) as src:
        dem = src.read(1).astype(np.float32)
    slope = compute_slope(dem)
    curvature = compute_curvature(dem)
    edges, var = edges_and_variance(dem)
    
    features = np.stack([slope.flatten(), curvature.flatten(), edges.flatten(), var.flatten()], axis=1)
    
    # Generate labels: top 25% DEM values = high risk
    threshold = np.percentile(dem, 75)
    labels = (dem.flatten() >= threshold).astype(int)
    
    return features, labels

# --- Main ---
dem_file = os.path.join(dataset_folder, "Permafrost_Extent_Elevation_Threshold.tif")
features, labels = make_features_labels(dem_file)

# Train model
model = RandomForestClassifier(n_estimators=200)
model.fit(features, labels)

# Save model
os.makedirs("models", exist_ok=True)
model_path = os.path.join("models", "rockfall_model.pkl")
joblib.dump(model, model_path)
print(f"Model trained and saved: {model_path}")
