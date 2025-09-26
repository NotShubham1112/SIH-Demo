import numpy as np
import cv2
import rasterio
from scipy import ndimage as ndi

# --- Feature extraction ---
def compute_slope(dem):
    gy, gx = np.gradient(dem)
    slope = np.sqrt(gx**2 + gy**2)
    return slope

def compute_curvature(dem):
    return ndi.laplace(dem)

def edges_and_variance(img, kernel=15):
    gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    mean = ndi.uniform_filter(gray.astype(np.float32), size=kernel)
    mean_sq = ndi.uniform_filter(gray.astype(np.float32)**2, size=kernel)
    var = mean_sq - mean**2
    var = (var - var.min()) / (np.ptp(var) + 1e-8)
    return edges, var

def make_feature_matrix(dem, img):
    slope = compute_slope(dem)
    curvature = compute_curvature(dem)
    edges, var = edges_and_variance(img)
    features = np.stack([slope.flatten(), curvature.flatten(), edges.flatten(), var.flatten()], axis=1)
    return features, slope.shape, slope, curvature, edges, var

# --- Synthetic label generation ---
def generate_labels(slope, curvature, edges, var):
    # Normalize features
    slope_n = (slope - slope.min()) / (np.ptp(slope) + 1e-8)
    curvature_n = (curvature - curvature.min()) / (np.ptp(curvature) + 1e-8)
    edges_n = edges
    var_n = var

    # Synthetic risk score: weighted sum
    risk_score = 0.4 * slope_n + 0.3 * curvature_n + 0.2 * edges_n + 0.1 * var_n
    # Threshold to generate label: top 20% high risk
    threshold = np.percentile(risk_score, 80)
    labels = (risk_score >= threshold).astype(int)
    return labels.flatten()

# --- Main ---
if __name__ == "__main__":
    # Example input files
    dem_path = "uploads/sample_dem.tif"
    img_path = "uploads/sample_img.png"

    # Read files
    with rasterio.open(dem_path) as src:
        dem = src.read(1).astype(np.float32)
    img_bgr = cv2.imread(img_path)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    # Feature extraction
    features, shape, slope, curvature, edges, var = make_feature_matrix(dem, img)

    # Generate synthetic labels
    labels = generate_labels(slope, curvature, edges, var)

    # Save dataset
    np.save("features.npy", features)
    np.save("labels.npy", labels)

    print("Dataset generated: features.npy and labels.npy")
