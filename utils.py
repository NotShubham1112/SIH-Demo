import numpy as np
import cv2
from scipy import ndimage as ndi
import rasterio

# --- Image-based features ---
def compute_edges(img_gray):
    return cv2.Canny(img_gray.astype(np.uint8), 50, 150).astype(np.float32)/255.0

def compute_variance(img_gray, kernel=15):
    mean = ndi.uniform_filter(img_gray.astype(np.float32), size=kernel)
    mean_sq = ndi.uniform_filter(img_gray.astype(np.float32)**2, size=kernel)
    var = mean_sq - mean**2
    return (var - var.min()) / (np.ptp(var)+1e-8)

def compute_slope_curvature(dem):
    gy, gx = np.gradient(dem)
    slope = np.sqrt(gx**2 + gy**2)
    curvature = ndi.laplace(dem)
    return slope, curvature

# --- Feature aggregation ---
def make_feature_matrix(img_gray, dem, rock_type_layer=None, permafrost_layer=None, vegetation_layer=None, climate_layer=None):
    edges = compute_edges(img_gray)
    var = compute_variance(img_gray)
    slope, curvature = compute_slope_curvature(dem)

    features = [edges.flatten(), var.flatten(), slope.flatten(), curvature.flatten()]

    if rock_type_layer is not None:
        features.append(rock_type_layer.flatten())
    if permafrost_layer is not None:
        features.append(permafrost_layer.flatten())
    if vegetation_layer is not None:
        features.append(vegetation_layer.flatten())
    if climate_layer is not None:
        features.append(climate_layer.flatten())

    return np.stack(features, axis=1), slope.shape

# --- Data fetching placeholders ---
def fetch_dem(lat, lon):
    dem_path = "data/DEM_sample.tif"  # Replace with real DEM download logic
    with rasterio.open(dem_path) as src:
        dem = src.read(1)
    return dem.astype(np.float32)

def fetch_geology(lat, lon):
    path = "data/rock_type_sample.tif"
    with rasterio.open(path) as src:
        rock_layer = src.read(1)
    return rock_layer.astype(np.float32)

def fetch_vegetation(lat, lon):
    path = "data/ndvi_sample.tif"
    with rasterio.open(path) as src:
        veg_layer = src.read(1)
    return veg_layer.astype(np.float32)

def fetch_permafrost(lat, lon):
    path = "data/permafrost_sample.tif"
    with rasterio.open(path) as src:
        perma_layer = src.read(1)
    return perma_layer.astype(np.float32)

def fetch_climate(lat, lon):
    path = "data/climate_sample.tif"
    with rasterio.open(path) as src:
        climate_layer = src.read(1)
    return climate_layer.astype(np.float32)
