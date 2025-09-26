import cv2, json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from scipy import ndimage as ndi

# --- Paths ---
INPUT_PATH = "mine.jpg"   # replace with your mine image
OUTPUT_DIR = Path("outputs")
OUTPUT_DIR.mkdir(exist_ok=True)

# --- Functions ---
def to_gray(img): return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def compute_edge_map(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag = (grad_mag - grad_mag.min()) / (grad_mag.ptp() + 1e-8)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    return np.clip(0.6 * grad_mag + 0.4 * edges, 0, 1)

def local_variance_map(gray, kernel=15):
    mean = ndi.uniform_filter(gray.astype(np.float32), size=kernel)
    mean_sq = ndi.uniform_filter((gray.astype(np.float32)**2), size=kernel)
    var = mean_sq - mean**2
    return (var - var.min()) / (var.ptp() + 1e-8)

def heatmap_color(risk_map):
    im8 = (255 * risk_map).astype(np.uint8)
    heat = cv2.applyColorMap(im8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

def overlay(base, heat, alpha=0.6):
    heat = cv2.resize(heat, (base.shape[1], base.shape[0]))
    return (base * (1 - alpha) + heat * alpha).astype(np.uint8)

def extract_points(risk_map, threshold=0.65, min_distance=20):
    mask = (risk_map >= threshold).astype(np.uint8)
    num, labels = cv2.connectedComponents(mask)
    points = []
    for lbl in range(1, num):
        ys, xs = np.where(labels == lbl)
        if len(xs) == 0: continue
        cx, cy = int(xs.mean()), int(ys.mean())
        score = float(risk_map[cy, cx])
        points.append({"x": cx, "y": cy, "score": score})
    return points

# --- Pipeline ---
img_bgr = cv2.imread(INPUT_PATH)
img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
gray = to_gray(img)

edges = compute_edge_map(gray)
var_map = local_variance_map(gray, 25)
risk_map = (0.7 * edges + 0.3 * var_map)
risk_map = (risk_map - risk_map.min()) / (risk_map.ptp() + 1e-8)

heat = heatmap_color(risk_map)
overlay_img = overlay(img, heat)
points = extract_points(risk_map)

# --- Save outputs ---
cv2.imwrite(str(OUTPUT_DIR / "heatmap.png"), cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))
cv2.imwrite(str(OUTPUT_DIR / "overlay.png"), cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))
with open(OUTPUT_DIR / "danger_points.json", "w") as f:
    json.dump(points, f, indent=2)

print("✅ Done. Check 'outputs/' folder for results.")