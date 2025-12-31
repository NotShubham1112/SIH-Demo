# 🪨 SIH-Demo – AI Rockfall Risk Visualization

A demo web app that simulates **rockfall risk detection** in open-pit mines using **image analysis + thermal heatmaps**.

---

## 🚀 Features
- Upload any mine image (JPG/PNG).
- Generates:
  - `heatmap.png` → thermal risk zones
  - `overlay.png` → heatmap over original mine
- Simple **Flask web app** with HTML interface.

---

## ⚡ Installation & Usage

1. Clone repo:
   ```bash
   git clone https://github.com/yourusername/SIH-Demo.git
   cd SIH-Demo
---

# 📂 Repo Structure

```
SIH-Demo/
│── app.py
│── requirements.txt
│── README.md
│── templates/
│    └── index.html
│── static/
│    └── style.css
│── uploads/        # (auto-created when you run)
│── outputs/        # (auto-created when you run)
```

---

# 📄 app.py

```python
import os, cv2
import numpy as np
from pathlib import Path
from flask import Flask, render_template, request, send_from_directory
from scipy import ndimage as ndi

UPLOAD_FOLDER = "uploads"
OUTPUT_FOLDER = "outputs"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(OUTPUT_FOLDER, exist_ok=True)

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER

# --- Functions ---
def to_gray(img): 
    return cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)

def compute_edge_map(gray):
    gx = cv2.Sobel(gray, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(gray, cv2.CV_32F, 0, 1, ksize=3)
    grad_mag = np.sqrt(gx**2 + gy**2)
    grad_mag = (grad_mag - grad_mag.min()) / (np.ptp(grad_mag) + 1e-8)
    edges = cv2.Canny(gray, 50, 150).astype(np.float32) / 255.0
    return np.clip(0.6 * grad_mag + 0.4 * edges, 0, 1)

def local_variance_map(gray, kernel=15):
    mean = ndi.uniform_filter(gray.astype(np.float32), size=kernel)
    mean_sq = ndi.uniform_filter((gray.astype(np.float32)**2), size=kernel)
    var = mean_sq - mean**2
    return (var - var.min()) / (np.ptp(var) + 1e-8)

def heatmap_color(risk_map):
    im8 = (255 * risk_map).astype(np.uint8)
    heat = cv2.applyColorMap(im8, cv2.COLORMAP_JET)
    return cv2.cvtColor(heat, cv2.COLOR_BGR2RGB)

def overlay(base, heat, alpha=0.6):
    heat = cv2.resize(heat, (base.shape[1], base.shape[0]))
    return (base * (1 - alpha) + heat * alpha).astype(np.uint8)

# --- Routes ---
@app.route("/")
def index():
    return render_template("index.html")

@app.route("/upload", methods=["POST"])
def upload():
    file = request.files["file"]
    if not file: 
        return "No file uploaded", 400
    filepath = os.path.join(UPLOAD_FOLDER, file.filename)
    file.save(filepath)

    img_bgr = cv2.imread(filepath)
    img = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    gray = to_gray(img)

    edges = compute_edge_map(gray)
    var_map = local_variance_map(gray, 25)
    risk_map = (0.7 * edges + 0.3 * var_map)
    risk_map = (risk_map - risk_map.min()) / (np.ptp(risk_map) + 1e-8)

    heat = heatmap_color(risk_map)
    overlay_img = overlay(img, heat)

    out_heat = os.path.join(OUTPUT_FOLDER, "heatmap.png")
    out_overlay = os.path.join(OUTPUT_FOLDER, "overlay.png")
    cv2.imwrite(out_heat, cv2.cvtColor(heat, cv2.COLOR_RGB2BGR))
    cv2.imwrite(out_overlay, cv2.cvtColor(overlay_img, cv2.COLOR_RGB2BGR))

    return f"""
    <h2>Results</h2>
    <p>Heatmap:</p>
    <img src='/outputs/heatmap.png' width='500'><br>
    <p>Overlay:</p>
    <img src='/outputs/overlay.png' width='500'><br>
    <a href='/'>Upload another image</a>
    """

@app.route("/outputs/<path:filename>")
def outputs(filename):
    return send_from_directory(OUTPUT_FOLDER, filename)

if __name__ == "__main__":
    app.run(debug=True)
```

---

# 📄 templates/index.html

```html
<!DOCTYPE html>
<html>
<head>
    <title>Rockfall Risk Demo</title>
    <link rel="stylesheet" href="/static/style.css">
</head>
<body>
    <h1>🪨 Rockfall Risk Prediction Demo</h1>
    <form action="/upload" method="post" enctype="multipart/form-data">
        <p>Select a mine image to analyze:</p>
        <input type="file" name="file" accept="image/*">
        <input type="submit" value="Upload & Process">
    </form>
</body>
</html>
```

---

# 📄 static/style.css

```css
body {
    font-family: Arial, sans-serif;
    background: #f7f7f7;
    text-align: center;
    padding: 30px;
}
h1 {
    color: #2c3e50;
}
form {
    margin-top: 20px;
}
input[type="submit"] {
    margin-top: 15px;
    padding: 10px 20px;
    background: #27ae60;
    border: none;
    color: white;
    font-size: 16px;
    cursor: pointer;
    border-radius: 6px;
}
input[type="submit"]:hover {
    background: #219150;
}
```

---

# 📄 requirements.txt

```
flask
opencv-python
numpy
matplotlib
scipy
```

---
2. Setup environment:

   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # Linux/Mac
   ```

3. Install dependencies:

   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:

   ```bash
   python app.py
   ```

5. Open in browser:

   ```
   http://127.0.0.1:5000
   ```

---

## 📂 Outputs

* `outputs/heatmap.png`
* `outputs/overlay.png`

---

## 🔮 Future Extensions

* 3D holographic mine surface (Blender + Three.js).
* Real-time AI detection (YOLO/Deep Learning).
* Integration with drone images + IoT sensors.

---

👨‍💻 Developed for **SIH 2025 Prototype Demo**

