import requests
from requests.auth import HTTPBasicAuth
from tqdm import tqdm
import os

# --- CONFIG ---
username = "notshubham"
password = ",92VM4EYcA8-k^G"
download_url = "https://data.ornldaac.earthdata.nasa.gov/protected/bundle/Permafrost_Thaw_Depth_YK_1598.zip"
output_folder = "downloads"
os.makedirs(output_folder, exist_ok=True)
output_file = os.path.join(output_folder, download_url.split("/")[-1])

# --- DOWNLOAD WITH PROGRESS BAR ---
with requests.get(download_url, auth=HTTPBasicAuth(username, password), stream=True) as r:
    r.raise_for_status()
    total_size = int(r.headers.get('content-length', 0))
    block_size = 1024  # 1 KB
    with open(output_file, 'wb') as f, tqdm(
        total=total_size, unit='iB', unit_scale=True, desc=output_file
    ) as bar:
        for data in r.iter_content(block_size):
            f.write(data)
            bar.update(len(data))

print(f"Download completed: {output_file}")
