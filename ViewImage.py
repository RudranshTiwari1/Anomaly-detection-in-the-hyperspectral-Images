import scipy.io as sio
import numpy as np
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA

# 👇 Just change the path to your .mat file
mat_path = "Indian_pines_corrected.mat"

# Load .mat file
mat = sio.loadmat(mat_path)

# Auto-detect the variable: pick the first 3D array
hsi = None
for key in mat:
    if not key.startswith("__"):  # ignore meta keys
        if isinstance(mat[key], np.ndarray) and mat[key].ndim == 3:
            hsi = mat[key]
            print(f"Auto-detected variable: '{key}'")
            break

if hsi is None:
    raise ValueError("No 3D hyperspectral variable found in the .mat file.")

print("HSI shape:", hsi.shape)

# PCA to RGB
h, w, d = hsi.shape
reshaped = hsi.reshape(-1, d)
pca = PCA(n_components=3)
rgb = pca.fit_transform(reshaped)
rgb = rgb.reshape(h, w, 3)
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

# Show image
plt.figure(figsize=(6, 6))
plt.imshow(rgb)
plt.title("HSI - PCA RGB View (Auto-loaded)")
plt.axis("off")
plt.tight_layout()
plt.show()
