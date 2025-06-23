from scipy.io import loadmat
import numpy as np
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow.keras import layers, models

# === Helper to auto-detect 3D hyperspectral data variable ===
def get_hsi_variable(mat_dict):
    for key in mat_dict:
        if not key.startswith("__"):
            arr = mat_dict[key]
            if isinstance(arr, np.ndarray) and arr.ndim == 3:
                print(f"Auto-detected HSI variable: '{key}'")
                return arr
    raise ValueError("No 3D hyperspectral data found in the .mat file.")

def get_gt_variable(mat_dict):
    for key in mat_dict:
        if not key.startswith("__"):
            arr = mat_dict[key]
            if isinstance(arr, np.ndarray) and arr.ndim == 2:
                print(f"Auto-detected GT variable: '{key}'")
                return arr
    raise ValueError("No 2D ground truth data found in the .mat file.")

# === Load and auto-detect variables ===
hsi_data = loadmat('Indian_pines_corrected.mat')
hsi = get_hsi_variable(hsi_data)
 
print("HSI shape:", hsi.shape)

# === Normalize ===
def min_max_normalize(hsi):
    hsi_min = hsi.min(axis=(0, 1), keepdims=True)
    hsi_max = hsi.max(axis=(0, 1), keepdims=True)
    return (hsi - hsi_min) / (hsi_max - hsi_min + 1e-6)

hsi_norm = min_max_normalize(hsi)
hsi_crop = hsi_norm[30:130, 30:130, :]  # Crop to 100x100

# PCA RGB
hsi_reshaped = hsi_crop.reshape(-1, hsi_crop.shape[2])
pca = PCA(n_components=3)
rgb = pca.fit_transform(hsi_reshaped)
rgb_img = rgb.reshape(hsi_crop.shape[0], hsi_crop.shape[1], 3)
rgb_img = (rgb_img - rgb_img.min()) / (rgb_img.max() - rgb_img.min())

 
# === Denoising with PCA ===
h, w, bands = hsi_norm.shape
hsi_reshaped = hsi_norm.reshape(-1, bands)
pca = PCA(n_components=30)
hsi_denoised = pca.fit_transform(hsi_reshaped)

print("Original bands:", bands)
print("After PCA:", hsi_denoised.shape[1])
hsi_denoised_img = hsi_denoised.reshape(h, w, -1)

# Show denoised RGB
rgb = hsi_denoised_img[:, :, :3]
rgb = (rgb - rgb.min()) / (rgb.max() - rgb.min())

 
# === Autoencoder ===
def build_autoencoder(input_shape):
    model = models.Sequential([
        layers.InputLayer(input_shape=input_shape),
        layers.Dense(128, activation='relu'),
        layers.Dense(64, activation='relu'),
        layers.Dense(128, activation='relu'),
        layers.Dense(input_shape[0], activation='sigmoid')
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    return model

input_shape = (30,)
autoencoder = build_autoencoder(input_shape)
autoencoder.summary()

X_train = hsi_denoised.reshape(-1, 30)
autoencoder.fit(X_train, X_train, epochs=50, batch_size=256, validation_split=0.2)

reconstructed = autoencoder.predict(X_train)
reconstruction_error = np.mean((X_train - reconstructed) ** 2, axis=1)

threshold = np.percentile(reconstruction_error, 95)
anomalies = reconstruction_error > threshold
anomaly_map = anomalies.reshape(h, w)

plt.imshow(anomaly_map, cmap='hot')
plt.title("Anomaly Detection Map")
plt.colorbar()
plt.show()

 