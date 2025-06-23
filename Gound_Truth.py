import scipy.io as sio
import matplotlib.pyplot as plt
import os

# Load Ground Truth from .mat file
mat_file = 'Indian_pines_gt.mat'  # Replace with your own file if needed

if not os.path.exists(mat_file):
    raise FileNotFoundError(f"{mat_file} not found in the directory!")

# Load the data
data = sio.loadmat(mat_file)

# The key will depend on the dataset
# For Indian Pines ground truth, the key is usually 'indian_pines_gt'
key = [k for k in data.keys() if not k.startswith('__')][0]
ground_truth = data[key]

# Plot the ground truth
plt.figure(figsize=(6, 6))
plt.imshow(ground_truth, cmap='nipy_spectral')
plt.title("Ground Truth Visualization")
plt.colorbar()
plt.axis('off')
plt.show()
