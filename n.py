import numpy as np
data = np.load("data/mimic3/standard/real_data/train_dual.npz")
print("Diag:", data["x_diag"].shape)
print("Proc:", data["x_proc"].shape)
print("Lens:", data["lens"].shape)
