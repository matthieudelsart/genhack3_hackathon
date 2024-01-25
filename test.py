import numpy as np

file_path = 'data/noise.npy'
noise_full = np.load(file_path)

indx_range = np.arange(0, len(noise_full))
indx_selected = np.random.choice(indx_range, size=25, replace=False)
noise = noise_full[indx_selected]

print(len(noise[0]))
print(noise)