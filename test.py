import numpy as np
import ot
import pandas as pd

output = np.load("output.npy")
print(len(output))
yields_df = pd.read_csv('CSVs/yields_subset.csv').iloc[:, 2:]
print(ot.sliced.sliced_wasserstein_distance(output[:, 0], yields_df.to_numpy()[:, 0], seed=0))