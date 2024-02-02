import ot
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

gen_yields = pd.read_csv(f'CSVs/vae_yields_subset.csv')
subset_yields = pd.read_csv('CSVs/yields_subset.csv').iloc[:, 2:]

yield_1_gen = gen_yields.to_numpy()
yield_1_true = subset_yields.to_numpy()

print(ot.sliced.sliced_wasserstein_distance(yield_1_true, yield_1_gen, seed=0))