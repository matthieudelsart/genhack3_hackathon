import pandas as pd
import matplotlib.pyplot as plt

gen_yields = pd.read_csv('CSVs/gen_yields_subset.csv')
subset_yields = pd.read_csv('CSVs/yields_subset.csv').iloc[:, 2:]

gen_yields.hist(figsize=(10, 8), bins=20)
plt.suptitle('Histograms of Generated Yields', fontsize=16)
plt.savefig('figures/gen_subset_histogram.png', dpi=100)
plt.show()

subset_yields.hist(figsize=(10, 8), bins=20)
plt.suptitle('Histograms of Original Yields', fontsize=16)
plt.savefig('figures/subset_histogram.png', dpi=100)
plt.show()