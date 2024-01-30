import pandas as pd
import matplotlib.pyplot as plt

model = 'vae' # 'gan' or 'vae'

gen_yields = pd.read_csv(f'CSVs/{model}_yields_subset.csv')
subset_yields = pd.read_csv('CSVs/yields_subset.csv').iloc[:, 2:]

for i in range(1, 5):
    plt.figure(figsize=(10, 8))

    column_to_plot_gen = gen_yields[f'YIELD_{i}']
    column_to_plot_subset = subset_yields[f'YIELD_{i}']

    plt.hist([column_to_plot_gen, column_to_plot_subset], 
             bins=50, alpha=0.8, label=[f'Histograms of Generated {model} Yields {i}',
                                         f'Histograms of Original Yields {i}'])

    plt.legend()
    plt.savefig(f'figures/YIELD{i}_histogram.png', dpi=100)
    plt.show()


# gen_yields.hist(figsize=(10, 8), bins=20)
# plt.suptitle(f'Histograms of Generated {model} Yields', fontsize=16)
# plt.savefig('figures/gen_subset_histogram.png', dpi=100)
# plt.show()

# subset_yields.hist(figsize=(10, 8), bins=20)
# plt.suptitle('Histograms of Original Yields', fontsize=16)
# plt.savefig('figures/subset_histogram.png', dpi=100)
# plt.show()