from sklearn.mixture import GaussianMixture
import pandas as pd
import numpy as np
import ot
from joblib import dump, load

yields_true = pd.read_csv('../CSVs/yields_subset.csv').iloc[:, 2:]
gm = GaussianMixture(n_components=4, n_init=10)
gm.fit(yields_true)
dump(gm, '../parameters/gmm.joblib')

noise = np.load('../data/noise.npy')[:, :4]

loaded_model = load('../parameters/gmm.joblib')

weights = loaded_model.weights_
means = loaded_model.means_
covariances = loaded_model.covariances_

simul = np.zeros((4,10_000))
for j in range(10000):
    component_idx = np.random.choice(np.arange(len(weights)), p=weights)
    S = np.linalg.cholesky(covariances[component_idx])
    simul[:, j] = S @ noise[j, :] + means[component_idx]

simul = (simul * (simul > 0)).T

gen_rnd = simul[np.random.randint(0, 9999, 1000)]

print(ot.sliced.sliced_wasserstein_distance(yields_true.to_numpy(), gen_rnd, seed=0))