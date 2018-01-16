import numpy
import pyximport;

pyximport.install(setup_args={"include_dirs": numpy.get_include()})
import numpy as np
import MapSampler as MS
from BoundingBox import BoundingBox
import matplotlib.pyplot as plt
import pickle
from chainconsumer import ChainConsumer


def test_diag(box, mu, var, expected_N, num_samps, num_steps, num_burn, epsilon, mask_frac=None):
    # Generate y_values.
    y_true = np.random.normal(mu, np.sqrt(var), [box.nx, box.ny, box.nz])

    # Create the mask.
    if mask_frac is None:
        mask = np.ones(y_true.shape)
    else:
        multiplier = 10
        mask = np.random.beta(multiplier * mask_frac, multiplier * (1 - mask_frac), y_true.shape)

    # Calculate the expected number counts for each pixel and generate Poisson draws.
    lam = expected_N * mask * np.exp(y_true)
    N_obs = np.random.poisson(lam).astype(np.double)

    # Attempt to reconstruct y from N_obs.
    N_obs_mask_adj = N_obs / mask
    mean_N = N_obs_mask_adj.mean()
    delta_obs = N_obs_mask_adj / mean_N - 1
    y_obs = np.log(1 + np.clip(delta_obs, -1 + 1e-3, np.inf))

    # Sample the map.
    cov = np.array([var, 0, 0, 0])
    ms = MS.MapSampler(None, box, N_obs, mask, y_obs, mu, cov, expected_N)
    chain, logp = ms.sample(num_samps, num_steps, num_burn, epsilon)

    return chain, logp, y_true, y_obs


# Test params.
nvox = 10
var = 1.
mu = -var / 2.
expected_N = 2.

# Sampling params.
num_samps = 1000
num_burn = 100
num_steps = 10
epsilon = 1 / 8.

box = BoundingBox(0, 0, 0, nvox, nvox, nvox, 1)
# results = test_diag(box, mu, var, expected_N, num_samps, num_steps, num_burn, epsilon)

# pickle.dump(results, open('results.p', 'wb'))

results = pickle.load(open('results.p', 'rb'))

chain = results[0]
logp = results[1]
y_true = results[2]
y_obs = results[3]

plt.hist(chain[:,9],25)
plt.savefig('hist.png')

plt.plot(range(len(logp)), logp)
plt.savefig('logp.png')
plt.clf()

c = ChainConsumer()
c.add_chain(chain)
c.configure(sigma2d=False)

y_mle = np.array([y[1] for y in c.analysis.get_summary().values()])

plt.scatter(y_true, y_obs)
plt.scatter(y_true, y_mle)
plt.savefig('comp.png')
