import numpy
import pyximport;

pyximport.install(setup_args={"include_dirs": numpy.get_include()})
import numpy as np
import MapSampler as MS
from BoundingBox import BoundingBox
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
from chainconsumer import ChainConsumer


def test_diag(box, var, expected_N, num_samps, num_steps, num_burn, epsilon, mask_frac=None):
    # Generate y_values.
    mu = -0.5 * var
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
    y_obs = np.random.standard_normal(y_true.shape)

    # Sample the map.
    cov = np.array([var, 0, 0, 0])
    ms = MS.MapSampler(None, box, N_obs, mask, y_obs, mu, cov, expected_N)
    chain, logp = ms.sample(num_samps, num_steps, num_burn, epsilon)

    return chain, logp, y_true, y_obs


def test_cov(box, cov, expected_N, num_samps, num_steps, num_burn, epsilon, maks_frac=None):
    # Create the full covariance matrix.
    num_vox = box.nx * box.ny * box.nz
    full_cov = np.zeros((num_vox, num_vox))
    # Loop over all voxels.
    for i in range(box.nx):
        for j in range(box.ny):
            for k in range(box.nz):
                ind_vox = i * box.ny * box.nz + j * box.nz + k

                # Loop over all neighbors.
                for a in range(-1, 2):
                    for b in range(-1, 2):
                        for c in range(-1, 2):

                            if 0 <= i + a < box.nx and 0 <= j + b < box.ny and 0 <= k + c < box.nz:
                                # Count the number of dimensions with non-zero offset.
                                num_off = 0
                                if a != 0:
                                    num_off += 1
                                if b != 0:
                                    num_off += 1
                                if c != 0:
                                    num_off += 1

                                ind_neighbor = (i + a) * box.ny * box.nz + (j + b) * box.nz + (
                                        k + c)

                                # Set the value in the full covariance matrix.
                                full_cov[ind_vox, ind_neighbor] = cov[num_off]

    # Diagonalize the covariance matrix.
    L = np.linalg.cholesky(full_cov)

    # Create a sample y_map.
    mu = -0.5 * cov[0]
    y_true = mu + np.reshape(np.matmul(L, np.random.standard_normal(num_vox)),
                             (box.nx, box.ny, box.nz))

    # Create a Poisson realization and calculate observed d and y maps.
    N_obs = np.random.poisson(expected_N * np.exp(y_true)).astype(np.double)
    d_obs = N_obs / expected_N - 1
    y_obs = np.log(1 + d_obs)
    y_obs = np.clip(y_obs, -6, 6)

    # Sample the map.
    mask = np.ones(y_true.shape)
    # y_obs = np.random.standard_normal(y_true.shape)
    ms = MS.MapSampler(None, box, N_obs, mask, y_obs, mu, cov, expected_N)
    chain, logp = ms.sample(num_samps, num_steps, num_burn, epsilon)

    return chain, logp, y_true, y_obs


# Test params.
nvox = 10
var = 0.1
expected_N = 2.

# Sampling params.
num_samps = 1000
num_burn = 100
num_steps = 10
epsilon = 1 / 32.

box = BoundingBox(0, 0, 0, nvox, nvox, nvox, 1)
# results = test_diag(box, var, expected_N, num_samps, num_steps, num_burn, epsilon)

cov = var * np.array([1, 0.1, 0.05, 0.01])
# cov = var * np.array([1, 0, 0, 0])
results = test_cov(box, cov, expected_N, num_samps, num_steps, num_burn, epsilon)

pickle.dump(results, open('results_%d.p' % expected_N, 'wb'))

results = pickle.load(open('results_%d.p' % expected_N, 'rb'))

chain = results[0]
logp = results[1]
y_true = results[2]
y_obs = results[3]

plt.plot(range(len(logp)), logp)
plt.savefig('logp_%d.png' % expected_N)
plt.clf()

c = ChainConsumer()
c.add_chain(chain)

y_mle = np.array([y[1] for y in c.analysis.get_summary().values()])

plt.scatter(y_true, y_obs, label='Random')
plt.scatter(y_true, y_mle, label='HMC MLE')
plt.plot([-4, 4], [-4, 4], c='k', label='y=x')
plt.legend()
plt.xlabel('y True')
plt.ylabel('y Sample')
plt.title('Samples: %d, Pix/Side: %d, $\sigma_y^2=%f$, $\\bar{N}=%d$' % (
    num_samps, nvox, var, expected_N))
plt.savefig('comp_%d.png' % expected_N)

unconstrained_inds = [2, 6, 8, 9, 12]
c.plotter.plot(parameters=unconstrained_inds, truth=y_true.ravel()[unconstrained_inds])
plt.savefig('unconstrained_contour.png')
