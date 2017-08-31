import numpy
import pyximport; pyximport.install(setup_args={"include_dirs":numpy.get_include()})
import numpy as np
import NeighborMapSampler as NMS
from pyhmc import hmc, autocorr1
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from corner import corner

def gen_indep_maps(nx,ny,nz,sigma_y=1,expected_N=100):
    """Generate y-map with diagonal covariance matrix."""

    # Calculate mu such that the mean of delta is 0.
    mu = -0.5 * (sigma_y ** 2)

    # Create a y map.
    y_map = mu + np.random.normal(0,sigma_y,(nx,ny,nz))

    print('INDEP MEAN: ',np.random.normal(0,sigma_y,(nx,ny,nz)).mean())

    # Create an N map.
    N_map = np.random.poisson(expected_N * np.exp(y_map),(nx,ny,nz))

    return y_map, N_map

def gen_dep_maps(nx,ny,nz,cov,expected_N=100):
    """Generate a y-map with covariances between neighbors."""

    # Make the covariance matrix.
    cov_mat = np.zeros((nx*ny*nz,nx*ny*nz))

    # Loop over all voxels.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                ind_1 = i*ny*nz + j*nz + k

                # Loop over all neighbors.
                for a in range(-1,2):
                    for b in range(-1,2):
                        for c in range(-1,2):
                            num_off = 0
                            if a != 0:
                                num_off += 1
                            if b != 0:
                                num_off += 1
                            if c != 0:
                                num_off += 1

                            # Check that the neighbor index is within the map.
                            if 0 <= (i+a) < nx and 0 <= (j+b) < ny and 0 <= (k+c) <nz:
                                ind_2 = (i+a)*ny*nz + (j+b)*nz + (k+c)

                                # Store the covariance.
                                cov_mat[ind_1,ind_2] = cov[num_off]

    # Calculate mu such that the mean of delta is 0.
    mu = -0.5 * (sigma_y ** 2)

    # Perform Cholesky decomposition.
    L = np.linalg.cholesky(cov_mat)

    # Create a y map.
    y_map = mu + np.reshape(np.matmul(L,np.random.normal(size=nx*ny*nz)),(nx,ny,nz))

    # Create an N map.
    N_map = np.random.poisson(expected_N * np.exp(y_map),(nx,ny,nz))

    return y_map, N_map

def logprob(y,N,f,mu,cov,expected_N,nx,ny,nz):
    """Function that preps the map and calls the likelihood function."""
    y = np.reshape(y,(nx,ny,nz))
    logp,grad = NMS.compute_log_prob(y,N,f,mu,cov,expected_N)
    return logp, np.ravel(grad)

# Define the map properties.
nx = 10
ny = 10
nz = 10
sigma_y = 1
expected_N = 2
mu = -0.5 * (sigma_y ** 2)
cov = np.array([sigma_y ** 2,0.05,0.025,0.0125])
f = np.ones((nx,ny,nz))

# Create a map.
y_true,N_true = gen_dep_maps(nx,ny,nz,cov,expected_N=expected_N)

print('Provided Cov: ', str(cov))
print('Realization Cov: ', str(NMS.compute_neighbor_covariances(y_true,f)))

# Create an initial (wrong) random y-map.
y0 = np.random.normal(size=(nx,ny,nz))

# Perform sampling.
tau = 130
n_samps = 1000 * tau
samples,logp,diag = hmc(logprob, x0=np.ravel(y0), args=(N_true,f,mu,cov,expected_N,nx,ny,nz,),
                        n_samples=int(n_samps),return_diagnostics=True,epsilon=1/8.,
                        return_logp=True,n_burn=int(0.25*n_samps))

# Print out the sample likelihoods to verify convergence.
plt.plot(range(len(logp)),logp)
plt.savefig('logp.png')
plt.clf()

print('Acceptance Rate: ', str(1-diag['rej']))

# Compute the MLE values for y from the chain.
yf = np.zeros((nx,ny,nz))
for i in range(nx):
    for j in range(ny):
        for k in range(nz):
            ind = i*ny*nz + j*nz + k

            # Perform histogramming.
            y_samps = samples[:,ind]
            hist,edges = np.histogram(y_samps,bins=50)
            mids = (edges[1]-edges[0])/2 + edges[:-1]

            # Calculate MLE
            yf[i,j,k] = mids[np.argmax(hist)]

print('HMC Recoverd Cov: ', str(NMS.compute_neighbor_covariances(yf,f)))

# Plot the MLE and initial y-values.
plt.scatter(y_true,y0,label='Initial y')
plt.scatter(y_true,yf,label='MLE y')
plt.xlabel('True $\ln(1+\delta)$')
plt.ylabel('Sampled $\ln(1+\delta)$')
plt.legend()
plt.tight_layout()
plt.savefig('scatter_N2.png')

# Plot a corner plot of some subset of the y values.
figure = corner(samples[:,:5],truths=y_true.ravel()[:5])
figure.savefig('corner.png')