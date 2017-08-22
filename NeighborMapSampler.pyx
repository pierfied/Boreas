import numpy as np
cimport numpy as np
from time import time
from HMCMapSampler import HMCMapSampler
from libc.math cimport exp, log

class NeighborMapSampler(HMCMapSampler):
    """Class that performs HMC sampling on delta map using a redshift catalog."""

    def __init__(self,cat,box,f_map,y_map,mu,cov):
        self.cat = cat
        self.box = box
        self.f_map = f_map
        self.y_map = y_map
        self.mu = mu
        self.cov = cov

        super().__init__(y_map)

    def log_prob(self,y_map):
        pass

def compute_log_prob(y_map,N,f_map,mu,cov,expected_N):
    return compute_log_prob_c(y_map,N,f_map,mu,cov,expected_N)

cdef double compute_log_prob_c(np.ndarray y_map, np.ndarray N, np.ndarray f_map,
                             double mu, np.ndarray cov, double expected_N):

    # Get the shape of the array.
    cdef int nx = y_map.shape[0]
    cdef int ny = y_map.shape[1]
    cdef int nz = y_map.shape[2]

    cdef int i,j,k
    cdef int a,b,c
    cdef int num_off
    cdef double log_gaussian = 0
    cdef double log_poisson = 0
    cdef double gaussian_j, lambda_k
    cdef np.ndarray grad = np.zeros(shape=nx*ny*nz)
    # Loop over all voxels.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):
                gaussian_j = 0

                # Loop over all neighbors.
                for a in range(-1,2):
                    for b in range(-1,2):
                        for c in range(-1,2):

                            # Check that this neighbor is within the map.
                            if 0 <= i+a < nx and 0 <= j+b < ny and 0 <= k+c < nz:

                                # Count the number of dimensions with non-zero offset.
                                num_off = 0
                                if a != 0:
                                    num_off += 1
                                if b != 0:
                                    num_off += 1
                                if c != 0:
                                    num_off += 1

                                # Compute the Gaussian component for the neighbor.
                                gaussian_j += cov[num_off]*(y_map[i+a,j+b,k+c] - mu)

                # Compute the overall Gaussian contribution for this voxel.
                log_gaussian += (y_map[i,j,k] - mu) * gaussian_j

                # Compute the expected number count for this voxel.
                lambda_k = f_map[i,j,k] * expected_N * exp(y_map[i,j,k])

                # Compute the Poisson contribution for this voxel.
                log_poisson += N[i,j,k] * log(lambda_k) - lambda_k - log_factorial(N[i,j,k])

    log_gaussian *= -0.5

    print(log_gaussian)
    print(log_poisson)

    return log_gaussian + log_poisson

def lf(N):
    return log_factorial(N)

cdef double log_factorial(int N):
    cdef int i
    cdef double sum = 0
    for i in range(1,N+1):
        sum += log(i)

    return sum

def compute_neighbor_covariances(y_map,f_map):
    return compute_neighbor_covariances_c(y_map,f_map)

cdef np.ndarray compute_neighbor_covariances_c(np.ndarray y_map, np.ndarray f_map):
    """Computes the covariances of y-values for face, edge, and corner neighbors."""

    # Get the shape of the array.
    cdef int nx = y_map.shape[0]
    cdef int ny = y_map.shape[1]
    cdef int nz = y_map.shape[2]

    cdef int i,j,k
    cdef int a,b,c
    cdef double sum_self = 0
    cdef double sum_face = 0
    cdef double sum_edge = 0
    cdef double sum_corner = 0
    cdef double count_self = 0
    cdef double count_face = 0
    cdef double count_edge = 0
    cdef double count_corner = 0
    cdef int num_off
    # Loop over all pixels.
    for i in range(nx):
        for j in range(ny):
            for k in range(nz):

                # Verify that the voxel has reasonable occupancy.
                if f_map[i,j,k] > 0.5:

                    # Loop over all neighbors.
                    for a in range(-1,2):
                        for b in range(-1,2):
                            for c in range(-1,2):
                                # Count the number of dimensions with non-zero offset.
                                num_off = 0
                                if a != 0:
                                    num_off += 1
                                if b != 0:
                                    num_off += 1
                                if c != 0:
                                    num_off += 1

                                # Check if this neighbor is within the map,
                                # and has reasonable occupancy.
                                if 0 <= i+a < nx and 0 <= j+b < ny and 0 <= k+c < nz \
                                        and f_map[i+a,j+b,k+c] > 0.5:

                                    # Check if is self, face, edge, or corner neighbor
                                    # and add to the appropriate sum.
                                    if num_off == 0:
                                        sum_self += y_map[i,j,k] * y_map[i+a,j+b,k+c]
                                        count_self += 1
                                    elif num_off == 1:
                                        sum_face += y_map[i,j,k] * y_map[i+a,j+b,k+c]
                                        count_face += 1
                                    elif num_off == 2:
                                        sum_edge += y_map[i,j,k] * y_map[i+a,j+b,k+c]
                                        count_edge += 1
                                    elif num_off == 3:
                                        sum_corner += y_map[i,j,k] * y_map[i+a,j+b,k+c]
                                        count_corner += 1

    # Compute the covariances.
    cdef np.ndarray cov = np.zeros(shape=(4))
    cov[0] = sum_self/count_self
    cov[1] = sum_face/count_face
    cov[2] = sum_edge/count_edge
    cov[3] = sum_corner/count_corner
    cdef double mean = y_map[f_map > 0.5].mean()
    cov -= mean ** 2

    return cov