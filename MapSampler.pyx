import numpy as np
cimport numpy as np
import ctypes

class SampleChain(ctypes.Structure):
    _fields_ = [('num_samples', ctypes.c_int),
                ('num_params', ctypes.c_int),
                ('accept_rate', ctypes.c_double),
                ('samples', ctypes.POINTER(ctypes.POINTER(ctypes.c_double))),
                ('log_likelihoods', ctypes.POINTER(ctypes.c_double))]

class LikelihoodArgs(ctypes.Structure):
    _fields_ = [('num_params', ctypes.c_int),
                ('f', ctypes.POINTER(ctypes.c_double)),
                ('y_inds', ctypes.POINTER(ctypes.c_int)),
                ('N', ctypes.POINTER(ctypes.c_double)),
                ('nx', ctypes.c_int),
                ('ny', ctypes.c_int),
                ('nz', ctypes.c_int),
                ('mu', ctypes.c_double),
                ('cov', ctypes.POINTER(ctypes.c_double)),
                ('expected_N', ctypes.c_double)]

class MapSampler:
    """Class that performs map sampling by calling the C HMC code."""

    def __init__(self, cat, box, N, f_map, y_map, mu, cov, expected_N):
        self.cat = cat
        self.box = box
        self.N = N
        self.f_map = f_map
        self.y_map = y_map
        self.y0 = self.y_map.copy()
        self.mu = mu
        self.cov = cov
        self.expected_N = expected_N

        # Compute the values for the inverse covariance matrix.
        self.inv_cov = cov / (cov[0] ** 2)
        self.inv_cov[1:] *= -1

    def sample(self, num_samps, num_steps, num_burn, epsilon):
        # Load the likelihood/sampling library.
        sampler_lib = ctypes.cdll.LoadLibrary('map_likelihood/libmaplikelihood.so')

        # Setup the sampler function.
        sampler = sampler_lib.sample_map
        sampler.argtypes = [ctypes.POINTER(ctypes.c_double),
                            ctypes.POINTER(ctypes.c_double),
                            LikelihoodArgs, ctypes.c_int, ctypes.c_int,
                            ctypes.c_int, ctypes.c_double]
        sampler.restype = SampleChain

        # Get the number of parameters and usable indices.
        y_good = self.f_map.ravel() > 0.5
        num_params = np.sum(y_good)
        y_inds = -np.ones(len(y_good),dtype=np.int32)
        y_inds[y_good] = np.arange(num_params)
        print('Num Params: ',num_params)

        # Calculate the mass and step-size scales.
        mass = np.ones(num_params,dtype=np.double)
        epsilon *= np.sqrt(self.cov[0])

        # Create the LikelihoodArgs.
        args = LikelihoodArgs()
        args.num_params = num_params
        args.f = self.f_map.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        args.y_inds = y_inds.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
        args.N = self.N.ravel().ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        # args.nx = self.box.nx
        # args.ny = self.box.ny
        # args.nz = self.box.nz
        args.nx = self.N.shape[0]
        args.ny = self.N.shape[1]
        args.nz = self.N.shape[2]
        args.mu = self.mu
        args.cov = self.cov.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        args.expected_N = self.expected_N.ravel()[0]

        print('About to run that MOFO!')

        # Call that MoFo
        y0 = self.y0[self.f_map > 0.5].ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        m = mass.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        results = sampler(y0, m, args, num_samps, num_steps, num_burn, epsilon)

        print('DONE!')

        print(results.accept_rate)

        chain = np.array([[results.samples[i][j] for j in range(num_params)]
                            for i in range(num_samps)])

        likelihoods = np.array([results.log_likelihoods[i]
                               for i in range(num_samps)])

        return chain, likelihoods

def compute_neighbor_covariances(y_map, f_map):
    """Wrapper function to call the cython code."""
    return compute_neighbor_covariances_c(y_map, f_map)

cdef np.ndarray compute_neighbor_covariances_c(np.ndarray y_map, np.ndarray f_map):
    """Computes the covariances of y-values for face, edge, and corner neighbors."""

    # Get the shape of the array.
    cdef int nx = y_map.shape[0]
    cdef int ny = y_map.shape[1]
    cdef int nz = y_map.shape[2]

    cdef int i, j, k
    cdef int a, b, c
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
                if f_map[i, j, k] > 0.5:

                    # Loop over all neighbors.
                    for a in range(-1, 2):
                        for b in range(-1, 2):
                            for c in range(-1, 2):
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
                                if 0 <= i + a < nx and 0 <= j + b < ny and 0 <= k + c < nz \
                                        and f_map[i + a, j + b, k + c] > 0.5:

                                    # Check if is self, face, edge, or corner neighbor
                                    # and add to the appropriate sum.
                                    if num_off == 0:
                                        sum_self += y_map[i, j, k] * y_map[i + a, j + b, k + c]
                                        count_self += 1
                                    elif num_off == 1:
                                        sum_face += y_map[i, j, k] * y_map[i + a, j + b, k + c]
                                        count_face += 1
                                    elif num_off == 2:
                                        sum_edge += y_map[i, j, k] * y_map[i + a, j + b, k + c]
                                        count_edge += 1
                                    elif num_off == 3:
                                        sum_corner += y_map[i, j, k] * y_map[i + a, j + b, k + c]
                                        count_corner += 1

    # Compute the covariances.
    cdef np.ndarray cov = np.zeros(shape=(4))
    cov[0] = sum_self / count_self
    cov[1] = sum_face / count_face
    cov[2] = sum_edge / count_edge
    cov[3] = sum_corner / count_corner
    cdef double mean = y_map[f_map > 0.5].mean()
    cov -= mean ** 2

    return cov

def compute_neighbor_covariances_N(N_map, y_map, f_map):
    """Wrapper function to call the cython code."""
    return compute_neighbor_covariances_N_c(N_map, y_map, f_map)

cdef np.ndarray compute_neighbor_covariances_N_c(np.ndarray N_map, np.ndarray y_map, np.ndarray f_map):
    """Computes the covariances of y-values for face, edge, and corner neighbors."""

    # Get the shape of the array.
    cdef int nx = N_map.shape[0]
    cdef int ny = N_map.shape[1]
    cdef int nz = N_map.shape[2]

    cdef int i, j, k
    cdef int a, b, c
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
                if f_map[i, j, k] > 0.5:

                    # Loop over all neighbors.
                    for a in range(-1, 2):
                        for b in range(-1, 2):
                            for c in range(-1, 2):
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
                                if 0 <= i + a < nx and 0 <= j + b < ny and 0 <= k + c < nz \
                                        and f_map[i + a, j + b, k + c] > 0.5:

                                    # Check if is self, face, edge, or corner neighbor
                                    # and add to the appropriate sum.
                                    if num_off == 0:
                                        sum_self += N_map[i, j, k] * N_map[i + a, j + b, k + c]
                                        count_self += 1
                                    elif num_off == 1:
                                        sum_face += N_map[i, j, k] * N_map[i + a, j + b, k + c]
                                        count_face += 1
                                    elif num_off == 2:
                                        sum_edge += N_map[i, j, k] * N_map[i + a, j + b, k + c]
                                        count_edge += 1
                                    elif num_off == 3:
                                        sum_corner += N_map[i, j, k] * N_map[i + a, j + b, k + c]
                                        count_corner += 1

    # Compute the covariances.
    cdef np.ndarray cov = np.zeros(shape=(4))
    cov[0] = sum_self / count_self
    cov[1] = sum_face / count_face
    cov[2] = sum_edge / count_edge
    cov[3] = sum_corner / count_corner
    cdef double mean = N_map[f_map > 0.5].mean()
    cov -= mean ** 2

    # Convert the covariance in the number counts into covariance in y.
    mu = y_map[f_map > 0.5].mean()
    print(cov)
    cov_y = cov / ((mean ** 2) * np.exp(2 * mu))

    return cov_y

def compute_neighbor_covariances_N_at_z(N_map, y_map, f_map, z_mids, z, dz = 0.01):
    """Wrapper function to call the cython code."""
    return compute_neighbor_covariances_N_at_z_c(N_map, y_map, f_map, z_mids, z, dz)

cdef np.ndarray compute_neighbor_covariances_N_at_z_c(np.ndarray N_map, np.ndarray y_map,
                                                      np.ndarray f_map, np.ndarray z_mids,
                                                      double z, double dz):
    """Computes the covariances of y-values for face, edge, and corner neighbors."""

    # Get the shape of the array.
    cdef int nx = N_map.shape[0]
    cdef int ny = N_map.shape[1]
    cdef int nz = N_map.shape[2]

    cdef np.ndarray slice_inds = np.abs(z_mids - z) < dz

    cdef int i, j, k
    cdef int a, b, c
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
                if f_map[i, j, k] > 0.5 and slice_inds[i, j, k]:

                    # Loop over all neighbors.
                    for a in range(-1, 2):
                        for b in range(-1, 2):
                            for c in range(-1, 2):
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
                                if 0 <= i + a < nx and 0 <= j + b < ny and 0 <= k + c < nz \
                                        and f_map[i + a, j + b, k + c] > 0.5 \
                                        and slice_inds[i +a, j + b, k + c]:

                                    # Check if is self, face, edge, or corner neighbor
                                    # and add to the appropriate sum.
                                    if num_off == 0:
                                        sum_self += N_map[i, j, k] * N_map[i + a, j + b, k + c]
                                        count_self += 1
                                    elif num_off == 1:
                                        sum_face += N_map[i, j, k] * N_map[i + a, j + b, k + c]
                                        count_face += 1
                                    elif num_off == 2:
                                        sum_edge += N_map[i, j, k] * N_map[i + a, j + b, k + c]
                                        count_edge += 1
                                    elif num_off == 3:
                                        sum_corner += N_map[i, j, k] * N_map[i + a, j + b, k + c]
                                        count_corner += 1

    # Compute the covariances.
    cdef np.ndarray cov = np.zeros(shape=(4))
    cov[0] = sum_self / count_self
    cov[1] = sum_face / count_face
    cov[2] = sum_edge / count_edge
    cov[3] = sum_corner / count_corner
    cdef np.ndarray good_inds = np.logical_and(f_map > 0.5, slice_inds)
    cdef double mean = N_map[good_inds].mean()
    cov -= mean ** 2

    # Convert the covariance in the number counts into covariance in y.
    mu = y_map[good_inds].mean()
    print(cov)
    cov_y = cov / ((mean ** 2) * np.exp(2 * mu))

    return cov_y