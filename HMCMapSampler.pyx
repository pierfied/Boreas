import numpy as np
cimport numpy as np
from time import time

class HMCMapSampler:
    """Class that performs HMC sampling on delta map using a redshift catalog."""

    def __init__(self,cat,box,f_map,d_map,cov):
        self.cat = cat
        self.box = box
        self.f_map = f_map
        self.d_map = d_map
        self.cov = cov

    def sample(self):
        """Perform HMC sampling using the specified parameters."""

        # Calculate the y values from delta and ensure values are define.
        y_map = np.log(1 + self.d_map.map)
        y_map[self.d_map.map == -1] = -5

        start = time()

        print(compute_neighbor_covariances(y_map, self.f_map.map))

        stop = time()

        print(stop-start)

cdef np.ndarray compute_neighbor_covariances(np.ndarray y_map, np.ndarray f_map):
    """Computes the covariances of y-values for face, edge, and corner neighbors."""

    # Get the shape of the array.
    cdef int nx = y_map.shape[0]
    cdef int ny = y_map.shape[1]
    cdef int nz = y_map.shape[2]

    cdef int i,j,k
    cdef int a,b,c
    cdef double sum_face = 0
    cdef double sum_edge = 0
    cdef double sum_corner = 0
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

                                    # Check if a face, edge, or corner neighbor
                                    # and add to the appropriate sum.
                                    if num_off == 1:
                                        sum_face += y_map[i,j,k] * y_map[i+a,j+b,k+c]
                                        count_face += 1
                                    elif num_off == 2:
                                        sum_edge += y_map[i,j,k] * y_map[i+a,j+b,k+c]
                                        count_edge += 1
                                    elif num_off == 3:
                                        sum_corner += y_map[i,j,k] * y_map[i+a,j+b,k+c]
                                        count_corner += 1

    # Compute the covariances.
    cdef np.ndarray cov = np.zeros(shape=(3))
    cov[0] = sum_face/count_face
    cov[1] = sum_edge/count_edge
    cov[2] = sum_corner/count_corner
    cdef double mean = y_map[f_map > 0.5].mean()
    cov -= mean ** 2

    return cov