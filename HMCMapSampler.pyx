import numpy as np
cimport numpy as np
from time import time

class HMCMapSampler:
    """Class that performs HMC sampling on delta map using a redshift catalog."""

    def __init__(self,cat,box,f_map,d_map):
        self.cat = cat
        self.box = box
        self.f_map = f_map
        self.d_map = d_map

    def sample(self):
        """Perform HMC sampling using the specified parameters."""

        # Calculate the y values from delta and ensure values are define.
        y_map = np.log(1 + self.d_map.map)
        y_map[self.d_map.map == -1] = -5

        start = time()

        print(compute_face_covariance(y_map,self.f_map.map))
        print(compute_edge_covariance(y_map,self.f_map.map))

        stop = time()

        print(stop-start)

cdef double compute_face_covariance(np.ndarray y_map, np.ndarray f_map):

    # Get the shape of the array.
    cdef int nx = y_map.shape[0]
    cdef int ny = y_map.shape[1]
    cdef int nz = y_map.shape[2]

    cdef int i,j,k
    cdef int a,b,c
    cdef double sum = 0
    cdef double count = 0
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

                                # Check if this neighbor is a face, is within the map,
                                # and has reasonable occupancy.
                                if num_off == 1 and 0 <= i+a < nx and 0 <= j+b < ny \
                                        and 0 <= k+c < nz and f_map[i+a,j+b,k+c] > 0.5:
                                    sum += y_map[i,j,k] * y_map[i+a,j+b,k+c]
                                    count += 1

    cdef double mean = y_map[f_map > 0.5].mean()
    cdef double cov = sum/count - (mean ** 2)

    return cov

cdef double compute_edge_covariance(np.ndarray y_map, np.ndarray f_map):

    # Get the shape of the array.
    cdef int nx = y_map.shape[0]
    cdef int ny = y_map.shape[1]
    cdef int nz = y_map.shape[2]

    cdef int i,j,k
    cdef int a,b,c
    cdef double sum = 0
    cdef double count = 0
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

                                # Check if this neighbor is a face, is within the map,
                                # and has reasonable occupancy.
                                if num_off == 2 and 0 <= i+a < nx and 0 <= j+b < ny \
                                        and 0 <= k+c < nz and f_map[i+a,j+b,k+c] > 0.5:
                                    sum += y_map[i,j,k] * y_map[i+a,j+b,k+c]
                                    count += 1

    cdef double mean = y_map[f_map > 0.5].mean()
    cdef double cov = sum/count - (mean ** 2)

    return cov