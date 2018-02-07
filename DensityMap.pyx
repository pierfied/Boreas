import numpy as np

class DensityMap:
    """Map of the density contrasts with associated functions."""

    def __init__(self,cat,cosmo,box,occ_map):
        self.cat = cat
        self.cosmo = cosmo
        self.box = box
        self.occ_map = occ_map

        self.initialize_photo_map()

    def initialize_photo_map(self):
        """Initialize the density map to those from the mean photo-z values."""

        # Get the Cartesian coordinates of the photo-z means.
        cart_photo = self.cat.get_cart_photo_mean()

        # Get map/box properties for convenience.
        x0 = self.box.x0
        y0 = self.box.y0
        z0 = self.box.z0
        nx = self.box.nx
        ny = self.box.ny
        nz = self.box.nz
        dl = self.box.vox_len

        # Compute the map edges for the histogram.
        x_edges = np.linspace(x0,x0+nx*dl,1+nx)
        y_edges = np.linspace(y0,y0+ny*dl,1+ny)
        z_edges = np.linspace(z0,z0+nz*dl,1+nz)

        # Compute the middles of the bins.
        x_mids = np.tile(np.reshape(x_edges[:-1] + 0.5*dl, [nx, 1, 1]), [1, ny, nz])
        y_mids = np.tile(np.reshape(y_edges[:-1] + 0.5*dl, [1, ny, 1]), [nx, 1, nz])
        z_mids = np.tile(np.reshape(z_edges[:-1] + 0.5*dl, [1, 1, nz]), [nx, ny, 1])
        r_mids = np.sqrt(x_mids ** 2 + y_mids ** 2 + z_mids ** 2)
        z_mids = self.cosmo.redshift(r_mids)

        # Compute the number counts.
        self.map,_ = np.histogramdd(cart_photo,(x_edges,y_edges,z_edges))
        self.N = self.map.copy()

        # Adjust the values to account for occupancy fractions.
        self.map /= np.minimum(1,self.occ_map.map)
        self.N2 = self.map.copy()

        # Compute the delta values.
        rand_ratio = self.cat.cat_len / self.occ_map.cat.cat_len
        self.expected_N = self.occ_map.expected_n(z_mids) * rand_ratio \
                     * (self.box.vox_len ** 3)
        self.map = self.map/self.expected_N - 1

        self.z_mids = z_mids

        return self.map

    def regularize(self):
        ind = self.occ_map.map > 0.9

        delta = self.map.copy()
        y = np.log(1 + delta)
        y[delta == -1] = -3

        ind2 = np.logical_and(ind,np.isfinite(y))
        mean = y[ind2].mean()

        a = 0.5
        b = 0.1
        weights = 1/(1+np.exp(-(self.occ_map.map - a)/b))

        y = y * weights + mean * (1-weights)

        d = np.exp(y) - 1

        d[np.logical_not(np.isfinite(y))] = mean

        self.map = d

    def initialize_spec_map(self):
        """Initialize the density map to those from the mean photo-z values."""

        # Get the Cartesian coordinates of the photo-z means.
        cart_spec = self.cat.get_cart_spec()

        # Get map/box properties for convenience.
        x0 = self.box.x0
        y0 = self.box.y0
        z0 = self.box.z0
        nx = self.box.nx
        ny = self.box.ny
        nz = self.box.nz
        dl = self.box.vox_len

        # Compute the map edges for the histogram.
        x_edges = np.linspace(x0,x0+nx*dl,1+nx)
        y_edges = np.linspace(y0,y0+ny*dl,1+ny)
        z_edges = np.linspace(z0,z0+nz*dl,1+nz)

        # Compute the middles of the bins.
        x_mids = np.tile(np.reshape(x_edges[:-1] + 0.5*dl, [nx, 1, 1]), [1, ny, nz])
        y_mids = np.tile(np.reshape(y_edges[:-1] + 0.5*dl, [1, ny, 1]), [nx, 1, nz])
        z_mids = np.tile(np.reshape(z_edges[:-1] + 0.5*dl, [1, 1, nz]), [nx, ny, 1])
        r_mids = np.sqrt(x_mids ** 2 + y_mids ** 2 + z_mids ** 2)
        z_mids = self.cosmo.redshift(r_mids)

        # Compute the number counts.
        self.map,_ = np.histogramdd(cart_spec,(x_edges,y_edges,z_edges))
        self.N = self.map.copy()

        # Adjust the values to account for occupancy fractions.
        self.map /= np.minimum(1,self.occ_map.map)
        self.N2 = self.map.copy()

        # Compute the delta values.
        rand_ratio = self.cat.cat_len / self.occ_map.cat.cat_len
        rand_ratio = 0.4269865896/10
        self.expected_N = self.occ_map.expected_n(z_mids) * rand_ratio \
                     * (self.box.vox_len ** 3)
        self.map = self.map/self.expected_N - 1

        self.z_mids = z_mids

        return self.map