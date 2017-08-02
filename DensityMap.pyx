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

        # Get map/box properties (unitless) for convenience.
        x0 = self.box.x0.value
        y0 = self.box.y0.value
        z0 = self.box.z0.value
        nx = self.box.nx
        ny = self.box.ny
        nz = self.box.nz
        dl = self.box.vox_len.value

        # Compute the map edges for the histogram.
        x_edges = np.linspace(x0,x0+nx*dl,1+nx)
        y_edges = np.linspace(y0,y0+ny*dl,1+ny)
        z_edges = np.linspace(z0,z0+nz*dl,1+nz)

        # Compute the number counts.
        self.map,_ = np.histogramdd(cart_photo.value,(x_edges,y_edges,z_edges))
        self.N = self.map.copy()

        # Adjust the values to account for occupancy fractions.
        self.map /= np.minimum(1,self.occ_map.map)

        # Compute the delta values.
        rand_ratio = self.cat.cat_len / self.occ_map.cat.cat_len
        expected_N = self.occ_map.expected_n * rand_ratio \
                     * (self.box.vox_len ** 3)
        self.map = self.map/expected_N - 1

        return self.map

    def initialize_spec_map(self):
        """Initialize the density map to those from the mean photo-z values."""

        # Get the Cartesian coordinates of the photo-z means.
        cart_photo = self.cat.get_cart_spec()

        # Get map/box properties (unitless) for convenience.
        x0 = self.box.x0.value
        y0 = self.box.y0.value
        z0 = self.box.z0.value
        nx = self.box.nx
        ny = self.box.ny
        nz = self.box.nz
        dl = self.box.vox_len.value

        # Compute the map edges for the histogram.
        x_edges = np.linspace(x0,x0+nx*dl,1+nx)
        y_edges = np.linspace(y0,y0+ny*dl,1+ny)
        z_edges = np.linspace(z0,z0+nz*dl,1+nz)

        # Compute the number counts.
        self.map,_ = np.histogramdd(cart_photo.value,(x_edges,y_edges,z_edges))
        self.N = self.map.copy()

        # Adjust the values to account for occupancy fractions.
        self.map /= np.minimum(1,self.occ_map.map)

        # Compute the delta values.
        rand_ratio = self.cat.cat_len / self.occ_map.cat.cat_len
        expected_N = self.map[self.occ_map.map > 0.5].mean()
        self.map = self.map/expected_N - 1

        return self.map