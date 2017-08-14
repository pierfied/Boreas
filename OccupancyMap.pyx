import numpy as np
from astropy import units, constants

class OccupancyMap:
    """Map of the percentages of each voxel in the survey region (frac_good)."""
    def __init__(self,cat,cosmo,box,omega):
        self.cat = cat
        self.cosmo = cosmo
        self.box = box
        self.omega = omega

        self.calc_expected_n()

        self.calc_occupancies()

    def calc_expected_n(self,dz=0.01):
        """Compute the expected number counts density of the survey."""

        # Create the bins for the number counts calculations.
        min_z = np.min(self.cat.z_spec)
        max_z = np.max(self.cat.z_spec)
        edges = np.arange(min_z, max_z+dz, dz)
        mid_z = edges[:-1] + 0.5*dz

        # Compute the histogram of the counts.
        N,_ = np.histogram(self.cat.z_spec, edges)

        # Compute the number count densities.
        Dc = self.cosmo.com_dist(mid_z)
        delta_Dc = (constants.c/self.cosmo.H0 * self.cosmo.inv_efunc(mid_z)
                    * dz).to(units.Mpc).value
        n = N / ((Dc**2) * self.omega * delta_Dc)

        # Return the mean number count density.
        self.expected_n = np.mean(n)

        return self.expected_n

    def calc_occupancies(self):
        """Compute the occupancy fractions of the voxels."""

        # Compute the Cartesian coordinates of the mean photo-zs.
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

        # Compute the occupancy fractions.
        self.map,_ = np.histogramdd(cart_photo,(x_edges,y_edges,z_edges))
        expected_N = (self.expected_n * (self.box.vox_len ** 3))
        self.map /= expected_N

        return self.map