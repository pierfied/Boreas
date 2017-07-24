import numpy as np
from astropy import units, constants

class OccupancyMap:
    """Map of the percentages of each voxel in the survey region (frac_good)."""
    def __init__(self,cat,cosmo,box,omega):
        self.cat = cat
        self.cosmo = cosmo
        self.box = box
        self.omega = omega

        print('Calculating expected n.')

        self.calc_expected_n()

        print('Calculating occupancies.')

        self.calc_occupancies()

    def calc_expected_n(self,dz=0.01):
        """Compute the expected number counts density of the survey."""

        # Create the bins for the number counts calculations.
        min_z = np.min(self.cat.z_spec)
        max_z = np.max(self.cat.z_spec)
        mid_z = np.arange(min_z+0.5*dz, max_z, dz)
        edges = np.arange(min_z, max_z+dz, dz)

        # Compute the histogram of the counts.
        N,_ = np.histogram(self.cat.z_spec, edges)

        # Compute the number count densities.
        Dc = self.cosmo.comoving_distance(mid_z)
        delta_Dc = constants.c/self.cosmo.H0 * self.cosmo.inv_efunc(mid_z) * dz
        n = N / ((Dc**2) * self.omega * delta_Dc)

        # Return the mean number count density.
        self.expected_n = np.mean(n).to(units.Mpc ** -3)
        return self.expected_n

    def calc_occupancies(self):
        """Compute the occupancy fractions of the voxels."""

        # Compute the Cartesian coordinates of the mean photo-zs.
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
        x_edges = np.arange(x0,x0+(1+nx)*dl,dl)
        y_edges = np.arange(y0,y0+(1+ny)*dl,dl)
        z_edges = np.arange(z0,z0+(1+nz)*dl,dl)

        # Compute the occupancy fractions.
        self.map,_ = np.histogramdd(cart_photo.value,(x_edges,y_edges,z_edges))
        expected_N = self.expected_n * (self.box.vox_len ** 3)
        self.map /= expected_N

        return self.map