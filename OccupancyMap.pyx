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