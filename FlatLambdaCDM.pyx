import numpy as np
from scipy import interpolate
from astropy import cosmology, units

class FlatLambdaCDM(cosmology.FlatLambdaCDM):
    """Class that extends Astropy FlatLambdaCDM adding new features."""

    def __init__(self,H0,Om0,max_z=10,dz=0.01):
        cosmology.FlatLambdaCDM.__init__(self,H0=H0,Om0=Om0)
        self.max_z = max_z
        self.dz = dz

        self.setup_redshift_interpolators(max_z, dz)

    def setup_redshift_interpolators(self,max_z,dz):
        """Creates the interpolators for the redshift and comoving distance
        calculations.
        """

        # Compute the comoving distances for the appropriate redshifts.
        z = np.arange(0,max_z+dz,dz)
        com_dist = self.comoving_distance(z).to(units.Mpc)

        # Create the interpolators.
        self.z_interp = interpolate.interp1d(com_dist,z)
        self.com_dist_interp = interpolate.interp1d(z,com_dist)

    def redshift(self,com_dist):
        """Calculate redshift from comoving distance."""

        return self.z_interp(com_dist)

    def com_dist(self,redshift):
        """Calculate the comoving distances from redshift with units."""

        return self.com_dist_interp(redshift)