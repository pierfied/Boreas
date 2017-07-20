import numpy as np
from BoundingBox import BoundingBox
from math import ceil

class Catalog:
    """Galaxy catalog and relevant functions."""
    def __init__(self,ra,dec,cosmo,z_spec=None,z_photo=None,z_photo_err=None):
        self.ra = ra
        self.dec = dec
        self.cosmo = cosmo
        self.z_spec = z_spec
        self.z_photo = z_photo
        self.z_photo_err = z_photo_err

        # If no photometric values assume z_spec with no errors.
        if z_photo is None:
            z_photo = z_spec
        if z_photo_err is None:
            z_photo_err = np.zeros(z_spec.shape)

        # Compute spherical coordinates.
        phi = np.deg2rad(self.ra)
        theta = np.pi/2 - np.deg2rad(self.dec)

        # Compute the unit vector components for each galaxy.
        self.x_unit = np.cos(phi) * np.sin(theta)
        self.y_unit = np.sin(phi) * np.sin(theta)
        self.z_unit = np.cos(theta)

    def gen_cart_samples(self):
        """Generate comoving Cartesian samples for the galaxies."""

        # Draw new redshifts.
        z_samp = np.random.normal(self.z_photo,self.z_photo_err)

        # Calculate the comoving Cartesian coordinates.
        r_samp = self.cosmo.comoving_distance(z_samp)
        x_samp = r_samp * self.x_unit
        y_samp = r_samp * self.y_unit
        z_samp = r_samp * self.z_unit

        return x_samp,y_samp,z_samp

    def gen_bounding_box(self,vox_len):
        """Calculate the box that bounds all galaxies in the catalog."""

        # Calculate the cartesian coordinates for all galaxies w/ photo_z +/- 5*sigma.
        low_z_photo = self.z_photo - 5*self.z_photo_err
        high_z_photo = self.z_photo + 5*self.z_photo_err

        low_r = self.cosmo.comoving_distance(low_z_photo)
        high_r = self.cosmo.comoving_distance(high_z_photo)

        low_x = self.x_unit * low_r
        low_y = self.y_unit * low_r
        low_z = self.y_unit * low_r

        high_x = self.x_unit * high_r
        high_y = self.y_unit * high_r
        high_z = self.z_unit * high_r

        # Calculate the min and max x,y,z values.
        min_x = min(low_x.min(),high_x.min())
        min_y = min(low_y.min(),high_y.min())
        min_z = min(low_z.min(),high_z.min())

        max_x = max(low_x.max(),high_x.max())
        max_y = max(low_y.max(),high_y.max())
        max_z = max(low_z.max(),high_z.max())

        # Calculate the number of voxels on each side.
        nx = int(ceil((max_x - min_x)/vox_len))
        ny = int(ceil((max_y - min_y)/vox_len))
        nz = int(ceil((max_z - min_z)/vox_len))

        # Create and return the bounding box.
        box = BoundingBox(min_x,min_y,min_z,nx,ny,nz,vox_len)
        return box