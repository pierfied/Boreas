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

        # Throw an error if there is neither z_spec or z_photo passed.
        if z_spec is None and z_photo is None:
            raise ValueError('At least one of z_spec or z_photo must be passed.')

        # If no spectroscopic values assume z_photo.
        if z_spec is None:
            self.z_spec = z_photo

        # If no photometric values assume z_spec with no errors.
        if z_photo is None:
            self.z_photo = z_spec
        if z_photo_err is None:
            self.z_photo_err = np.zeros(z_spec.shape)

        # Compute spherical coordinates.
        phi = np.deg2rad(self.ra)
        theta = np.pi/2 - np.deg2rad(self.dec)

        # Compute the unit vector components for each galaxy.
        x_unit = np.cos(phi) * np.sin(theta)
        y_unit = np.sin(phi) * np.sin(theta)
        z_unit = np.cos(theta)
        self.unit_vec = np.stack([x_unit,y_unit,z_unit], axis=1)

    def gen_cart_samples(self):
        """Generate comoving Cartesian samples for the galaxies."""

        # Draw new redshifts.
        z_samp = np.random.normal(self.z_photo,self.z_photo_err)

        # Calculate the comoving Cartesian coordinates.
        r_samp = np.tile(self.cosmo.com_dist(z_samp),(3,1)).T
        cart_samp = self.unit_vec * r_samp

        return cart_samp

    def get_cart_photo_mean(self):
        """Generate comoving Cartesian samples for the galaxies."""

        # Calculate the comoving Cartesian coordinates.
        r_samp = np.tile(self.cosmo.com_dist(self.z_photo),(3,1)).T
        cart_samp = self.unit_vec * r_samp

        return cart_samp

    def get_cart_spec(self):
        """Generate comoving Cartesian samples for the galaxies."""

        # Calculate the comoving Cartesian coordinates.
        r_spec = np.tile(self.cosmo.com_dist(self.z_spec),(3,1)).T
        cart_spec = self.unit_vec * r_spec

        return cart_spec

    def gen_bounding_box(self,vox_len):
        """Calculate the box that bounds all galaxies in the catalog."""

        # Calculate the cartesian coordinates for all galaxies w/ photo_z +/- 5*sigma.
        low_z_photo = self.z_photo - 5*self.z_photo_err
        high_z_photo = self.z_photo + 5*self.z_photo_err

        low_r = np.tile(self.cosmo.com_dist(low_z_photo),(3,1)).T
        high_r = np.tile(self.cosmo.com_dist(high_z_photo),(3,1)).T

        low_cart = low_r * self.unit_vec
        high_cart = high_r * self.unit_vec

        # Calculate the min and max x,y,z values.
        min_cart = np.minimum(low_cart.min(0),high_cart.min(0))
        max_cart = np.maximum(low_cart.max(0),high_cart.max(0))

        # Calculate the number of voxels on each side.
        nx = int(ceil((max_cart[0] - min_cart[0])/vox_len))
        ny = int(ceil((max_cart[1] - min_cart[1])/vox_len))
        nz = int(ceil((max_cart[2] - min_cart[2])/vox_len))

        # Create and return the bounding box.
        box = BoundingBox(min_cart[0],min_cart[1],min_cart[2],nx,ny,nz,vox_len)

        return box