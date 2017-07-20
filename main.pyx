import pyximport; pyximport.install()
from Catalog import Catalog
from OccupancyMap import OccupancyMap
from astropy.io import fits
from astropy.cosmology import FlatLambdaCDM
from astropy import units

# Define the cosmology.
cosmo = FlatLambdaCDM(H0=70,Om0=0.286)

# Load the data.
redmagic = fits.open('/home/pierfied/Downloads/data/redmagic.fit')[1].data
randoms = fits.open('/home/pierfied/Downloads/data/randoms.fit')[1].data

# Construct the catalogs.
redmagic_cat = Catalog(redmagic['RA'],redmagic['DEC'],cosmo,
                       redmagic['ZSPEC'],redmagic['ZREDMAGIC'],redmagic['ZREDMAGIC_E'])
randoms_cat = Catalog(randoms['RA'],randoms['DEC'],cosmo,randoms['Z'])

# Set the voxel side-length.
vox_len = 20 * units.Mpc