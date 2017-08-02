import pyximport; pyximport.install()
from Catalog import Catalog
from OccupancyMap import OccupancyMap
from astropy.io import fits
from FlatLambdaCDM import FlatLambdaCDM
from astropy import units
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from DensityMap import DensityMap
from Patch import Patch

# Define the cosmology.
cosmo = FlatLambdaCDM(H0=70,Om0=0.286)

#print(cosmo.com_dist(1))
#exit(0)

# Load the data.
redmagic = fits.open('/calvin1/pierfied/sim/Asteria/data/redmagic.fit')[1].data
randoms = fits.open('/calvin1/pierfied/sim/Asteria/data/randoms.fit')[1].data
truth = fits.open('/calvin1/pierfied/sim/Asteria/data/cut_truth.fits')[1].data

# Construct the catalogs.
redmagic_cat = Catalog(redmagic['RA'],redmagic['DEC'],cosmo,
                       redmagic['ZSPEC'],redmagic['ZREDMAGIC'],redmagic['ZREDMAGIC_E'])
randoms_cat = Catalog(randoms['RA'],randoms['DEC'],cosmo,randoms['Z'])
truth_cat = Catalog(truth['RA'],truth['DEC'],cosmo,truth['Z'],truth['ZPHOTO'],truth['ZPHOTO_E'])

# Set the voxel side-length.
vox_len = 20 * units.Mpc

box = redmagic_cat.gen_bounding_box(vox_len)
print(box.nx)
print(box.ny)
print(box.nz)
print(box.x0)
print(box.y0)
print(box.z0)

#print(redmagic_cat.gen_cart_samples()[0].shape)

Omega = 0.27661288275

f_map = OccupancyMap(randoms_cat,cosmo,box,Omega)

ind = f_map.map > 0

print(f_map.map.shape)
print(np.max(f_map.map))
print(np.mean(f_map.map[ind]))

plt.hist(f_map.map[ind],bins=50)
plt.title('redmagic Randoms Occupancy Map')
plt.xlabel('Voxel Occupancies [%]')
plt.ylabel('Number of Voxels')
plt.tight_layout()
plt.savefig('f_map.png')
plt.clf()

d_map = DensityMap(redmagic_cat,cosmo,box,f_map)
d_map.initialize_spec_map()

ind = f_map.map > 0.5

print('d_mean: ' + str(d_map.map[ind].mean()))

bins = np.arange(0,11)
h,_ = np.histogram(d_map.N[ind],bins=bins,density=True)
plt.bar(bins[:-1],h/np.sum(h),width=0.1)
plt.title('redmagic Number Counts')
plt.xlabel('Number of redmagic Galaxies')
plt.ylabel('Number of Voxels')
plt.tight_layout()
plt.savefig('n_map.png')
plt.clf()

plt.hist(d_map.map[ind],bins=50)
plt.title('redmagic Mean Photo-z Density Map')
plt.xlabel('$\delta$')
plt.ylabel('Number of Voxels')
plt.tight_layout()
plt.savefig('d_map.png')
plt.clf()

plt.hist2d(f_map.map[ind],d_map.N[ind],200)
plt.colorbar()
plt.tight_layout()
plt.savefig('2n.png')
plt.clf()

plt.hist2d(f_map.map[ind],d_map.map[ind],200)
plt.colorbar()
plt.tight_layout()
plt.savefig('2d.png')
plt.clf()

Patch(truth_cat,d_map,cosmo,box,5,redmagic_cat)