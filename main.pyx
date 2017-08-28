import numpy
import pyximport; pyximport.install(setup_args={"include_dirs":numpy.get_include()})
from Catalog import Catalog
from OccupancyMap import OccupancyMap
from astropy.io import fits
from FlatLambdaCDM import FlatLambdaCDM
import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from DensityMap import DensityMap
from Patch import Patch
import NeighborMapSampler as NMS
from NeighborMapSampler import NeighborMapSampler

# Define the cosmology.
cosmo = FlatLambdaCDM(H0=70,Om0=0.286)

# Load the data.
redmagic = fits.open('/calvin1/pierfied/sim/Asteria/data/redmagic.fit')[1].data
randoms = fits.open('/calvin1/pierfied/sim/Asteria/data/randoms.fit')[1].data
randoms_truth = fits.open('/calvin1/pierfied/sim/Asteria/data/randoms_truth.fits')[1].data
truth = fits.open('/calvin1/pierfied/sim/Asteria/data/cut_truth.fits')[1].data

# Construct the catalogs.
redmagic_cat = Catalog(redmagic['RA'],redmagic['DEC'],cosmo,
                       redmagic['ZSPEC'],redmagic['ZREDMAGIC'],redmagic['ZREDMAGIC_E'])
randoms_cat = Catalog(randoms['RA'],randoms['DEC'],cosmo,randoms['Z'])
randoms_truth_cat = Catalog(randoms_truth['RA'],randoms_truth['DEC'],cosmo,randoms_truth['Z'])
truth_cat = Catalog(truth['RA'],truth['DEC'],cosmo,truth['Z'],truth['ZPHOTO'],truth['ZPHOTO_E'])

# Set the voxel side-length.
vox_len = 20

# Create the bounding box for the survey.
box = redmagic_cat.gen_bounding_box(vox_len)
print(box.nx)
print(box.ny)
print(box.nz)
print(box.x0)
print(box.y0)
print(box.z0)

# Define the survey area for each catalog.
Omega = 0.27661288275
Omega_truth = 0.29190631739

# Calculate the occupancy maps.
f_map = OccupancyMap(randoms_cat,cosmo,box,Omega)
f_map_truth = OccupancyMap(randoms_truth_cat,cosmo,box,Omega_truth)

# Plot the distribution of occupancy values.
ind = f_map.map > 0
plt.hist(f_map.map[ind],bins=50)
plt.title('redmagic Randoms Occupancy Map')
plt.xlabel('Voxel Occupancies [%]')
plt.ylabel('Number of Voxels')
plt.tight_layout()
plt.savefig('f_hist.png')
plt.clf()

# Calculate the delta maps.
d_map_photo = DensityMap(redmagic_cat,cosmo,box,f_map)
d_map = DensityMap(redmagic_cat,cosmo,box,f_map)
d_map.initialize_spec_map()
d_map_truth = DensityMap(truth_cat,cosmo,box,f_map_truth)
d_map_truth.initialize_spec_map()

# Regularize the map or deal with low occupancy pixels.
d_map.map[f_map.map < 0.5] = 0
d_map_photo.map[f_map.map < 0.5] = 0
#d_map.regularize()

# Get the indices for all reasonably occupied voxels.
ind = f_map.map > 0.5

# Plot the delta distribution.
plt.hist(d_map_truth.map[ind],bins=100,range=(-1,5))
plt.title('i<23 $z_{spec}$ Density Map')
plt.xlabel('$\delta$')
plt.ylabel('Number of Voxels')
plt.tight_layout()
plt.savefig('d_hist.png')
plt.clf()

# Plot the y distribution.
y = np.log(1+d_map_truth.map[ind])
plt.hist(y[np.isfinite(y)],bins=50)
plt.title('i<23 Catalog $z_{spec}$ y Map')
plt.xlabel('$\ln(1+\delta)$')
plt.ylabel('Number of Voxels')
plt.tight_layout()
plt.savefig('y_hist.png')
plt.clf()

# Adjust the delta-map for the redmagic bias.
d_map.map /= 1.5

# Create some patches.
# p = Patch(truth_cat,d_map,cosmo,box,1,redmagic_cat,214211)
# print(p.center_gal)
# p.compute_stacked_pdfs('patch_1_sq_deg_a.png',1)
#
# p = Patch(truth_cat,d_map,cosmo,box,1,redmagic_cat,382063)
# print(p.center_gal)
# p.compute_stacked_pdfs('patch_1_sq_deg_b.png',1)
#
# p = Patch(truth_cat,d_map,cosmo,box,5,redmagic_cat,103970)
# print(p.center_gal)
# p.compute_stacked_pdfs('patch_5_sq_deg_a.png',1)
#
# p = Patch(truth_cat,d_map,cosmo,box,5,redmagic_cat,165208)
# print(p.center_gal)
# p.compute_stacked_pdfs('patch_5_sq_deg_b.png',1)

# Calculate the y-map for the truth catalog and adjust for redmagic bias.
bias = 1.5
y_map_truth = np.log(1+np.maximum(-1,d_map_truth.map * bias))

# Deal with empty pixels and set low occupancy pixels to the mean.
y_map_truth[np.isinf(y_map_truth)] = -5
y_map_truth[f_map_truth.map < 0.5] = y_map_truth[f_map_truth.map > 0.5].mean()

print(y_map_truth)
print(y_map_truth[f_map_truth.map > 0.5])

print(NMS.compute_neighbor_covariances(y_map_truth,f_map_truth.map))
print(y_map_truth[f_map_truth.map > 0.5].var())

mu = y_map_truth[f_map_truth.map > 0.5].mean()
cov = NMS.compute_neighbor_covariances(y_map_truth,f_map_truth.map)

a,b=NMS.compute_log_prob(y_map_truth,d_map.N,f_map.map,mu,cov,d_map.expected_N)
print('Gaussian: ',str(a))
print('Poisson: ',str(b))
print('Total: ',str(a+b))
print()

y_map_photo = np.log(1+d_map_photo.map)
y_map_photo[np.isinf(y_map_photo)] = -5
y_map_photo[f_map.map < 0.5] = mu

print(d_map_photo.expected_N)
print(d_map.expected_N)

a,b = NMS.compute_log_prob(y_map_photo,d_map_photo.N,f_map.map,mu,cov,d_map.expected_N)
print('Gaussian: ',str(a))
print('Poisson: ',str(b))
print('Total: ',str(a+b))
print()

a,b = NMS.compute_log_prob(y_map_photo,d_map_photo.N,f_map.map,mu,cov,d_map_photo.expected_N)
print('Gaussian: ',str(a))
print('Poisson: ',str(b))
print('Total: ',str(a+b))
print()

a,b = NMS.compute_log_prob(y_map_truth,d_map_photo.N,f_map.map,mu,cov,d_map_photo.expected_N)
print('Gaussian: ',str(a))
print('Poisson: ',str(b))
print('Total: ',str(a+b))
exit(0)

sampler = NeighborMapSampler(redmagic_cat,box,f_map_truth,y_map_truth,None)