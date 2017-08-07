import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm

class Patch:
    """Class that updates the photo-z distributions using a given delta map."""

    def __init__(self,cat,d_map,cosmo,box,omega,center_cat=None,center_gal=None):
        self.cat = cat
        self.d_map = d_map
        self.cosmo = cosmo
        self.box = box
        self.omega = omega
        self.center_cat = center_cat
        self.center_gal = center_gal

        self.get_patch_gals()

    def get_patch_gals(self):
        """Determine which galaxies are members of a patch."""

        # If no catalog to draw patch centers is passed, use the main catalog.
        if self.center_cat is None:
            self.center_cat = self.cat

        # If no central galaxy was passed, randomly choose one.
        if self.center_gal is None:
            self.center_gal = np.random.choice(self.center_cat.cat_len,1)

        # Determine the middle of the patch.
        mid_dec = self.center_cat.dec[self.center_gal]
        mid_ra_cos_dec = self.center_cat.ra[self.center_gal] \
                         * np.cos(np.deg2rad(mid_dec))

        # Get the values of RA*cos(DEC) to ensure area is preserved.
        dec = self.cat.dec
        ra_cos_dec = self.cat.ra * np.cos(np.deg2rad(dec))

        # Compute the bounds of the patch.
        theta = np.sqrt(self.omega)
        min_dec = mid_dec - theta/2
        max_dec = mid_dec + theta/2
        min_ra_cos_dec = mid_ra_cos_dec - theta/2
        max_ra_cos_dec = mid_ra_cos_dec + theta/2

        # Get the patch galaxies.
        self.patch_gals = np.logical_and(np.logical_and(min_dec < dec, dec < max_dec),
                             np.logical_and(min_ra_cos_dec < ra_cos_dec,
                                            ra_cos_dec < max_ra_cos_dec))

        return self.patch_gals

    def compute_stacked_pdfs(self, f_name=None, z_max=None, dz=0.05, fine_res=100):

        # Set the max redshift to the max from the catalog if not provided.
        if z_max is None:
            z_max = self.cat.z_spec[self.patch_gals].max()

        # Compute the bin mids & edges for histogramming.
        edges = np.arange(0,z_max,dz)
        mids = edges[:-1] + dz/2
        num_bins = len(mids)

        # Compute the fine mids.
        r_edges = self.cosmo.com_dist(edges)
        fine_mids = np.zeros(shape=(num_bins,fine_res))
        for i in range(num_bins):
            fine_edges =  np.linspace(r_edges[i],r_edges[i+1],fine_res+1)
            dr = fine_edges[1]-fine_edges[0]
            fine_mids[i,:] = fine_edges[:-1] + dr/2

        # Get the unit vectors for the galaxies in the patch.
        unit_vec = self.cat.unit_vec[self.patch_gals,:]

        # Get the redshifts for each of the galaxies.
        z_spec = self.cat.z_spec[self.patch_gals]
        z_photo = self.cat.z_photo[self.patch_gals]
        z_photo_err = self.cat.z_photo_err[self.patch_gals]
        num_gals = len(z_spec)

        # Get map/box properties for convenience.
        x0 = self.box.x0
        y0 = self.box.y0
        z0 = self.box.z0
        nx = self.box.nx
        ny = self.box.ny
        nz = self.box.nz
        dl = self.box.vox_len

        # Calculate the photo-z distributions and LoS delta values.
        photo_dist = np.zeros(shape=(num_gals,num_bins))
        delta = np.zeros(shape=(num_gals,num_bins))
        updated_dist = np.zeros(shape=(num_gals,num_bins))
        for i in range(num_gals):
            # Calculate the probability in each bin.
            photo_dist[i,:] = norm.pdf(mids,z_photo[i],z_photo_err[i]) * dz

            # Calculate the Cartesian coordinates of each location.
            x = unit_vec[i,0] * fine_mids
            y = unit_vec[i,1] * fine_mids
            z = unit_vec[i,2] * fine_mids

            # Calculate the associated indices.
            a = ((x - x0)/dl).astype(np.int32)
            b = ((y - y0)/dl).astype(np.int32)
            c = ((z - z0)/dl).astype(np.int32)

            # Compute the delta values for each location and set delta to zero
            # for indices outside of the box.
            los_delta = np.zeros(shape=fine_mids.shape)
            valid_inds = np.logical_and(
                np.logical_and(a >=0, a < nx),
                np.logical_and(
                    np.logical_and(b >= 0, b < ny),
                    np.logical_and(c >= 0, c < nz)
                )
            )
            los_delta[valid_inds] = self.d_map.map[a[valid_inds],b[valid_inds],
                                                   c[valid_inds]]

            # Compute the average delta value for each bin.
            delta[i,:] = los_delta.mean(axis=1)

            # Compute the updated photo-z distribution and normalize.
            updated_dist[i,:] = photo_dist[i,:] * (1 + delta[i,:])
            updated_dist[i,:] *= photo_dist[i,:].sum() / updated_dist[i,:].sum()

        # Calculate the stacked distributions..
        stacked_photo = photo_dist.sum(axis=0)
        stacked_updated = updated_dist.sum(axis=0)

        # Check if a filename was provided.
        if f_name is None:
            f_name = 'patch.png'

        # Create dN/dz plot for the patch.
        plt.hist(z_spec,edges,label='$z_{spec}$')
        plt.plot(mids,stacked_photo,label='$z_{photo}$')
        plt.plot(mids,stacked_updated,label='$z_{photo}$ w/ $\delta$ Term')
        plt.xlabel('z')
        plt.ylabel('Number of Galaxies')
        plt.title(str(self.omega) + ' Sq. Deg. Patch')
        plt.legend()
        plt.tight_layout()
        plt.savefig(f_name)
        plt.clf()