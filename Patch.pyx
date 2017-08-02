import numpy as np
import matplotlib; matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy import interpolate
from astropy import constants

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

        self.compute_pdfs()

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

    def compute_pdfs(self):

        z_max = np.ceil(self.cat.z_spec[self.patch_gals].max())
        z_max = 1
        r_max = self.cosmo.com_dist(z_max)
        edges = np.arange(0,z_max,0.05)

        z_spec = self.cat.z_spec[self.patch_gals]
        z_photo = self.cat.z_photo[self.patch_gals]
        z_photo_err = self.cat.z_photo_err[self.patch_gals]

        DH = constants.c / self.cosmo.H0
        r_photo = self.cosmo.com_dist(self.cat.z_photo[self.patch_gals])
        r_photo_err = np.sqrt((z_photo_err ** 2) \
                      * ((DH * self.cosmo.inv_efunc(z_photo)) ** 2))

        unit_vec = self.cat.unit_vec[self.patch_gals,:]

        fine_edges = np.linspace(0,r_max,len(edges)*100)
        dr = fine_edges[1] - fine_edges[0]
        mid_rs = fine_edges[:-1] + dr
        mid_zs = self.cosmo.redshift(mid_rs)
        stacked_photo = np.zeros(shape=len(fine_edges)-1)
        stacked_improved = np.zeros(shape=len(fine_edges)-1)

        for ind in range(len(z_spec)):
            photo_dist = norm.pdf(mid_rs,r_photo[ind],r_photo_err[ind])*dr

            x = unit_vec[ind,0] * mid_rs
            y = unit_vec[ind,1] * mid_rs
            z = unit_vec[ind,2] * mid_rs

            i = np.floor((x - self.box.x0)/self.box.vox_len).astype(np.int32)
            j = np.floor((y - self.box.y0)/self.box.vox_len).astype(np.int32)
            k = np.floor((z - self.box.z0)/self.box.vox_len).astype(np.int32)

            contained_inds = np.logical_and(i >= 0, i < self.box.nx)
            contained_inds = np.logical_and(contained_inds,
                                            np.logical_and(j >= 0, j < self.box.ny))
            contained_inds = np.logical_and(contained_inds,
                                            np.logical_and(k >= 0, k < self.box.nz))

            i = i[contained_inds]
            j = j[contained_inds]
            k = k[contained_inds]

            delta_vals = self.d_map.map[i,j,k]

            contained_inds[contained_inds] *= np.isfinite(delta_vals)

            delta_vals = delta_vals[np.isfinite(delta_vals)]

            improved_dist = photo_dist.copy()
            improved_dist[contained_inds] *= 1 + delta_vals/1.5
            improved_dist[contained_inds] *= photo_dist[contained_inds].sum() \
                                             / improved_dist[contained_inds].sum()

            # plt.plot(mid_zs,improved_dist)
            # plt.plot(mid_zs,photo_dist)
            # plt.savefig('i.png')
            # plt.clf()
            #
            # plt.plot(mid_zs[contained_inds],1 + delta_vals/1.5)
            # plt.savefig('j.png')
            # plt.clf()
            #
            # print(improved_dist[contained_inds].sum())
            # print(photo_dist[contained_inds].sum())
            # exit(0)

            stacked_photo += photo_dist
            stacked_improved += improved_dist


        z_stacked_photo = interpolate.interp1d(mid_zs,stacked_photo)
        z_stacked_improved = interpolate.interp1d(mid_zs,stacked_improved)

        min_z = mid_zs.min()
        max_z = mid_zs.max()

        mid_zs = np.arange(0,z_max,0.05/100)
        mid_zs = mid_zs[:-1] + (mid_zs[1]-mid_zs[0])*0.5
        mid_zs = mid_zs[np.logical_and(mid_zs >= min_z,mid_zs <= max_z)]

        plt.hist(z_spec,edges)
        h,_ = np.histogram(mid_zs,edges,weights=z_stacked_photo(mid_zs))
        plt.plot(edges[:-1] + (edges[1]-edges[0])*0.5, h)
        h,_ = np.histogram(mid_zs,edges,weights=z_stacked_improved(mid_zs))
        plt.plot(edges[:-1] + (edges[1]-edges[0])*0.5, h)
        plt.savefig('patch.png')
        plt.clf()