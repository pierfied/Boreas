import numpy as np

class DensityMap:
    """Map of the density contrasts with associated functions."""

    def __init__(self,cat,cosmo,box):
        self.cat = cat
        self.cosmo = cosmo
        self.box = box

    def initialize_photo_map(self):
        """Initialize the density map to those from the mean photo-z values."""