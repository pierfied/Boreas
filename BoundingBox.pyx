class BoundingBox:
    """Simple class containing information on the box that bounds a given catalog."""
    def __init__(self,x0,y0,z0,nx,ny,nz,vox_len):
        self.x0 = x0
        self.y0 = y0
        self.z0 = z0
        self.nx = nx
        self.ny = ny
        self.nz = nz
        self.vox_len = vox_len