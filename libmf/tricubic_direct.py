import numpy as np
import sys

__author__ = "Michael Weis"
__version__ = "0.0.1.0"

################################################################################
##### CLASS: direct tricubic interpolation #####################################
################################################################################
# This is tricubic interpolation using 3 passes 1d cubic interpolation;
# the interface is designed for tensor based full tricubic interpolation,
# but monotony enforcement is not working yet in the corresponding module.

tri_x =  np.expand_dims(np.array((0,1)), -1)
tri_p = np.array((0,1,2,3))
def d01(x,p,d=0):
    """ Evaluate 0. or 1. derivative d of x to the positive power of p """
    return np.logical_not(p<0)* p**d * x**np.clip(p-d,0,None)
tri_fd0 = d01(tri_x, tri_p, 0)
tri_fd1 = d01(tri_x, tri_p, 1)
tri_X = np.concatenate((tri_fd0, tri_fd1))
tri_M = np.linalg.inv(tri_X)

ijk = (-1,0,1,2)
giy, giz, gix = np.meshgrid(ijk, ijk, ijk)

def intp_cubic_local(data, dcx, monotone=True):
    secant = data[...,1:]-data[...,:-1]
    tangent = .5* (secant[...,:-1] +secant[...,1:])
    if monotone:
        # constrain tangent slope to preserve monotony -> prevent overshoot,
        # see Fritsch, Carlson 1978
        with np.errstate(divide='ignore', invalid='ignore'):
            alpha = tangent[...,:-1] / secant[...,1:-1]
            beta = tangent[...,1:] / secant[...,1:-1]
            tau = 3./np.sqrt(alpha**2+beta**2)
            modifier = np.ones_like(tangent)
            factor_a = np.nan_to_num(np.where(beta<0., 1./alpha, tau))
            np.clip(modifier[...,:-1], 0., factor_a*(alpha>=0.), modifier[...,:-1])
            factor_b = np.nan_to_num(np.where(alpha<0., 1./beta, tau))
            np.clip(modifier[...,1:], 0., factor_b*(beta>=0.), modifier[...,1:])
            tangent *= modifier
    #
    tri_a  = 0.
    tri_a += np.inner(tri_M[:,:2], (data[...,1:-1]))
    tri_a += np.inner(tri_M[:,2:], tangent)
    #
    xp = (np.expand_dims(dcx.T, -1)**tri_p)
    #
    return np.sum(tri_a.T*xp, axis=-1).T
        
def intp_tricubic_local(data, dcz, dcy, dcx, monotone=True):
    tx = intp_cubic_local(data, dcx, monotone=monotone)
    ty = intp_cubic_local(tx, dcy, monotone=monotone)
    tz = intp_cubic_local(ty, dcz, monotone=monotone)
    return tz
    
class tricubic(object):
    def __init__(self, gbld, verbose=False, ng=2, monotone=True):
        self.blocks, self.nz, self.ny, self.nx = (gbld[:,ng:-ng,ng:-ng,ng:-ng]).shape
        self.ng = ng
        self.monotone = bool(monotone)
        self.gbld = gbld
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        return False
    def __del__(self):
        pass
    def eval(self, block, fcellindex_):
        fcz, fcy, fcx = fcellindex_
        ng = self.ng
        # Separate float cell indicies into floor integer ic and offset dc
        ## Temporarily shift values up to evade rounding dirswitch at 0.
        icx = (fcx+ng).astype(np.int32) -ng
        icy = (fcy+ng).astype(np.int32) -ng
        icz = (fcz+ng).astype(np.int32) -ng
        # Calculate cell indicies float offset
        dcx = fcx-icx
        dcy = fcy-icy
        dcz = fcz-icz
        #
        ix = icx.reshape(icx.shape+(1,1,1)) +gix +ng
        iy = icy.reshape(icy.shape+(1,1,1)) +giy +ng
        iz = icz.reshape(icz.shape+(1,1,1)) +giz +ng
        ibl = block.reshape(block.shape+(1,1,1))
        #
        data = self.gbld[ibl,iz,iy,ix]
        return intp_tricubic_local(data, dcz, dcy, dcx, self.monotone)

# ==== TEST ====================================================================
if __name__ == '__main__':
    """
    import micflash as mf
    filename = 'M:/scratchbin/MW_supermuc/CF97L/3.TEST/CF97L_T3_hdf5_plt_cnt_0190'
    ffile = mf.plotfile(filename)
    ffile.learn(mf.var_mhd)
    ffile.learn(mf.var_ch5)
    ffile.learn(mf.var_grid)
    fgrid = mf.pm3dgrid(ffile)
    dd = ffile['numdens']
    ddgrid = mf.datagrid(dd, fgrid, verbose=True)
    
    block = 3344
    iz, iy, ix = 4, 1, 6
    dcz, dcy, dcx = np.array((.625)), np.array((.25)), np.array((.25))
    ng = 2
    
    data3 = ddgrid.gbld[block:block+1, iz+ng-1:iz+ng+3, iy+ng-1:iy+ng+3, ix+ng-1:ix+ng+3]
    x3 = intp_tricubic_local(data3, dcz, dcy, dcx)
    
    print(x3)
    """
    data = np.array([ 62.7801018 ,   4.29359123,   2.62801502,   2.44613934])
    dcx = np.array([0.2920792079208143])
    print(intp_cubic_local(data, dcx))
