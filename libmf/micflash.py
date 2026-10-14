# -*- coding: utf-8 -*-
import numpy as np
import sys
from math import floor, ceil
from copy import deepcopy
from .micvar import plotfile, flashfile, var_grid, var_mhd, var_ch5, var_ch15, var_2d, var_chx
#from .micvar import plotfile, flashfile, OpenPlotfile
import libmf.blocktree as blocktree
from .datagrid import datagrid, divergence

__author__ = "Michael Weis"
__version__ = "1.0.0.1"

# ==== HELPER ==================================================================
def rangeselect(L, L_min, L_max):
    a = np.array(L)
    mask = (a>=L_min)*(a<L_max)
    return a[mask]

def modclip(a, a_min, a_max):
    return a_min + np.mod(a-a_min, a_max-a_min)
    
def pscale(X):
    """
    Rank data from 0.0 to 1.0 using cubic rank spacing.
    Useful for displaying data of unknown/irregular scaling behaviour.
    """
    S = np.sort(X, axis=None)
    R = np.searchsorted(S, X)
    P = R*1./R.max()
    return P**3

# ==== PARAMETER READER ========================================================
def ff_lmax(ffile, default=None):
    try:
        ljeans_max = ffile['integer runtime parameters/lrefine_max_jeans']
        lstd_max = ffile['integer runtime parameters/lrefine_max']
        refine_max = max(ljeans_max, lstd_max)
    except KeyError:
        try:
            refine_max = ffile['integer runtime parameters/lrefine_max']
        except KeyError:
            refine_max = default
    return refine_max
    
def ff_usegrav(ffile, default=True):
    try:
        sim_gravity = ffile['logical runtime parameters/usegravity']
    except KeyError:
        # Workaround for h5py bug not reading 'logical runtime parameters' in 2.60:
        try:
            gpot = ffile['gpot']
            sim_gravity = ( gpot.min() != gpot.max() )
        except KeyError:
            sim_gravity = default
    return sim_gravity


################################################################################
##### CLASS: flat leaf-level blocktree (mainly 1D / 2D / uniform 3D)  ##########
################################################################################

class FlatTree(object):
    def __init__(self, block_bbox, nodetype):
        # 3D Array block_bbox[nblock,dim.(x/y/z),orient.(min/max)]
        # 1D Array nodetype[nblock] (=1 for leafs, =0 for inner blocks)
        
        leafblock_mask = (nodetype == 1)
        lfnblock = np.arange(len(nodetype))[leafblock_mask]

        self._bbox_min = block_bbox[:,:,0]
        self._bbox_max = block_bbox[:,:,1]
        self._bbox_size = self._bbox_max -self._bbox_min

        self._leaf_size = self._bbox_size.min(axis=0)
        self._domain_min = self._bbox_min.min(axis=0)
        self._domain_max = self._bbox_max.max(axis=0)
        self._domain_size = self._domain_max - self._domain_min
        #print self._domain_min, self._domain_max
        
        #pivot_dim = np.argmax(self._leaf_size)
        #trivial_dims = self._leaf_size < 1e-8*self._leaf_size[pivot_dim]
        #self._leaf_size[trivial_dims] = self._leaf_size[pivot_dim]
        #self._domain_size[trivial_dims] = self._leaf_size[pivot_dim]
        #self._domain_min[trivial_dims] -= .5*self._leaf_size[pivot_dim]
        #print 'FlatTree: Trivial Dims:', trivial_dims
        
        domainres_f = self._domain_size / self._leaf_size
        #print domainres_f
        self._domainres = list((domainres_f+.5).astype(np.int32))
        
        bbox_fimin = (self._bbox_min -self._domain_min) / self._leaf_size
        bbox_fimax = (self._bbox_max -self._domain_min) / self._leaf_size

        lfbbox_imin = (bbox_fimin[leafblock_mask] +.5).astype(np.int32)
        lfbbox_iend = (bbox_fimax[leafblock_mask] +.5).astype(np.int32)
            
        self._leafblock_LUT = np.full(self._domainres, -1)
        for nbl, imin, iend in zip(lfnblock, lfbbox_imin, lfbbox_iend):
            #print nbl, imin, iend
            self._leafblock_LUT[imin[0]:iend[0],imin[1]:iend[1],imin[2]:iend[2]] = nbl
            
        if np.any(self._leafblock_LUT < 0):
            raise RuntimeError('FlatTree Init: Incomplete Leaf LUT')

    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.Flush()
        return False
    def __del__(self):
        self.Flush()

    def Findblock(self, x, y, z):
        LUT_fix = ((x -self._domain_min[0]) / self._leaf_size[0])
        LUT_fiy = ((y -self._domain_min[1]) / self._leaf_size[1])
        LUT_fiz = ((z -self._domain_min[2]) / self._leaf_size[2])
        LUT_ix = np.clip(LUT_fix.astype(np.int32), 0, self._domainres[0]-1)
        LUT_iy = np.clip(LUT_fiy.astype(np.int32), 0, self._domainres[1]-1)
        LUT_iz = np.clip(LUT_fiz.astype(np.int32), 0, self._domainres[2]-1)
        nbl = self._leafblock_LUT[LUT_ix, LUT_iy, LUT_iz]
        return nbl
        
    def Flush(self):
        if self._leafblock_LUT is not None:
            self._leafblock_LUT = None
        return 1


################################################################################
##### CLASS: flash file 3d paramesh octree structure ###########################
################################################################################

class pm3dgrid(object):
    def __init__(self, flashfile, verbose=True, force_flat_tree=False):
        if verbose:
            print('Reading Layout Data...')
            sys.stdout.flush()
        block_bbox = flashfile['bounding box']
        self.__nxb = flashfile['integer scalars/nxb']
        self.__nyb = flashfile['integer scalars/nyb']
        self.__nzb = flashfile['integer scalars/nzb']
        self.__rlevel = flashfile['refine level']
        self.__ntype = flashfile['node type']
        if verbose:
            print('Processing Layout Data...')
            sys.stdout.flush()
        self._nb = np.array((self.__nxb, self.__nyb, self.__nzb))
        self._blockcells = self.__nxb*self.__nyb*self.__nzb
        self._bbox_min = block_bbox.min(axis=2)
        self.__block_xmin = self._bbox_min[:,0]
        self.__block_ymin = self._bbox_min[:,1]
        self.__block_zmin = self._bbox_min[:,2]
        self._bbox_max = block_bbox.max(axis=2)
        self._bbox_center = .5*(self._bbox_min+self._bbox_max)
        self._bbox_size = self._bbox_max - self._bbox_min
        self._bbox_minsize = self._bbox_size.min(axis=0)
        self._bbox_maxsize = self._bbox_size.max(axis=0)
        self._domain_min = self._bbox_min.min(axis=0)
        self._domain_max = self._bbox_max.max(axis=0)
        self._domain_size = self._domain_max - self._domain_min
        self._domain_blocks = (self._domain_size/self._bbox_minsize +.5).astype(np.int32)
        self._domain_cells = self._domain_blocks * self._nb
        # Precalculate (reciprocal) values to aleviate findblock/findblockcell
        self.__cell_xsize_recp = self.__nxb * 1./self._bbox_size[:,0]
        self.__cell_ysize_recp = self.__nyb * 1./self._bbox_size[:,1]
        self.__cell_zsize_recp = self.__nzb * 1./self._bbox_size[:,2]
        # Store information about periodicity
        self.__xl_periodic = str(flashfile['string runtime parameters/xl_boundary_type']).lower() == 'periodic'
        self.__yl_periodic = str(flashfile['string runtime parameters/yl_boundary_type']).lower() == 'periodic'
        self.__zl_periodic = str(flashfile['string runtime parameters/zl_boundary_type']).lower() == 'periodic'
        self.__xr_periodic = str(flashfile['string runtime parameters/xr_boundary_type']).lower() == 'periodic'
        self.__yr_periodic = str(flashfile['string runtime parameters/yr_boundary_type']).lower() == 'periodic'
        self.__zr_periodic = str(flashfile['string runtime parameters/zr_boundary_type']).lower() == 'periodic'
        self.__x_periodic = self.__xl_periodic and self.__xr_periodic
        self.__y_periodic = self.__yl_periodic and self.__yr_periodic
        self.__z_periodic = self.__zl_periodic and self.__zr_periodic
        # Store information about dimensionality
        self.__trivial_dim = (self._nb == 1)
        self.__uniform_dim = (self._bbox_minsize / self._bbox_maxsize) > (1.-1e-4)
        ### Build a block index tree indicating the spacial block layout
        if np.any(self.__trivial_dim) or np.all(self.__uniform_dim) or force_flat_tree:
            if verbose:
                print('Building Block Index Field...')
                sys.stdout.flush()
            self._blocktree = FlatTree(block_bbox, self.__ntype.astype(np.uint64))
        else:
            if verbose:
                print('Building Block Index Octree...')
                sys.stdout.flush()
            self._blocktree = blocktree.blocktree(block_bbox, self.__ntype.astype(np.uint64))
        ###
        self.__cell_coords = None # postpone construction till needed
        self.__cellbox_min = None # postpone construction till needed
        self.__cellbox_max = None # postpone construction till needed
        self.__tree = None # postpone construction till needed
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self._blocktree.Flush()
        return False
    def __del__(self):
        try:
            self._blocktree.Flush()
        except AttributeError:
            pass
    def __build_cell_coords(self):
        if self.__cell_coords is None:
            ### Calculate coordinates of the cells inside the blocks
            x_reloffset = (np.linspace(-.5, .5, self.__nxb, endpoint=False) + .5/self.__nxb)
            y_reloffset = (np.linspace(-.5, .5, self.__nyb, endpoint=False) + .5/self.__nyb)
            z_reloffset = (np.linspace(-.5, .5, self.__nzb, endpoint=False) + .5/self.__nzb)
            reloffset = np.array(np.meshgrid(y_reloffset, z_reloffset, x_reloffset))[(2,0,1),:,:,:]
            offset = (self._bbox_size).reshape(-1,3,1,1,1)*reloffset
            self.__cell_coords = (self._bbox_center).reshape(-1,3,1,1,1) +offset
            cellbox_radius = (.5*self._bbox_size/self._nb).reshape(-1,3,1,1,1)
            self.__cellbox_min = self.__cell_coords - cellbox_radius
            self.__cellbox_max = self.__cell_coords + cellbox_radius
        else:
            pass            
    def __repr__(self, level=0):
        answer  = 'Flashfile 3D Paramesh AMR Grid' + '\n'
        answer += 'Block layout:  ' + str(self._domain_blocks) + ' '
        answer += '(' + str(len(self._bbox_min)) + ')' + '\n'
        answer += 'Block celling: ' + str(self._nb) + '\n'
        answer += 'Cell layout:   ' + str(self._domain_cells) + ' '
        answer += '(' + str(len(self._bbox_center)*self._blockcells) + ')' + '\n'
        return answer
    
    def periodic(self):
        prdc = self.__x_periodic, self.__y_periodic, self.__z_periodic
        return prdc
        
    def blk_rlevel(self):
        return self.__rlevel
        
    def blk_ntype(self):
        return self.__ntype
        
    def blk_shape(self):
        return (len(self.__rlevel), self.__nzb, self.__nyb, self.__nxb)
        
    def cell_rlevel(self):
        shape_ones = np.ones(self.blk_shape(), dtype=int)
        return self.__rlevel[:,None,None,None] *shape_ones
    
        
    def coords(self, block=slice(None), axis=slice(None), ng=0, force=False):
        """ Get metrics of the bounding boxes of the blocks """
        if not force:
            raise RuntimeError('octree.coords is deprecated. Please replace hdfdata[pos_].')
        # ng: number of guard cells on each side
        # Calculate relative block extent correction factors for guard cell consideration
        fx = (self.__nxb+2.*ng) / self.__nxb
        fy = (self.__nyb+2.*ng) / self.__nyb
        fz = (self.__nzb+2.*ng) / self.__nzb
        #
        x_reloffset = np.linspace(-.5*fx, .5*fx, self.__nxb+2*ng, endpoint=False) + .5/self.__nxb
        y_reloffset = np.linspace(-.5*fy, .5*fy, self.__nyb+2*ng, endpoint=False) + .5/self.__nyb
        z_reloffset = np.linspace(-.5*fz, .5*fz, self.__nzb+2*ng, endpoint=False) + .5/self.__nzb
        reloffset = np.array(np.meshgrid(y_reloffset, z_reloffset, x_reloffset))[(2,0,1),:,:,:]
        coords  = reloffset[axis] * self._bbox_size[block,axis,None,None,None]
        coords += self._bbox_center[block,axis,None,None,None]
        return coords
        
    def cellsize(self, block=slice(None), axis=None):
        if axis is None:
            return np.array((self.cellsize(block=block, axis=0),
                             self.cellsize(block=block, axis=1),
                             self.cellsize(block=block, axis=2)))
        elif not axis in (0,1,2):
            raise ValueError(f'Axis {axis} out of range(3).')
        return self._bbox_size[block,axis] / self._nb[axis]

    def fingerprint(self):
        try:
            hashable_repr = (self._bbox_center).tostring()
        except AttributeError:
            hashable_repr = (self._bbox_center).tobytes()
        return hash(hashable_repr)
        
# --- FIND BLOCK ID BY COORDINATE -------------------------------------------------
    def findblock(self, x, y, z, lmax=None):
        """
        Return the index of the block containing the coordinates x, y, z.
        Capable of processing multiple data on receiving vectorized input.
        """
        if lmax is None:
            return self._blocktree.Findblock(x, y, z).astype(np.int64)
        else:
            return self._blocktree.Findblock_lmax(x, y, z, int(lmax)).astype(np.int64)
    
# --- FIND BLOCK AND CELL ID BY COORDINATE -------------------------------------
    def findblockcell(self, block, x, y, z):
        """
        Return the index of the blocks cell containing the coordinates x, y, z.
        Make sure that the provided block actually containes the coordinates.
        If the correct block index is unknown, findblock should be used in advance.
        Capable of processing multiple data on receiving vectorized input.
        """
        ix = ((x-self.__block_xmin[block])*self.__cell_xsize_recp[block]).astype(np.int32)
        iy = ((y-self.__block_ymin[block])*self.__cell_ysize_recp[block]).astype(np.int32)
        iz = ((z-self.__block_zmin[block])*self.__cell_zsize_recp[block]).astype(np.int32)
        i = np.clip(ix, 0, self.__nxb-1)
        j = np.clip(iy, 0, self.__nyb-1)
        k = np.clip(iz, 0, self.__nzb-1)
        return (k, j, i)
        
    def findcell(self, x, y, z, lmax=None):
        """
        Return the index of the block and the flat index of the blockcell
        containing the coordinates x, y, z.
        Capable of processing multiple data on receiving vectorized input.
        """
        block = self.findblock(x, y, z, lmax)
        kji = self.findblockcell(block, x, y, z)
        return block, kji
    
    def find(self, x, y, z, lmax=None, wrap=True):
        """
        Return the index of the block and the flat index of the blockcell
        containing the coordinates x, y, z.
        Capable of processing multiple data on receiving vectorized input.
        """
        xw, yw, zw = x, y, z
        if wrap:
            xmin, ymin, zmin = self._domain_min
            xmax, ymax, zmax = self._domain_max
            if self.__x_periodic: xw = modclip(x, xmin, xmax)
            if self.__y_periodic: yw = modclip(y, ymin, ymax)
            if self.__z_periodic: zw = modclip(z, zmin, zmax)
        block = self.findblock(xw, yw, zw, lmax)
        k,j,i = self.findblockcell(block, xw, yw, zw)
        return (block,k,j,i)
    
# --- FIND BLOCK AND FRACTIONAL CELL ID BY COORDINATE --------------------------
    def findblockfc(self, block, x, y, z):
        """
        Return the float index of the blocks cell containing the coordinates x, y, z.
        Make sure that the provided block actually containes the coordinates!
        If the correct block index is unknown, findblock should be used in advance.
        Capable of processing multiple data on receiving vectorized input.
        """
        fx = (x-self.__block_xmin[block])*self.__cell_xsize_recp[block] -.5
        fy = (y-self.__block_ymin[block])*self.__cell_ysize_recp[block] -.5
        fz = (z-self.__block_zmin[block])*self.__cell_zsize_recp[block] -.5
        return (fz, fy, fx)
        
    def findfc(self, x, y, z):
        """
        Return the index of the block and the float index of the blockcell
        containing the coordinates x, y, z.
        Capable of processing multiple data on receiving vectorized input.
        """
        block = self.findblock(x, y, z)
        return block, self.findblockfc(block, x, y, z)

# --- NEIGHBORHOOD DISCOVERY ---------------------------------------------------
    def findneigh(self, block, iz_bl, iy_bl, ix_bl, combine=False):
        """
        Implementation in progress; NOT YET CHECKED!
        """
        select_outskirt = np.array(np.meshgrid(*3*((1,0,0,1),))).sum(axis=0) > 0
        # Calculate coordinates of given cells:
        x_reloffset = np.linspace(-.5, .5, self.__nxb, endpoint=False) +.5/self.__nxb
        y_reloffset = np.linspace(-.5, .5, self.__nyb, endpoint=False) +.5/self.__nyb
        z_reloffset = np.linspace(-.5, .5, self.__nzb, endpoint=False) +.5/self.__nzb
        grid_reloffset = np.array(np.meshgrid(y_reloffset, z_reloffset, x_reloffset))
        reloffset = grid_reloffset[(2,0,1),:,:,:][:,iz_bl,iy_bl,ix_bl].T
        offset = self._bbox_size[block,:]*reloffset
        coords = offset +self._bbox_center[block,:]
        # Calculate coords of all potential neigborhood cells,
        # considering they may be refined by one level:
        cellsize = self.cellsize(block).T
        n_reloffset = np.array([-.75, -.25, .25, .75])
        grid_neigh_reloffset = np.array(np.meshgrid(*3*(n_reloffset,)))
        # neigh_reloffset = grid_neigh_reloffset[(2,0,1),:,:,:].reshape(3,-1)
        neigh_reloffset = grid_neigh_reloffset[(2,0,1),:,:,:][:,select_outskirt]
        neigh_offset = cellsize[:,:,None] *neigh_reloffset
        neigh_coords = coords[:,:,None] +neigh_offset
        #X, Y, Z = neigh_coords[:,0], neigh_coords[:,1], neigh_coords[:,2]        
        # Wrap neighborhood coordinates around cell boundaries
        ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = self.extent()
        X = modclip(neigh_coords[:,0], xmin, xmax)
        Y = modclip(neigh_coords[:,1], ymin, ymax)
        Z = modclip(neigh_coords[:,2], zmin, zmax)
        # Get blocks+cells from neighbor cell coordinates
        neigh_block, (neigh_iz, neigh_iy, neigh_ix) = self.findcell(X, Y, Z)
        # Remove doublettes; obviously, data has to be repacked to lists here,
        # because the number of neighbor doublettes per cell is not constant,
        # so the result will not be of regular format.
        if combine:
            N = set(zip(neigh_block.ravel(), neigh_iz.ravel(), neigh_iy.ravel(), neigh_ix.ravel()))
        else:
            result_iter = list(zip(neigh_block, neigh_iz, neigh_iy, neigh_ix))
            N = [set(zip(*cellneigh)) for cellneigh in result_iter]
        return deepcopy(N)
        
# --- SPATIAL BASEGRID ---------------------------------------------------------
    def extent(self, axis=None):
        xmin, ymin, zmin = self._domain_min
        xmax, ymax, zmax = self._domain_max
        extent = np.array([[xmin, xmax], [ymin, ymax], [zmin, zmax]])
        if axis is None:
            return extent
        elif axis in (0,1,2):
            ax_x, ax_y = {0:[1,2], 1:[0,2], 2:[0,1]}[axis]
            return np.array([extent[ax_x], extent[ax_y]])
        else:
            raise ValueError('Axis out of range.')
                    
    def resolution(self, axis=None):
        if axis is None:
            return self._domain_cells
        elif axis in (0,1,2):
            return self._domain_cells[axis]
        else:
            raise ValueError('Axis out of range.')
            
    def spacing(self, axis=None, res=None):
        if axis is None:
            if res is None:
                res = self._domain_cells
            if res[0] is None:
                res[0] = self._domain_cells[0]
            if res[1] is None:
                res[1] = self._domain_cells[1]
            if res[2] is None:
                res[2] = self._domain_cells[2]
            spacing = self._domain_size / res
        elif axis in (0,1,2):
            if res is None:
                res = self._domain_cells[axis]
            spacing = self._domain_size[axis]/res
        else:
            raise ValueError('Axis out of range.')
        return spacing
        
    def grating(self, axis, res=None, extent=None, align=None):
        """
        Partition the given axis of the given extend / entire simulation domain,
        equally into res parts (return the center coordinates of those parts).
        If no resolution is provided, the resolution will correspond to the
        best grid refinements resolution, with the coordinates in the centers
        of those grids cells.
        Typically used to construct grids for sampling points in the sim space.
        """
        if not axis in (0,1,2):
            raise ValueError('Axis out of range.')
        #
        if extent is None:
            extent_min = self._domain_min[axis]
            extent_max = self._domain_max[axis]
        else:
            extent_min, extent_max = extent
            
        if res is None or res=='full':
            provisional_step = self.spacing(axis)
        elif res=='half':
            provisional_step = self.spacing(axis) * 2.
        elif res=='quarter':
            provisional_step = self.spacing(axis) * 4.
        elif (0.<res) and (res<=1.):
            provisional_step = self.spacing(axis) * (1./res)
        else:
            provisional_step = (extent_max-extent_min) / res
            
        # If close to native res: align grating with cells
        if np.isclose(provisional_step, self.spacing(axis), rtol=1./7., atol=0.):
            align = True if align is None else bool(align)
            
        if align:
            grating_step = self.spacing(axis)
            grating_ref = self._domain_min[axis] +.5*grating_step
        else:
            grating_step = provisional_step
            extent_excess = (extent_max-extent_min) % grating_step
            grating_ref = extent_min +.5*grating_step +.5*extent_excess

        s = np.sign(extent_max-extent_min)
        ref_offset = extent_min-grating_ref
        ref_offset_steps = s*ceil(s*ref_offset/grating_step)
        grating_start = grating_ref +grating_step*ref_offset_steps
        grating_steps = floor((extent_max-extent_min) / grating_step)
        grating = grating_start +grating_step*np.arange(grating_steps) 
        return grating
            
    def basegrid_2d(self, axis, xres=None, yres=None, zres=None,
        x_extent=None, y_extent=None, z_extent=None):
        if axis == 0:
            z_grating = self.grating(2, zres, z_extent)
            y_grating = self.grating(1, yres, y_extent)
            Z, Y = np.meshgrid(z_grating, y_grating)
            X = None
        elif axis == 1:
            z_grating = self.grating(2, zres, z_extent)
            x_grating = self.grating(0, xres, x_extent)
            Z, X = np.meshgrid(z_grating, x_grating)
            Y = None
        elif axis == 2:
            y_grating = self.grating(1, yres, y_extent)
            x_grating = self.grating(0, xres, x_extent)
            Y, X = np.meshgrid(y_grating, x_grating)
            Z = None
        else:
            raise ValueError(f'Axis {axis} out of range.')
        return X, Y, Z

    def basegrid_3d(self, xres=None, yres=None, zres=None,
        x_extent=None, y_extent=None, z_extent=None):
        #
        y_grating = self.grating(1, yres, y_extent)
        x_grating = self.grating(0, xres, x_extent)
        z_grating = self.grating(2, zres, z_extent)
        return np.meshgrid(y_grating, x_grating, z_grating)

    def varblockgrating(self, axis):
        '''
        Return a grating along the given axis inside the simulation domain,
        so that the spacing is maximized while hitting the coordinate range
        of each block and along the axis at least once.
        The depth "range" covered by each step of the grating is given as well.
        The combination of both informations can be used to do integration or
        averaging along the axis by sampling block data that has already been
        averaged along the given axis per block.
        This is useful e.g. for rapidly calculating column densities.
        '''
        if axis not in (0,1,2):
            raise ValueError('Axis %i out of range(3).' % axis)
        blockbounds = np.concatenate((self._bbox_min[:,axis], self._bbox_max[:,axis]))
        II = (blockbounds-self._domain_min[axis])/self._bbox_minsize[axis]
        Ipartition = np.sort(list(set(np.around(II.T))))    
        IMid = .5*(Ipartition[1:]+Ipartition[:-1])
        IWidth = (Ipartition[1:]-Ipartition[:-1])
        depth = IWidth*self._bbox_minsize[axis]
        grating = self._domain_min[axis] + IMid*self._bbox_minsize[axis]
        #return grating, IWidth
        return grating, depth

    def varcellgrating(self, axis, extent=None):
        if axis not in (0,1,2):
            raise ValueError('Axis %i out of range(3).' % axis)
        self.__build_cell_coords()
        cellbounds = np.concatenate((
            self.__cellbox_min[:,axis].flatten(),
            self.__cellbox_max[:,axis].flatten(), ))
        II = (cellbounds-self._domain_min[axis])/self.spacing(axis=axis)
        Ipartition = np.sort(list(set(np.around(II.T))))    
        IMid = .5*(Ipartition[1:]+Ipartition[:-1])
        IWidth = Ipartition[1:]-Ipartition[:-1]    
        grating = self._domain_min[axis] + IMid*self.spacing(axis=axis)
        if extent is None:
            return grating, IWidth
        else:
            grating_min, grating_max = extent
            mask = (grating>=grating_min)*(grating<grating_max)
            return grating[mask], IWidth[mask]
            

pm_octree = pm3dgrid




# ==== TEST SETTINGS ===========================================================  
filename = 'M:/scratchbin/MW_supermuc/CF97M/E.B100N1/CF97TE_hdf5_plt_cnt_0160'

# ==== TEST ====================================================================    
if __name__ == '__main__':
    pc = 3.0856776e+18     # parsec (cm)
    ffile = plotfile(filename, memorize=False)
    ffile.learn(var_mhd)
    ffile.learn(var_ch5)
    ffile.learn(var_grid)
    fgrid = pm3dgrid(ffile)
    
    #dd = fgrid.coords(axis=2)/pc
    dd = ffile['temp']
    wd = ffile['numdens']
    ffile.close()

    ddgrid = datagrid(dd, fgrid, verbose=True) # NOTE: guard cell count ng=0 (OFF)

    axis = 2    
    offset = 1e-5*pc
    #Si = ddgrid.read_slice(axis, offset, interpolation='linear', verbose=True)
    #Mi = ddgrid.read_meanslice(axis, interpolation='linear', weight_data=wd,
    #                           verbose=True, xres=2852, yres=713, zres=713)
    
    #import cProfile
    #import pstats
    #print '#############################################################'
    #cProfile.run("Mi = ddgrid.read_meanslice(axis, interpolation='linear', verbose=True)", 'restats')
    #p = pstats.Stats('restats')
    #p.strip_dirs().sort_stats('time').print_stats(15)
    #p.strip_dirs().sort_stats('cumtime').print_stats(15)
