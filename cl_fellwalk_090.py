#!/usr/bin/env python
# -*- coding: utf-8 -*-
import numpy as np
import sys, os
from copy import deepcopy

import libmf.micflash as mflash

__author__ = "Michael Weis"
__version__ = "0.0.8.0"

verbose = False


# ==== HELPER ==================================================================
def lmap(*args, **kwargs):
    return list(map(*args, **kwargs))

def lzip(*args, **kwargs):
    return list(zip(*args, **kwargs))

def select(cluster):
    if cluster:
        return tuple(lmap(np.array, zip(*cluster)))
    else:
        #raise ValueError('Select received empty selection!')
        return tuple(4*[np.array([], dtype=np.int64),])

def select_cells(hdfdata, select_criteria):
    dens = hdfdata['dens']
    mask = (dens == dens)
    for criterion in select_criteria:
        data = hdfdata[criterion]
        datamin, datamax = select_criteria[criterion]
        mask *= (data == np.clip(data, datamin, datamax))
    return mask
    
def instaprint(*args):
    print([str(a) for a in args])
    sys.stdout.flush()
    
def verboseprint(*args):
    if verbose:
        instaprint(*args)

# ==============================================================================
# ==== FELLWALKER CLUMP FINDING ALGORITHM ======================================
# ==============================================================================
_fw_chunksize = 7e+7


# ==== FELLWALKER SANITY CHECKS ================================================

def _fw_check_clumps_disjoint(octree, clump_list):
    baseshape = octree.blk_shape()
    claccu = np.zeros(baseshape, dtype=int)
    clump_selectors = lmap(select, clump_list)
    for clump in clump_selectors:
        claccu[clump] += 1
    clcells = np.sum(claccu>0)
    cljoints = np.sum(claccu>1)
    print(f'Cell selection intersections: {cljoints}/{clcells}')
    return np.any(cljoints)
    
def _fw_check_jumps(JLT, cellID, height):
    # Prepare return format
    JLT_errormask = np.zeros(JLT.shape, dtype=bool)
    # Build height lookup table (cellID->height)
    HLU = np.zeros((cellID.max()+1), dtype=height.dtype)
    HLU[cellID] = height
    # Check height lookup table
    assert np.all(height==HLU[cellID])
    # Check link lookup table for: links to zero, descents, cyclic ascents
    ascents = (JLT[cellID]!=cellID)*(JLT[cellID]!=0)
    height_0 = height[ascents]
    cellID_0 = cellID[ascents]
    i = 0
    height_i = np.copy(height[ascents])
    cellID_i = np.copy(cellID[ascents])
    while not np.all(np.logical_or(height_i>height_0,cellID_i==0)):
        cellID_ii = JLT[cellID_i]
        i+= 1
        cellID_i = np.copy(cellID_ii)
        height_i = HLU[cellID_i]
        n_nonascents = np.sum(height_i<=height_0)
        print(f'#DEBUG: JLT cycle {i} ({n_nonascents} non-ascents)')
        # Check for cycles
        try:
            assert not np.any(cellID_i==cellID_0)
        except:
           raise ValueError('Jump lookup table is cyclic. This is not recoverable!')
        # Check for indirect links to zero
        try:
            assert not np.any(cellID_i==0)
        except:
            n_zerolinks = np.sum(cellID_i==0)
            print(f'WARNING: Found {n_zerolinks} links to zero (total).')
            zeroIDs = cellID_0[cellID_i==0]
            JLT_errormask[zeroIDs] = True
        # Check for descents
        n_descents = np.sum(height_i<height_0)
        print(f'#DEBUG: Descents: {n_descents}') 
        assert i<1000
    print('#DEBUG: JLT OK')
    return JLT_errormask


# ==== FELLWALKER HELPER =======================================================
def _fw_gbld_stepdown(data, ng_in, ng_out):
    dg = int(.5 +ng_in -ng_out)
    if dg==0:
        return data
    elif dg>0:
        return data[:,dg:-dg,dg:-dg,dg:-dg]
    else:
        raise ValueError('Tried to enlarge guard area by stepdown.')

def _fw_XYZ_gbld(octree, ng):
    # Get guarded blockcell center position data;
    # wrap guarded blockcell center position data around sim domain boundaries
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    X_gbld = mflash.modclip(octree.coords(ng=ng, axis=0), xmin, xmax)
    Y_gbld = mflash.modclip(octree.coords(ng=ng, axis=1), ymin, ymax)
    Z_gbld = mflash.modclip(octree.coords(ng=ng, axis=2), zmin, zmax)
    return X_gbld, Y_gbld, Z_gbld
    
def _fw_block_gbld(octree, ng, XYZ_gbld=None):
    if XYZ_gbld is None:
        X_gbld, Y_gbld, Z_gbld = _fw_XYZ_gbld(octree, ng)
    else:
        X_gbld, Y_gbld, Z_gbld = XYZ_gbld
    Bl_gbld = np.arange(len(X_gbld))[...,None,None,None] *np.ones(X_gbld.shape, dtype=int)
    select_guard = np.ones(X_gbld.shape, dtype=bool)
    select_guard[...,ng:-ng,ng:-ng,ng:-ng] = False
    gX = X_gbld[select_guard]
    gY = Y_gbld[select_guard]
    gZ = Z_gbld[select_guard]
    Bl_gbld[select_guard] = octree.findblock(gX, gY, gZ)
    return Bl_gbld
    
def _fw_blocksearch_piecewise(func):
    def func_wrapper(a, b):
        assert a.shape == b.shape
        pieces = max(1,int((a.size*27.)/_fw_chunksize +.5))
        ra_0, rb_0 = func(a[0::pieces], b[0::pieces])
        ra = np.empty(a.shape[0:1]+ra_0.shape[1:])
        rb = np.empty(b.shape[0:1]+rb_0.shape[1:])
        ra[0::pieces] = ra_0
        rb[0::pieces] = rb_0
        for i in range(1,pieces):
            print(('*'), end=' ') #XXX
            sys.stdout.flush()
            a_i, b_i = a[i::pieces], b[i::pieces]
            ra[i::pieces], rb[i::pieces] = func(a_i, b_i)
        return ra, rb
    return func_wrapper
    
def _fw_cellmask(clump_list, datashape):
    cellmask = np.zeros(datashape, dtype=bool)
    clump_where = select(clump_list)
    cellmask[clump_where] = True
    return cellmask


# ==== FELLWALKER CLUMP MERGER =================================================    
def _fw_getneigh(cellmask, octree, XYZ_1gbld, Bl_1gbld):
    print(('*'), end=' ') #XXX
    sys.stdout.flush()

    verboseprint('### Create guarded (ng=1) block data boolean mask marking ')
    verboseprint('### the direct per-block neighborhood of the cellmask.')
    nbl, nk, nj, ni = cellmask.shape
    neighmask_gbld = np.zeros([nbl,nk+2,nj+2,ni+2], dtype=bool)
    neighmask_gbld[:, 1:1+nk, 1:1+nj, 1:1+ni] = cellmask
    neighmask_gbld[:, 0:0+nk, 1:1+nj, 1:1+ni] += neighmask_gbld[:, 1:1+nk, 1:1+nj, 1:1+ni]
    neighmask_gbld[:, 2:2+nk, 1:1+nj, 1:1+ni] += neighmask_gbld[:, 1:1+nk, 1:1+nj, 1:1+ni]
    neighmask_gbld[:, :,      0:0+nj, 1:1+ni] += neighmask_gbld[:, :,      1:1+nj, 1:1+ni]
    neighmask_gbld[:, :,      2:2+nj, 1:1+ni] += neighmask_gbld[:, :,      1:1+nj, 1:1+ni]
    neighmask_gbld[:, :,      :,      0:0+ni] += neighmask_gbld[:, :,      :,      1:1+ni]
    neighmask_gbld[:, :,      :,      2:2+ni] += neighmask_gbld[:, :,      :,      1:1+ni]
    
    verboseprint('## Remove actual data cells from neighborhood mask')
    neighmask_gbld[:, 1:1+nk, 1:1+nj, 1:1+ni] *= np.logical_not(cellmask)

    verboseprint('## Resolve guarded area of the neighborhood mask to the data area')
    verboseprint('# Creake a mask to select the guardcell area of the neighborhood mask')
    select_gn = np.array(neighmask_gbld)
    select_gn[..., 1:-1, 1:-1, 1:-1] = False
    verboseprint('# Resolve guard cell positions to blocks')
    X_gbld, Y_gbld, Z_gbld = XYZ_1gbld
    gnX = X_gbld[select_gn]
    gnY = Y_gbld[select_gn]
    gnZ = Z_gbld[select_gn]
    gnBl = Bl_1gbld[select_gn]

    verboseprint('## Compare refinement level of found block to local refinement level')
    blk_rlevel = octree.blk_rlevel()
    rlevel_gbld = blk_rlevel[:,None,None,None]*np.ones(neighmask_gbld.shape)
    GcRef = rlevel_gbld[select_gn].astype(int)
    BlRef = blk_rlevel[gnBl].astype(int)
    #assert np.all(np.isin(GcRef-BlRef,[-1,0,1]))

    verboseprint('## If guardcell at least as resolved as data cell, resolve selection directly')
    direct = GcRef>=BlRef
    Arg = (gnBl[direct], gnX[direct], gnY[direct], gnZ[direct])
    K, J, I = octree.findblockcell(*Arg)
    Bl = gnBl[direct]
    verboseprint('# Set indicated neighborhood mask cells to true')
    neighmask_gbld[:,1:1+nk,1:1+nj,1:1+ni][Bl,K,J,I] = True

    verboseprint('## If guardcell is underresolved, spread selection over underlying data cells')
    resolve = np.logical_not(direct)
    Arg = (gnBl[resolve], gnX[resolve], gnY[resolve], gnZ[resolve])
    verboseprint('# Get floor cell indices from positions')
    Kf, Jf, If = octree.findblockfc(*Arg)
    verboseprint('# Convert float indicies to floor indicies')
    Kfloor = Kf.astype(int)
    Jfloor = Jf.astype(int)
    Ifloor = If.astype(int)
    verboseprint('# Add extra dimension to floor index field iterating over')
    verboseprint('# all (floor,ceil)x(floor,ceil)x(floor,ceil) index combinations')
    K  = Kfloor[...,None] +np.array([0,1,0,1,0,1,0,1])
    J  = Jfloor[...,None] +np.array([0,0,1,1,0,0,1,1])
    I  = Ifloor[...,None] +np.array([0,0,0,0,1,1,1,1])
    Bl = gnBl[resolve][...,None] *np.ones(8, dtype=int)
    verboseprint('# Set indicated neighborhood mask cells to true')
    neighmask_gbld[:,1:1+nk,1:1+nj,1:1+ni][Bl,K,J,I] = True    

    verboseprint('## Remove actual data cells from neighborhood mask')
    neighmask_gbld[:, 1:1+nk, 1:1+nj, 1:1+ni] *= np.logical_not(cellmask)
    #
    return neighmask_gbld[:, 1:1+nk, 1:1+nj, 1:1+ni]
    
def _fw_clump_boundary(clump, octree, XYZ_1gbld, Bl_1gbld):
    ''' Get list of cells comprising the outer boundary of a cell ensemble '''
    datashape = octree.blk_shape()
    # Extract list of cells directly adjacent to a clump without being
    # included in it.
    clmask = _fw_cellmask(clump, datashape)
    boundmask = _fw_getneigh(clmask, octree, XYZ_1gbld, Bl_1gbld)
    assert not np.any(clmask*boundmask)
    boundwhere = np.where(boundmask)
    boundlist = set(zip(*boundwhere))
    return boundlist
    
def _fw_cell_peak(cells, height):
    ''' Get peak height value of a (possibly empty) cell ensemble '''
    if cells:
        cellselect = select(cells)
        heights = height[cellselect]
        col = np.amax(heights)
    else:
        col = -np.inf
    return col


def fellwalk_merge(clumplists, height, octree, MinDip):

    clumps = lmap(set, clumplists)
    clIDs = [set([ID,]) for ID in range(1, len(clumps)+1)]

    instaprint('-> Collecting clump peak heights...')
    clpeaks = lmap(lambda clump: _fw_cell_peak(clump, height), clumps)
    
    instaprint('-> Generating guard area block lookup table...')
    XYZ_1gbld = _fw_XYZ_gbld(octree, ng=1)
    Bl_1gbld  = _fw_block_gbld(octree, ng=1, XYZ_gbld=XYZ_1gbld)

    instaprint('-> Identifying clump boundaries...')
    nbl, nk, nj, ni = height.shape
    args = (octree, XYZ_1gbld, Bl_1gbld)
    clbounds = lmap(lambda clump: _fw_clump_boundary(clump, *args), clumps)
    _fw_check_clumps_disjoint(octree, clbounds)
    
    # Check data
    assert isinstance(clumps, list)
    assert isinstance(clbounds, list)
    assert isinstance(clIDs, list)
    assert isinstance(clpeaks, list)
    assert len(clumps)==len(clbounds)
    assert len(clumps)==len(clIDs)
    assert len(clumps)==len(clpeaks)
        
    instaprint('-> Considering clump merger...')
    
    cycles = len(clumps)
    while cycles:
        # Output status information:
        n_cycles = len(clumps)
        i_cycle = n_cycles -cycles +1
        instaprint('\r%i/%i         '%(i_cycle,n_cycles)),
        # Pop a new central clump and unpack its information
        clc_cells = clumps.pop(0)
        clc_bound = clbounds.pop(0)
        clc_IDs   = clIDs.pop(0)
        clc_peak  = clpeaks.pop(0)
        # Check if there is a second clump left to merge to
        if not clumps:
            break
        # Get highest peaks of the of the valleys, defined as
        # outer boundaries of the central clump and part of the adjacent clump
        valleys = lmap(lambda clump: clump&clc_bound, clumps)
        cols = lmap(lambda valley: _fw_cell_peak(valley, height), valleys)
        # Get the index of the clump providing the valley with the highest peak
        i_col = np.argmax(cols)
        # Check whether the central clump should be merged to that clump
        col = cols[i_col]
        is_merger = (clc_peak < (col+MinDip))
        # Print Debug information
        #_fw_check_clumps_disjoint(octree, valleys)
        #print '#DEBUG: len(clc_bound) :', len(clc_bound)
        #print '#DEBUG: len(clumps)    :', map(len, clumps)
        #print '#DEBUG: len(valleys)   :', map(len, valleys)
        #print '#DEBUG: cols           :', cols
        #print '#DEBUG: i_col          : %.5g'%i_col
        #print '#DEBUG: col            : %.5g'%col
        #print '#DEBUG: clc_peak       : %.5g'%clc_peak
        #print '#DEBUG: MinDip         : %.5g'%MinDip
        #print '#DEBUG: is_merger      :', is_merger
        #
        if is_merger:
            merger_cells = clc_cells | clumps.pop(i_col)
            merger_bound = (clc_bound | clbounds.pop(i_col)) -merger_cells
            merger_IDs   = clc_IDs | clIDs.pop(i_col)
            merger_peak  = max(clc_peak,clpeaks.pop(i_col))
            clumps.append(merger_cells)
            clbounds.append(merger_bound)
            clIDs.append(merger_IDs)
            clpeaks.append(merger_peak)
            cycles = len(clumps)
            print()
            print(str(clc_IDs)+'->'+str(merger_IDs-clc_IDs))
        else:
            clumps.append(clc_cells)
            clbounds.append(clc_bound)
            clIDs.append(clc_IDs)
            clpeaks.append(clc_peak)
            cycles -=1
    print()
    n_rawcl = len(clumplists)
    n_clumps = len(clumps)
    instaprint(f'...merged {n_rawcl} raw clumps into {n_clumps} clump candidates.')
    
    return deepcopy(clumps)


# ==== FELLWALKER CLUMP IDENTIFICATION =========================================

def _fw_stack_neighborhood_indices(blshape):
    # Create lists of all dk,dj,di combinations in [0,-1,1]x[0,-1,1]x[0,-1,1]
    # to do neighborhood searches by iterating over a single additional dimension.
    dd = [0,-1,1]
    offset_k, offset_j, offset_i = lmap(np.ravel,np.meshgrid(dd,dd,dd))
    # Create k,j,i index field with additional dk,dj,di offset dimension
    nbl, nk, nj, ni = blshape
    K, J, I = lmap(lambda l: range(1,l-1), (nk,nj,ni))
    k__, _j_, __i = np.meshgrid(K, J, I, indexing='ij')
    k__o = k__[...,None] +offset_k
    _j_o = _j_[...,None] +offset_j
    __io = __i[...,None] +offset_i
    return k__o, _j_o, __io

@_fw_blocksearch_piecewise
def _fw_blocksearch_maxneigh(height_g, cellid_g):
    ''' SEARCH BLOCK-WISE FOR MAXIMA IN 1-NEIGHBORHOOD OF CELLS '''
    # Check input
    assert height_g.shape == cellid_g.shape
    # Append extra dimension stacking the celldata panned through the neighborhood
    K, J, I = _fw_stack_neighborhood_indices(height_g.shape)
    height_gn = height_g[:,K,J,I]
    cellid_gn = cellid_g[:,K,J,I]
    # Search for o-argument of height maxima in 1-neighborhood
    o = np.argmax(height_gn, axis=-1)
    # Get height and cellid from found maximum by inserting found o-argument
    bl, k, j, i = np.meshgrid(*lmap(range, height_gn.shape[:-1]), indexing='ij')
    heightmax_height_gn = height_gn[bl,k,j,i,o]
    heightmax_cellid_gn = cellid_gn[bl,k,j,i,o]
    return heightmax_height_gn, heightmax_cellid_gn
    
def _fw_resolve_guard_jumps(jumpID, cellID, cellid_pos, height, octree):
    ''' Resolve links to guard cells (jumpID<0) to cellids of respective blocks '''
    # Create selector for guard cell link assignments from link assignment array
    gclink = jumpID<0
    ## Prepare fields holding results
    jumpID_gcK = -1*np.ones(jumpID[gclink].shape, dtype=int)
    jumpID_gcJ = -1*np.ones(jumpID[gclink].shape, dtype=int)
    jumpID_gcI = -1*np.ones(jumpID[gclink].shape, dtype=int)
    # Resolve link assignments linking guard cells (jumpID<0) to X,Y,Z cell positions
    jumpID_gcID = -jumpID[gclink]
    jumpID_gcX, jumpID_gcY, jumpID_gcZ = cellid_pos[jumpID_gcID].T
    # Resolve positions to blocks
    jumpID_gcBl = octree.findblock(jumpID_gcX, jumpID_gcY, jumpID_gcZ)
    # Compare refinement level of found block to local refinement level
    
    GcRef = octree.cell_rlevel()[gclink].astype(int)
    BlRef = octree.blk_rlevel()[jumpID_gcBl].astype(int)
    assert GcRef.shape == BlRef.shape
    # As neighboring blocks are must differ by at most one refinement level,
    # and the link assignments must always hit neighboring blocks,
    # the refinement deficit must be in [-1,0,1] for each link assignment.
    # If not, something went seriously wrong!
    assert np.all(np.isin(GcRef-BlRef, [-1,0,1]))
    
    ## If linked guardcell is as refined as target block: resolve directly
    rMatch = GcRef==BlRef
    Arg = (jumpID_gcBl[rMatch], jumpID_gcX[rMatch], jumpID_gcY[rMatch], jumpID_gcZ[rMatch])
    jumpID_gcK[rMatch], jumpID_gcJ[rMatch], jumpID_gcI[rMatch] = octree.findblockcell(*Arg)

    ## If linked guardcell is finer refined than target block: resolve directly
    rOver = GcRef>BlRef
    Arg = (jumpID_gcBl[rOver], jumpID_gcX[rOver], jumpID_gcY[rOver], jumpID_gcZ[rOver])
    jumpID_gcK[rOver], jumpID_gcJ[rOver], jumpID_gcI[rOver] = octree.findblockcell(*Arg)
    
    ## If linked guardcell does not resolve data cell:
    ## search height maximum cell around position
    rUnder = GcRef<BlRef
    Arg = (jumpID_gcBl[rUnder], jumpID_gcX[rUnder], jumpID_gcY[rUnder], jumpID_gcZ[rUnder])
    # Get floor cell indices from positions
    Kf, Jf, If = octree.findblockfc(*Arg)
    # Convert float indicies to floor indicies
    Kfloor = Kf.astype(int)
    Jfloor = Jf.astype(int)
    Ifloor = If.astype(int)
    # Add extra dimension to floor index field iterating over
    # all (floor,ceil)x(floor,ceil)x(floor,ceil) index combinations
    dK, dJ, dI = lmap(np.ravel, np.meshgrid([0,1],[0,1],[0,1]))
    K = Kfloor[...,None] +dK
    J = Jfloor[...,None] +dJ
    I = Ifloor[...,None] +dI
    Bl = jumpID_gcBl[rUnder][...,None]
    H = height[Bl,K,J,I]
    # Search height (floor,ceil)x(floor,ceil)x(floor,ceil) cell index combinations,
    # (for which the iterating dimension was appended in the last step)
    # for maximum
    O = np.argmax(H, axis=-1)
    # Use O-index of height maxima to extract the corresponding cellids
    M = np.meshgrid(*lmap(range, Kf.shape), indexing='ij')
    try:
        jumpID_gcK[rUnder] = K[tuple(M)+(O,)]
        jumpID_gcJ[rUnder] = J[tuple(M)+(O,)]
        jumpID_gcI[rUnder] = I[tuple(M)+(O,)]
    except:
        print('DEBUG - H:')
        print(H)
        print('DEBUG - Kf.shape:')
        print(Kf.shape)
        print('DEBUG - tuple(M)+(O,):')
        print( tuple(M)+(O,))
        print('DEBUG - rUnder:')
        print(rUnder)
        raise
    
    # The guard cells are now resolved into cell indices of actual data cells
    # of corresponding blocks. Now read the cellids from those.
    jumpID[gclink] = cellID[jumpID_gcBl,jumpID_gcK,jumpID_gcJ,jumpID_gcI]
    
    # For overresolved guard areas, the interpolated guard cell height
    # value may be higher than the underlying data block value,
    # if adjacent data blocks directed to the guard cell lean direction
    # have values exceeding that of the data block containing the guard cell.
    # In some cases, this will push the hight after guard cell resolution below
    # the hight value of the data block that from where the jump started,
    # transforming the ascend jump into a descend jump.
    # As the next jump would then be guaranteed to hit at least the hight value
    # of the adjacent cell causing the problem, this poses no problem for the
    # resulting clump structure;
    # WHILE NOT AN ACTUAL ISSUE, THIS WILL HOWEVER TRIP JUMP CORRECTNESS CHECKS,
    # SOURCING AN INSANELY HARD TO FIND ERROR!
    # To remedy this issue, the jumpID result will now be iterated
    # in advance by one ascension jump.
    jumpID[gclink] = jumpID[jumpID_gcBl,jumpID_gcK,jumpID_gcJ,jumpID_gcI]
    
    # Check guard jump resolution result
    assert np.all(jumpID[gclink]!=-1) # Check for disregarded guard jumps
    assert np.all(jumpID[gclink]!=0) # Check for resolutions to zero
    assert np.all(jumpID[gclink]>0) # Check against resolution to guard areas
    #
    return jumpID

def _fw_traverse_jumps(JLT, cellID):
    # Check CellID properties
    assert np.all(cellID>0)
    assert len(cellID.ravel()) == len(set(cellID.ravel())) # CellID uniqueness    
    # Check lookup table properties
    assert not np.any(JLT<0) # all guard cell links resolved
    jump1ID = JLT[cellID]
    selflinked = jump1ID==cellID
    try:
        assert np.any(selflinked) # Check for the existence of at least one peak
    except:
        instaprint('WARNING: Found no peaks!')
    # Walk ascension links by iterating the link lookup table
    # on CellIDs until getting stationary
    IDWalk = np.copy(cellID)
    ascents = IDWalk==IDWalk # = selflinked
    while np.any(ascents):
        print(('*'), end=' ') #XXX
        sys.stdout.flush()
        IDWalk_asc_old = IDWalk[ascents]
        IDWalk_asc_new = JLT[IDWalk_asc_old]
        IDWalk[ascents] = IDWalk_asc_new
        ascents[ascents] *= np.logical_not(IDWalk_asc_new==IDWalk_asc_old)
    print()
    PeakIDs = set(IDWalk[IDWalk!=0].ravel())
    # Check walk results
    assert np.all(IDWalk==JLT[IDWalk]) # Check for overlooked iterations
    #assert not np.any(IDWalk[jump1ID!=0]==0) # Check for indirect walks to zero
    # if parts of the domain are masked out for other reasons then height,
    # then indirect walks uphill into the masked out area are to be expected.
    assert PeakIDs == set(cellID[selflinked]) # Check for peak consistency    
    # Collect ascents
    rawclumps = lmap(lambda ID: lzip(*np.where(IDWalk==ID)), PeakIDs)
    #
    return rawclumps


def fellwalk_identify(height, octree, ng, domain_mask=None):
    
    instaprint('-> Generating guarded data map...')
    # Generate guarded block data height array
    height_grid = mflash.datagrid(height, octree, ng=ng)
    height_gbld = height_grid.gbld
    nbl, nk, nj, ni = height.shape
    assert height_gbld.shape == (nbl, nk+2*ng, nj+2*ng, ni+2*ng)
    # Get a boolean gbld shaped field marking the guard cells
    select_guard = np.ones(height_gbld.shape, dtype=bool)
    select_guard[...,ng:-ng,ng:-ng,ng:-ng] = False
    select_data = np.logical_not(select_guard)
    
    #instaprint('-> Generating guarded data cell IDs...')
    # Generate array of unique indices in the height_gbld format,
    # <0 for guard cells, >0 for payload data cells
    gbld_shape = height_gbld.shape
    gbld_cellid = np.zeros(gbld_shape, dtype=int)
    gbld_cellid[select_data]  = +1 +np.arange(np.sum(select_data))
    gbld_cellid[select_guard] = -1 -np.arange(np.sum(select_guard))
    cellID = gbld_cellid[...,ng:-ng,ng:-ng,ng:-ng]
        
    jumpID = np.copy(cellID)
    heightmax_gni, linkedid_gni = height_gbld, gbld_cellid
    ngnj = ng
    for n_jumps in range(1, ng+1):
        instaprint('-> Searching ascension jumps in %i cell radius...'%n_jumps)
        # Search height maxima in 2-Neighborhood of cells
        # by recursively searching 1-Neighborhood of 1-Neighborhood results
        heightmax_gnj, linkedid_gnj = _fw_blocksearch_maxneigh(heightmax_gni, linkedid_gni)
        # Select core data cells from (necessarily) guarded linkid field
        ngnj -= 1
        assert ngnj >= 0
        linkedid_nj = linkedid_gnj[:, ngnj:ngnj+nk, ngnj:ngnj+nj, ngnj:ngnj+ni]
        assert linkedid_nj.shape == jumpID.shape
        # Copy 2-neighbor height maximum link to link assignment field
        # if link assignment points to self (read: is not zero -
        # - and not pointing to 1-neighbor)
        is_selflinked = (jumpID==cellID)
        n_ascents = np.sum(jumpID[is_selflinked] != linkedid_nj[is_selflinked])
        jumpID[is_selflinked] = linkedid_nj[is_selflinked]
        # Output result
        n_peaks = np.sum((jumpID==cellID)*(height>0.))
        instaprint('...found %i jumps, %i peaks left.'%(n_ascents,n_peaks))
        # Copy data for next turn
        ngni = ngnj
        heightmax_gni = np.copy(heightmax_gnj)
        linkedid_gni = np.copy(linkedid_gnj)
        
    #instaprint('-> Generating guard cell lookup table...')
    # Get guarded blockcell center position data;
    X_gbld, Y_gbld, Z_gbld = _fw_XYZ_gbld(octree, ng=ng)
    # Build an array to resolve (negative) cellids to cell positions,
    # format:  cellid_pos[-ID,axis], usage: X,Y,Z = cellid_pos[-ID]
    # this will be needed later to resolve links into the block guard cell area
    # (outside the actual block boundary) to links to other blocks
    XYZ = np.vstack((X_gbld[select_guard], Y_gbld[select_guard], Z_gbld[select_guard]))
    gcID = -gbld_cellid[select_guard]
    assert np.all(gcID>0)
    cellid_pos = np.zeros((gcID.max()+1,3))
    cellid_pos[gcID] = XYZ.T
    
    instaprint('-> Resolving guard area jumps...')
    jumpIDr = _fw_resolve_guard_jumps(jumpID, cellID, cellid_pos, height, octree)
    
    #instaprint('-> Generating ascension jump lookup table...')
    if domain_mask is not None:
        # Disregard jump destinations located outside the allowed domain
        # If used correctly, the domain mask MUST imply the height criterion
        disallowed_mask = np.invert(domain_mask)
        jumpIDr[disallowed_mask] = 0
        assert np.all(jumpIDr[height<=0.] == 0)
    else:
        # Disregard jump destinations not meeting the height threshold
        jumpIDr[height<=0.] = 0
    
    # Build jump lookup table (cellID->destinationID)
    JLT = np.zeros((cellID.max()+1), dtype=jumpIDr.dtype)
    JLT[cellID] = jumpIDr
    # Check jump lookup table
    assert np.all(jumpIDr==JLT[cellID])
    assert JLT[0] == 0
    
    instaprint('-> Traversing jump structure...')
    # Check jump lookup table
    JLT_errormask =_fw_check_jumps(JLT, cellID, height)
    JLT[JLT_errormask] = 0
    # Execute jump structure traversal
    rawclumps = _fw_traverse_jumps(JLT, cellID)
    # Filter isolated peaks
    rawclumps_f = [cl for cl in rawclumps if len(cl)>1]
    
    instaprint('...found %i raw clumps.'%len(rawclumps_f))
    return rawclumps_f
    

# ==============================================================================
# ==== FELLWALKER INTERFACE ====================================================
# ==============================================================================

def fellwalk(hdfdata, octree, var='gpot', threshold=0., direction='up',
        MaxJump=4, MinDipStd=1., MinPts=100, criteria={}, mask=None,
        preselect='IGNORE'):
        
    # Find out which cells are meeting the var-threshold and additional criteria
    cells_meeting_criteria = select_cells(hdfdata, criteria)
    
    if mask is not None:
        cells_meeting_criteria *= mask
    
    vardata = hdfdata[var]
    
    # Make up threshold, consistent to additional criteria, if None given,
    # to cut data accordingly at height 0
    if threshold is None:
        select_limit = np.max if direction=='down' else np.min
        try:
            threshold = select_limit(vardata[cells_meeting_criteria])
        except ValueError as EE:
            threshold = select_limit(vardata)
            print(f'Warning: {EE}')

    # Scale block data field named var to a height field:
    # 0. is the cutoff value, clump linking along increasing height to height max
    if direction=='up':
        height = vardata-threshold
    elif direction=='down':
        height = threshold-vardata
    else:
        raise ValueError('Direction "%s" not recognized.'%direction)
        
    allowed_cells = cells_meeting_criteria * (height>0.)

    # Check if there are any cells left to put into clumps;
    # bypass algorithm if not
    Pts = np.sum(allowed_cells)
    if Pts<MinPts:
        instaprint('-> Not enough cells beyond threshold!')
        return list()
    else:
        instaprint('-> Considering %i cells'%Pts)
        
    # Check MaxJump parameter
    nbl, nk, nj, ni = height.shape
    ng_max = int(np.min(height.shape[1:])/2)
    assert MaxJump in range(1, ng_max+1)
    ng = MaxJump
        
    rawclumps = fellwalk_identify(height, octree, ng, domain_mask=allowed_cells)
    
    _fw_check_clumps_disjoint(octree, rawclumps)
    
    # Process MinDip parameter
    height_std = np.std(height)
    MinDip = MinDipStd*height_std
    
    if MinDip > 0.01*height_std:
        clumps = fellwalk_merge(rawclumps, height, octree, MinDip)
    else :
        clumps = rawclumps
    
    _fw_check_clumps_disjoint(octree, clumps)

    # Filter peaks (MinPts)
    clumps_f = [cl for cl in clumps if len(cl)>=MinPts]
    
    return deepcopy(clumps_f)


def fellwalk_clist(pre_clist, hdfdata, octree, **kwargs):
    clumps_f = list()
    
    blockshape = hdfdata['dens'].shape
    cluster_selectors = lmap(select, pre_clist)
    for selector in cluster_selectors:
        mask = np.zeros(blockshape, dtype=np.bool)
        mask[selector] = True
        clumps_cl = fellwalk(hdfdata, octree, mask=mask, **kwargs)
        clumps_f.extend(clumps_cl)

    return clumps_f

def fellwalk_clist_fused(pre_clist, hdfdata, octree, **kwargs):
    blockshape = hdfdata['dens'].shape
    cluster_selectors = lmap(select, pre_clist)
    mask = np.zeros(blockshape, dtype=np.bool)
    for selector in cluster_selectors:
        mask[selector] = True
    clumps_f = fellwalk(hdfdata, octree, mask=mask, **kwargs)

    return clumps_f

# ==============================================================================
# === MAIN =====================================================================
# ==============================================================================
if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("-i", "--infile", help="input plotfile", type=str, required=True)
    parser.add_argument("-o", "--outfile", help="output filename", type=str)
    parser.add_argument("-v", "--var", help="plotfile/micflash variable name", type=str, required=True)
    parser.add_argument("-t", "--threshold", help="threshold value for variable", type=float, required=True)
    parser.add_argument("-d", "--downwards", help="walk downwards from threshold", action="store_true")
    parser.add_argument("-k", "--MaxJump", help="maximum walk search distance", type=int, default=4)
    parser.add_argument("--MinDip", help="minimum positive depth of valley between core peaks", type=float)
    parser.add_argument("-n", "--MinPts", help="minimum number of core cells", type=int, default=100)
    
    args = parser.parse_args()

    plotfile = args.infile
    outfile = args.outfile if args.outfile else plotfile +'_fwcl.txt'
    direction = 'down' if args.downwards else 'up'
    MinDip = args.MinDip if args.MinDip else None
    
    hdfdata = mflash.plotfile(plotfile, memorize=False)
    hdfdata.learn(mflash.var_mhd)
    hdfdata.learn(mflash.var_grid)
    hdfdata.learn(mflash.var_ch5)
    octree = mflash.pm_octree(hdfdata)
    
    cluster_list = fellwalk(hdfdata, octree, args.var,
        threshold=args.threshold, direction=direction, MaxJump=args.MaxJump,
        MinDip=MinDip, MinPts=args.MinPts)
    
    import csv
    with open(outfile, 'wb') as f:
        wr = csv.writer(f, quoting=csv.QUOTE_ALL)
        wr.writerow(cluster_list)

