# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as la
import sys, subprocess, os
import matplotlib.pylab as plt
from copy import deepcopy
from scipy.signal import convolve

import libmf.micflash as mflash

import cl_fellwalk_090 as fw

__author__ = "Michael Weis"
__version__ = "0.1.4.2"

# ==== CONSTANTS ===============================================================
from constants import *

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

def modclip(a, a_min, a_max):
    return a_min + np.mod(a-a_min, a_max-a_min)    

def mask_cells(blockdata, select_criteria):
    dens = blockdata['dens']
    mask = (dens == dens)
    for criterion in select_criteria:
        data = blockdata[criterion]
        datamin, datamax = select_criteria[criterion]
        mask *= (data == np.clip(data, datamin, datamax))
    return mask
    
def GetMask(baseshape, clist):
    cl_where = select(clist)
    mask = np.zeros(baseshape, dtype=bool)
    mask[cl_where] = True
    return mask


# ==============================================================================
# ==== ISOCONTOUR NEIGHBORS OF NEIGHBORS =======================================
# ==============================================================================

def GetNeighMask(octree, mask):
    bs = mask.shape
    try:
        assert np.any(mask)
    except AssertionError:
        print('W! GetNeighMask: Input mask empty.')
        return np.zeros_like(mask)
    BC = np.where(mask)
    bcset = set(zip(*BC))
    neighset = octree.findneigh(*lmap(np.array, lzip(*bcset)), combine=True)
    neighmask = GetMask(bs, neighset)
    return neighmask
    
    
def GetMaxima(height, octree, mask, nneigh=4):
    """
    Get local maxima in a radius of nneigh cells.
    """
    # Decide how many layers of guard cells should be used,
    # which also determines the search radius.
    # This needs to be at least 1 layer to compare any point against,
    # and is usually limited to 4 layers (so that the guard cell layer does
    # not extend beyond the neighbouring blocks)
    ng = np.clip(int(nneigh+.5),1,4)
    # Reconstruct a number of guard cell layers according to nneigh.
    # With this we will be able to do the search for each block separately.
    height_grid = mflash.datagrid(height, octree, ng=ng)
    height_gbld = height_grid.gbld
    nbl, nk, nj, ni = height.shape
    # Get a boolean gbld shaped field marking the guard cells
    select_guard = np.ones(height_gbld.shape, dtype=bool)
    select_guard[...,ng:-ng,ng:-ng,ng:-ng] = False
    select_data = np.logical_not(select_guard)
    # Generate array of unique indicies for the height_gbld field,
    # with <0 for guard cells, >0 for payload data cells
    gbld_shape = height_gbld.shape
    gbld_cellid = np.zeros(gbld_shape, dtype=np.int)
    gbld_cellid[select_data]  = +1 +np.arange(np.sum(select_data))
    gbld_cellid[select_guard] = -1 -np.arange(np.sum(select_guard))
    cellID = gbld_cellid[...,ng:-ng,ng:-ng,ng:-ng]
    # Use the ng-radius jump finder algorithm from the fellwalk implementation
    jumpID = np.copy(cellID)
    heightmax_gni, linkedid_gni = height_gbld, gbld_cellid
    ngnj = ng
    for n_jumps in range(1, ng+1):
        print(f'-> Searching ascension jumps in {n_jumps} cell radius...')
        sys.stdout.flush()
        # Search height maxima in 2-Neighborhood of cells
        # by recursively searching 1-Neighborhood of 1-Neighborhood results
        heightmax_gnj, linkedid_gnj = fw._fw_blocksearch_maxneigh(heightmax_gni, linkedid_gni)
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
        n_peaks = np.sum((jumpID==cellID)*np.isfinite(height))
        print(f'...found {n_ascents} jumps, {n_peaks} peaks left.')
        sys.stdout.flush()
        # Copy data for next turn
        ngni = ngnj
        heightmax_gni = np.copy(heightmax_gnj)
        linkedid_gni = np.copy(linkedid_gnj)
    # Return the self-linked cells from the jump finder,
    # as those are the cells from which no local ascent inside the radius is
    # possible (i.e. they are the lokal maxima)
    maxmask = (jumpID==cellID)*np.isfinite(height)
    #
    return maxmask
    
def GetMinima(potential, *args, **kwargs):
    return GetMaxima(-1.*potential, *args, **kwargs)


def isocont_non(blockdata, octree, mask, var=None, verbose=True, **kwargs):
    """
    Select those volumes that enclose minima
    (exactly one minimum for each contour) of the given field
    with the largest possible isocontour of that field that is completely
    contained inside the selected volume (marked by the mask).
    """
    #
    MinPts_param = kwargs.pop("MinPts", 30)
    MinPts = max(2, int(MinPts_param))
    #
    if var is None:
        raise RuntimeError("ISOS: var parameter not provided")
    else:
        fieldvar = var
    #
    direction = kwargs.pop("direction", "up")
    if direction == "down":
        potential = -1.*blockdata[fieldvar]
    else:
        potential = blockdata[fieldvar]
    #
    nneigh_param = kwargs.pop("nneigh", 4)
    nneigh = max(1, int(nneigh_param))
    #
    dperc_param = kwargs.pop("dperc", 1.)
    dperc = min(max(0.01, dperc_param), 10.)

    #
    if np.sum(mask)<MinPts:
        return list()
    #
    if verbose:
        print(f'Detecting minima ({nneigh} cell radius)...')
        sys.stdout.flush()
    #
    m_minima = GetMinima(potential, octree, mask, nneigh=nneigh)
    # Only work minima inside the mask-selected area
    m_minima *= mask
    #
    if verbose:
        print(f'Detected {np.sum(m_minima)} Minima.')
        sys.stdout.flush()
    #
    if not np.any(m_minima):
        return list()
    #
    Neighbours = lambda mask: GetNeighMask(octree, mask)
    #
    object_list = list()
    #
    for w_seed in zip(*np.where(m_minima)):
    
        m_seed = np.zeros_like(m_minima, dtype=bool)
        m_seed[w_seed] = True

        # Limit the contour around the seed:
        # 1. The contour must not exceed the area given by the input mask
        m_excluded = np.logical_not(mask)
        # 2. The contour must not enclose any other minimum
        m_foreignseeds = m_minima*np.logical_not(m_seed)
        # Create a mask for the area that the object inside the contour is
        # not allowed to touch:
        m_forbidden = m_excluded | m_foreignseeds

        seed_potential = potential[w_seed]
        percsample = potential[(potential>seed_potential)*mask]
        isocont_percentile = 0.
        isocont_limit = np.percentile(percsample, isocont_percentile)
        m_contour = (potential<isocont_limit)
        
        m_members = np.zeros_like(m_seed)
        m_recruits = np.zeros_like(m_seed)
        m_siding = np.zeros_like(m_seed)
        
        # Initialize state with seed minimum
        m_recruits |= m_seed
        m_fouls = m_recruits * m_forbidden
        m_siding = Neighbours(m_seed)
        
        while not np.any(m_fouls):
            # Print a lifesign indicator
            print('*', end=' ')
            sys.stdout.flush()
            # No fouls, promote all recruits to members
            m_members |= m_recruits
            m_recruits = np.zeros_like(m_recruits)
            # Check against the isocontour level hitting the maximum global value.
            # Obviously, something went seriously wrong if this happens
            # (i.e. the isocontour containing the entire domain)
            if isocont_percentile > 100.:
                raise ValueError("ISOS: Runaway isocontour level")
            # Increase isocontour level by 1 percentile of the range seed..max
            isocont_percentile += dperc
            isocont_limit = np.percentile(percsample, isocont_percentile)
            m_contour = (potential<isocont_limit)            
            # Promote sidings to contacts up to the isocontour level
            m_contacts = m_siding * m_contour
            m_siding *= np.logical_not(m_contour)
            # Find new recruits by iterating contacts of contacts
            while np.any(m_contacts):
                if verbose:
                    print('.', end=' ')
                    sys.stdout.flush()
                m_recruits |= m_contacts
                m_hood = Neighbours(m_contacts)
                m_applicants = m_hood * np.logical_not(m_members|m_recruits)
                m_siding  |= m_applicants * np.logical_not(m_contour)
                m_contacts = m_applicants * m_contour
                m_fouls = m_recruits * m_forbidden
                if np.any(m_fouls):
                    break
            
        ## New recruits violate forbidden territory, as the isocontour level 
        ## went too high. However, the last level before was fine, hence:
        ## Consider the members as of the last valid isocontour level as a new object
        #
        if np.sum(m_members) < MinPts:
            print('X')
            if verbose:
                print(f'Rejected Isocontour Core: {np.sum(m_members)} Cells')
                print(f'  Contour Percentile: {isocont_percentile}')
                print(f'  Contour Potential: {isocont_limit}')
                print(f'  Seed    Potential: {seed_potential}')
            continue
        else:
            print('!')
            if verbose:
                print(f'Saving Isocontour Core: {np.sum(m_members)} Cells')
                print(f'  Contour Percentile: {isocont_percentile}')
                print(f'  Contour Potential: {isocont_limit}')
                print(f'  Seed    Potential: {seed_potential}')
        # Return object in set of [blk,k,j,i] cells format
        w_object = np.where(m_members)
        s_object = set(zip(*w_object))
        object_list.append(s_object)
        
    return deepcopy(object_list)
        
        
# ==============================================================================
# ==== HYBRID ISOSURFACE CATOUT ALGORITHM ======================================
# ==============================================================================

def isocont_cutout(blockdata, octree, mask, objlist_seed, var=None, verbose=True, **kwargs):
    """
    Select those volumes that enclose the given seed object
    with the largest possible isocontour of that field that does not
    intersect the other seed objects and is completely
    contained inside the selected volume (marked by the mask).
    """
    #
    if var is None:
        raise RuntimeError("ISOS: var parameter not provided")
    else:
        fieldvar = var
    #
    direction = kwargs.pop("direction", "up")
    if direction == "down":
        potential = -1.*blockdata[fieldvar]
    else:
        potential = blockdata[fieldvar]
    #
    #
    dperc_param = kwargs.pop("dperc", 1.)
    dperc = min(max(0.01, dperc_param), 10.)
    #    
    if not objlist_seed:
        return list()
    #
    baseshape = potential.shape
    seedmasks = [GetMask(baseshape, obj) for obj in objlist_seed]
    m_allobj  = np.logical_or.reduce(seedmasks)
    #
    Neighbours = lambda mask: GetNeighMask(octree, mask)
    #
    object_list = list()
    obj_iso_potentials = list()
    #    
    for o,obj in enumerate(objlist_seed):
        m_object = GetMask(baseshape, obj)
        # Limit the contour around the object:
        # 1. The contour must not exceed the area given by the input mask
        m_excluded = np.logical_not(mask)
        # 2. The contour must not touch any other object
        m_foreignobjs = m_allobj * np.logical_not(m_object)
        # Create a mask for the area that the object inside the contour is
        # not allowed to touch:
        m_forbidden = m_excluded | m_foreignobjs
        
        # Prepare masks holding the progress of the algorithm
        m_members  = np.zeros_like(m_object) # cells inside last isocontour
        m_siding   = np.zeros_like(m_object) # cells already in contact with object, but outside of last isocontour
        m_recruits = np.zeros_like(m_object) # cells inside next possible isocontour, if isocontour valid
        # Initialize state with seed minimum
        m_recruits |= m_object
        m_fouls = m_recruits * m_forbidden
        m_siding = Neighbours(m_object)

        obj_potentials = potential[m_object]
        obj_highpot = np.max(obj_potentials)
        obj_basepot = np.min(obj_potentials)

        percsample = potential[(potential>obj_highpot)*mask]
        isocont_percentile = 0.
        isocont_value = np.percentile(percsample, isocont_percentile)
        m_belowcont = (potential<isocont_value)
        m_abovecont = np.logical_not(m_belowcont)
        
        isocont_obj = None

        print()
        while not np.any(m_fouls):
            # Print a lifesign indicator
            print('*', end=' ')
            sys.stdout.flush()
            # No fouls in prior cycle, promote all recruits to members
            m_members |= m_recruits
            print(f'Isolevel {isocont_value}: Added {np.sum(m_recruits)} cells')
            # Purge recruits
            m_recruits = np.zeros_like(m_recruits)
            # Increase isocontour level by 1 percentile of the range seed..max
            isocont_value = np.percentile(percsample, isocont_percentile)
            m_belowcont = (potential<isocont_value)          
            # Separate sidings into contacts up to the isocontour level
            m_contacts = m_siding * m_belowcont # Add upgraded cells to contacts
            m_siding *= np.logical_not(m_belowcont) # Remove upgraded cells from sidings
            # Find new recruits by iterating contacts of contacts
            while np.any(m_contacts):
                if verbose:
                    print('.', end=' ')
                    sys.stdout.flush()
                # Promote known contacts to recruits
                m_recruits |= m_contacts
                # Check for recruits running afoul of forbidden territory
                m_fouls = m_recruits * m_forbidden
                if np.any(m_fouls):
                    print()
                    print(f'Foreign object fouls: {np.sum(m_recruits*m_foreignobjs)}')
                    print(f'Input mask fouls: {np.sum(m_recruits*m_excluded)}')
                    break
                # So far the recruits are all fine, search can go on..
                # Search for new applicants for contact in neighbourhood of contacts
                m_hood = Neighbours(m_contacts)
                m_applicants = m_hood * np.logical_not(m_members|m_recruits)
                # Separate applicants:
                # contacts up to the isocontour level, else sidings
                m_contacts = m_applicants * m_belowcont
                m_siding |= m_applicants * np.logical_not(m_belowcont)
                
            # No contacts left, recruits all fine (will become promoted to members in next cycle)
            else: # only triggers if np.any(m_contacts) becomes false without breaking the loop
                isocont_obj = isocont_value
                isocont_percentile += dperc
                # Check against the isocontour level hitting the maximum global value.
                # Obviously, something went seriously wrong if this happens
                # (i.e. the isocontour containing the entire domain)
                if isocont_percentile > 100.:
                    raise ValueError("ISOS: Runaway isocontour level")
                
        print(f'Object maxlevel: {obj_highpot}')
        print(f'Object isolevel: {isocont_obj}')
        print(f'Last   isolevel: {isocont_value}')
        
        w_object = np.where(m_members)
        s_object = set(zip(*w_object))
        object_list.append(s_object)
        obj_iso_potentials.append(isocont_obj)
        
        
        
    return deepcopy(object_list)


# ==============================================================================
# ==== NEIGHBORS OF NEIGHBORS ALGORITHM ========================================
# ==============================================================================

def neigh_of_neigh(blockdata, octree, selector, MinPts=100, **unk):
    #
    if unk:
        raise RuntimeError(f'EE neigh_of_neigh: Unknown parameters: {unk.keys()}')
    #
    if MinPts is None:
        print('W! neigh_of_neigh: MinPts not specified. Assuming MinPts=100.')
        MinPts = 100
    #
    BC = np.where(selector)
    blockcell_ = lzip(*BC)
    # Hack: put blockcell locator into 1-element-lists of B, CZ, CY, CX
    # TODO: REPAIR BUG IN MICFLASH TO REMOVE NECESSITY OF THIS NASTY HACK!
    # TODO: Why do map(np.array, zip(*bc)) boilerplate bullshit on any call? WTF?
    bchack = lambda bc: lmap(np.array, lzip(*[bc, ]))
    # Iterative cluster linking algorithm with in-place neighborhood discovery
    cluster_list = list()
    candidates = set(blockcell_)
    while candidates: # As long as there are candidate cells left:
        seed = candidates.pop() # Pick a cluster seed from the remaining candidates
        # Iteratively loop over contacts of contacts, starting from the cluster seed:
        cluster = set([seed, ])
        seed_neigh = (octree.findneigh(*bchack(seed)))[0]
        contacts = seed_neigh&candidates
        while contacts:
            # Integrate contacts into cluster
            candidates -= contacts
            cluster |= contacts
            # Get neighbors of contacts:
            noc = octree.findneigh(*lmap(np.array, lzip(*contacts)), combine=True)
            # The neighbors of the added contacts are the next turns contacts,
            # iff they are candidates (satisfying selector and not yet assigned):
            contacts = noc & candidates
            print('*', end=' ')
            sys.stdout.flush()
        # There are no more contacts of contacts in the candidate cells,
        # meaning there is nothing more to add to the cluster.
        # Save it if it is large enough, else leave it.
        if len(cluster) > MinPts:
            print(f'Found Cluster: {len(cluster)} Cells')
            sys.stdout.flush()
            cluster_list.append(cluster)
        else:
            print(f'Found Group: {len(cluster)} Cells (Discarded)')
            sys.stdout.flush()
        # If there are candidate cells left, start over (at while candidates);
        # if not, all clusters have been found.
    return deepcopy(cluster_list)


# ==============================================================================
# ==== CAVITY STITCHING ALGORITHM ==============================================
# ==============================================================================

def FindCavities(leafdata, octree, cluster_list, outer_criteria):
    # Get the baseshape of the simulation domain data field
    mass = leafdata['mass']
    baseshape = mass.shape
    
    # STEP A: Mark cluster lists (one cumulative mask)
    cl_masks = np.array([GetMask(baseshape,l) for l in cluster_list])
    cl_mask = np.any(cl_masks, axis=0)
    
    # STEP B: Mark cells that are inside the range defined by the outer criterion,
    # but not part of one of the cluster lists
    inside_mask = mask_cells(leafdata, outer_criteria)
    noncl_mask = np.logical_not(cl_mask)
    noncl_inside_mask = inside_mask * noncl_mask
    
    # STEP C: Find ambient area:
    print('-> Detecting Ambient Linking...')
    sys.stdout.flush()
    # 1. Mark cells outside the range defined by the outer criterion
    outside_mask = np.logical_not(inside_mask)
    assert np.any(outside_mask)
    # 2. Get the boundary of the inside_mask part.
    # This does not have to catch 100%, (but should be fast and conserve memory),
    # as is used as a seed to find the ambient part of the inner region by NoN from its boundary.
    pattern = np.ones((3,3,3))
    f_convolve = lambda block: convolve(block, pattern, mode='same')
    convolved_mask = np.array(lmap(f_convolve, outside_mask), dtype=bool)
    boundary_mask = noncl_inside_mask*convolved_mask   
    n_seed_max = 100000
    stride = int(np.sum(boundary_mask) / n_seed_max) +1
    print(f'Boundary points: {np.sum(boundary_mask)}')
    # 3. From there, do FOF with the cells marked in Step B as noncl parts:
    ambient_mask = np.copy(outside_mask)
    contacts = set(zip(*np.where(boundary_mask))[::stride])
    non_assigned = set(zip(*np.where(noncl_inside_mask)))
    print(f'Boundary seed len: {len(contacts)}')
    while contacts:
        non_assigned -= contacts
        ambient_mask[select(contacts)] = True
        # Get neighbors of contacts:
        ambient_neigh = octree.findneigh(*lmap(np.array, lzip(*contacts)), combine=True)
        # The neighbors of the added contacts are the next turns contacts,
        # iff they are candidates (satisfying selector and not yet assigned):
        contacts = ambient_neigh & non_assigned
        print(f'{len(contacts)} contacts')
        sys.stdout.flush()
    # What is left of the non-assigned cells must now be part of a cavity:
    cavity_mask = GetMask(baseshape, non_assigned)
    # Sanity check: This must amount to what is not ambient or in a cluster:
    non_cavity_mask = np.logical_or(ambient_mask, cl_mask)
    assert np.any(cavity_mask == np.logical_not(non_cavity_mask))
    
    # STEP D: Do FOF on the cells that are neither outside area nor in a cluster
    print('-> Detecting Cavity Linking...')
    sys.stdout.flush()
    cavity_list = neigh_of_neigh(leafdata, octree, cavity_mask, MinPts=0)
    return cavity_list
    
def MatchCavities(leafdata, octree, cluster_list, cavity_list):
    # Get the baseshape of the simulation domain data field
    mass = leafdata['mass']
    baseshape = mass.shape
    # Create field to hold cluster indicies per cell
    icl = np.full(baseshape, -1)
    # Fill field holding the cluster indicies
    for i,cl in enumerate(cluster_list):
        cl_where = select(cl)
        icl[cl_where] = i
    #
    cl_mask = icl > -1
    #
    print('-> Stitching Cluster Cavities...')
    sys.stdout.flush()
    #
    for i,cav in enumerate(cavity_list):
        cav_neigh = octree.findneigh(*lmap(np.array, lzip(*cav)), combine=True)
        cav_neigh_mask = GetMask(baseshape, cav_neigh)
        cav_cl_mask = cl_mask*cav_neigh_mask
        if np.any(cav_cl_mask):
            cav_icl = icl[cav_cl_mask][0]
            if not np.all(icl[cav_cl_mask]==cav_icl):
                n_stitch = np.sum(cav_cl_mask)
                print(f'W! WARNING: Ambiguous stitch {n_stitch} discarded!')
                continue
            cav_where = select(cav)
            icl[cav_where] = cav_icl
    # Read back stitched cluster indicies into lists:
    cluster_list_stitched = list()
    for i,cl in enumerate(cluster_list):
        cl_new_where = np.where(icl==i)
        cl_list = lzip(*cl_new_where)
        cluster_list_stitched.append(cl_list)
        
    return cluster_list_stitched

def StitchCluster(leafdata, octree, cluster_list, outer_criteria):
    # If cluster list empty (nothing there to stitch): Quick bailout
    if not len(cluster_list):
        return list()
    # 
    cavity_list = FindCavities(leafdata, octree, cluster_list, outer_criteria)
    cluster_list_stitched = MatchCavities(leafdata, octree, cluster_list, cavity_list)
    return deepcopy(cluster_list_stitched)


# ==============================================================================
# ==== CLUSTER FINDING ALGORITHM ===============================================
# ==============================================================================

def findcluster_adv(blockdata, octree, **param):
    ''' 
    Does neighbors of neighbors cell linking, removing cavities from linked groups.
    This is the 2019 (slow) version of the cavity removal
     '''
    print('-> Selecting Cells...')
    sys.stdout.flush()
    cell_criteria = param.pop('criteria')
    mask = mask_cells(blockdata, cell_criteria)
    
    print('-> Detecting Antiselection Linking...')
    sys.stdout.flush()
    antimask = np.logical_not(mask)
    anticluster_list = neigh_of_neigh(blockdata, octree, antimask, MinPts=0)

    print('-> Adding Antiselection Crumbs to Selection...')
    sys.stdout.flush()
    simvol = blockdata['vol'].sum()
    for crumb in lmap(select, anticluster_list):
        vol = blockdata['vol'][crumb]
        crumbvol = vol.sum()
        fraction = crumbvol/simvol
        if  fraction < .2:
            mask[crumb] = True
            print(f'Crumb: {vol.size} cells {fraction}')
        else:
            print(f'Large Part: {len(crumb)} cells ({fraction})')

    print('-> Detecting Neighbors of Neighbors Linking...')
    sys.stdout.flush()
    cluster_list = neigh_of_neigh(blockdata, octree, mask, **param)

    return cluster_list

def findcluster_direct(blockdata, octree, regmask=None, **param):
    ''' Does neighbors of neighbors cell linking '''
    print('-> Selecting Cells...')
    sys.stdout.flush()
    cell_criteria = param.pop('criteria')
    print(cell_criteria)
    mask = mask_cells(blockdata, cell_criteria)
    if regmask is not None:
        assert mask.shape == regmask.shape
        mask *= regmask
    print(f'#DEBUG# Selected {np.sum(mask)} cells')
    print('-> Detecting Neighbors of Neighbors Linking...')
    sys.stdout.flush()
    object_list = neigh_of_neigh(blockdata, octree, mask, **param)
    return object_list    
    
def findcluster_isos(blockdata, octree, **param):
    ''' Does neighbors of neighbors cell linking up to a contained isocontour '''
    print('-> Selecting Cells...')
    sys.stdout.flush()
    cell_criteria = param.pop('criteria')
    mask = mask_cells(blockdata, cell_criteria)
    print(f'#DEBUG# Selected {np.sum(mask)} cells')
    print('-> Detecting Potential Well Isocontour Boundaries...')
    sys.stdout.flush()
    object_list = isocont_non(blockdata, octree, mask, **param)
    return object_list
    
def padcluster_isos(blockdata, octree, objlist, **param):
    ''' 2nd stage algorithm: Puts isosurfaces around already found objects '''
    print('-> Selecting Cells...')
    sys.stdout.flush()
    cell_criteria = param.pop('criteria')
    mask = mask_cells(blockdata, cell_criteria)
    print(f'#DEBUG# Selected {np.sum(mask)} cells')
    print('-> Detecting Potential Well Isocontour Boundaries...')
    sys.stdout.flush()
    object_list = isocont_cutout(blockdata, octree, mask, objlist, **param)
    return object_list
    

# ==== TEST SETTINGS ===========================================================
plotfile = '/home/mweis/CF97W6R_N1B0250_hdf5_plt_cnt_0200'
findcluster = findcluster_direct

# ==== TEST ====================================================================
if __name__ == '__main__':
    pass
