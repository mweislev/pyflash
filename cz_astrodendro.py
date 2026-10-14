import sys
import numpy as np
from astrodendro import Dendrogram

import cl_fofg_142 as fof
import libmf.tools as tools
import cz_tools as cz

from copy import deepcopy


#===============================================================================
# ==== OBJECT BULK TRANSFORMATIONS =============================================
#===============================================================================

def ObjCenter(hdfdata, octree, objwhere):
    # Get object cell raw positions
    posx_raw = hdfdata['posx']
    posy_raw = hdfdata['posy']
    posz_raw = hdfdata['posz']
    # Rotate the coordinate system window to center one of the object
    # grid points (i.e. select the first one) according to per. bounds.
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    xsize, ysize, zsize = xmax -xmin, ymax -ymin, zmax -zmin
    posx = tools.modclip(posx_raw, posx_raw[0]-.5*xsize, posx_raw[0]+.5*xsize)
    posy = tools.modclip(posy_raw, posy_raw[0]-.5*ysize, posy_raw[0]+.5*ysize)
    posz = tools.modclip(posz_raw, posz_raw[0]-.5*zsize, posz_raw[0]+.5*zsize)
    cellpos = (posx,posy,posz)
    # Calculate center of mass using rotated grid window coordinates
    mass = hdfdata['mass'][objwhere]
    center = [np.average(cellpos[i], weights=mass) for i in range(3)]
    center_unw = cz.BG_UnwindPeriodic(octree, center)
    return np.array(center_unw)

def ObjRadius(hdfdata, octree, objwhere, objcenter):
    relcoords = cz.BG_RelCoords(octree, objcenter)
    x, y, z = [relcoords[i][objwhere] for i in range(3)] 
    rsq = x**2 + y**2 + z**2
    radius = np.sqrt(np.max(rsq))
    return radius

def ObjBoundingBox(hdfdata, octree, objwhere, objcenter, margin=1.1):
    relcoords = cz.BG_RelCoords(octree, objcenter)
    xrel, yrel, zrel = [relcoords[i][objwhere] for i in range(3)]
    rx = np.max(np.fabs(xrel))
    ry = np.max(np.fabs(yrel))
    rz = np.max(np.fabs(zrel))
    r3d = np.array([rx,ry,rz])
    bbox = cz.RadiusExtent(objcenter, r3d, a=margin)
    return bbox


# ==============================================================================
# ==== AMR ADJACENCY LOGIC (LUT) ===============================================
# ==============================================================================

# This function tries to solve for the following requirements:
# a) Avoid necessity of eliminating doublettes from naive neighbourhood probing
# b) Do not call octree on every single neighbourhood call,
#    since the function will be called often and only has to process one cell
#    for each call.
# c) Avoid doing any branching / type inference for above in situ
#    (precalculate as much shit as possible),

# Ideas:
# * Pre-compute a table that can be used to look up the neighbours of
#   any given cell.
# * If the cellsize of an adjacent block is smaller,
#   we might not need to hit all neighbours opposite of the given face.
#   Fetching one of them per cell face is enough.
#   This means we can keep 6 pre-calculated sets of neighbouring cells
#   (one for each orientation and direction) in a lookup table,
#   which will be pre-filled once when invoking the closure.
# * We only need to invoke the octree finder at the block boundaries.

def modclip(a, a_min, a_max):
    return a_min + np.mod(a-a_min, a_max-a_min)

def AmrAdjacencyLUT(hdfdata, octree, lut_dtype=np.int32):
    '''
    Return a neighbour function that can be used for the neighbour parameter in astrodendro.
    This will make astrodendro work directly on paramesh AMR data from Flash.
    '''
    # CAUTION: DO NOT CHANGE THE LUT TYPE TO AN UNSIGNED DATATYPE!
    # Note that astrodendro expects an index of -1 for left adjacency in any dimension
    # where the cell face coincides with the left domain boundary.
    print('Preparing Flash AMR adjacency LUT...', end='')
    sys.stdout.flush()
    nbl, nzb, nyb, nxb = hdfdata.dshape()
    # The last index of the neighbour lookup tables will count through the
    # neighbours (ravelled in neg,pos and x,y,z ie. i,j,k)
    nneigh = 6
    # Prepare a field of ones in the shape and dtype of the final datatype.
    # This will be used to conveniently cast sparse LUT calculations into the
    # full final shape of the LUTs by using it as a multiplicator.
    ones = np.ones((nbl,nzb,nyb,nxb,nneigh), dtype=int)
    # Initialize neighbourhood LUT.
    # For now, the neighbour entries point at the cells themselves
    bl_neigh_LUT = np.arange(nbl)[:,None,None,None,None] * ones
    k_j_i = np.meshgrid(*[np.arange(l) for l in (nzb,nyb,nxb)], indexing='ij')
    k_neigh_LUT, j_neigh_LUT, i_neigh_LUT = [np.array(a[None,:,:,:,None]*ones, dtype=lut_dtype) for a in k_j_i]
    # Add neighbourhood offsets. Obviously, the result will still not
    # be valid at the block boundaries.
    i_neigh_LUT[...,0] -= 1
    i_neigh_LUT[...,1] += 1
    j_neigh_LUT[...,2] -= 1
    j_neigh_LUT[...,3] += 1
    k_neigh_LUT[...,4] -= 1
    k_neigh_LUT[...,5] += 1
    ## Resolve the block boundary neighbours via coordinate calculation,
    ## followed by an octree call at that coordinate.
    ## TODO: CHECK FOLLOWING CODE BLOCK, THIS WAS WRITTEN IN THE HOUR OF WOLF...
    # Read grid layout data
    nb = nxb,nyb,nzb
    coords = octree.coords(force=True)
    xyz_per = octree.periodic()
    xyz_min = octree._domain_min
    xyz_max = octree._domain_max
    #
    XYZ = np.array([coords[:,dim] for dim in range(3)])
    dXdYdZ = np.array([hdfdata['lenx'], hdfdata['leny'], hdfdata['lenz']])
    for dim in range(3):
        for a,idx in enumerate((0,nb[dim]-1)):
            neigh = a+2*dim
            sgn = [-1.,+1.][a]
            # Mask boundary. Note that k_j_i lists dimensions in backwards order.
            # (This feels a bit too clever, meaning: This line is probably a bug!)
            mask = (k_j_i[2-dim]*ones[...,neigh]) == idx
            # Calculate offset coordinates: Go a 3/4 cell in/out in dim direction,
            # 1/4 in/out in the other directions
            XYZm_offset = XYZ[:,mask]
            XYZm_offset[dim] += .75*sgn*dXdYdZ[dim,mask]
            XYZm_offset[(dim+1)%3] += .25*sgn*dXdYdZ[(dim+1)%3,mask]
            XYZm_offset[(dim+2)%3] += .25*sgn*dXdYdZ[(dim+2)%3,mask]
            # Wrapping / domain bound detection
            # Astrodendro allows out-of-bound indicies of -1 and nb for 
            # out-of-bound adjacency. This means we can leave the
            # index values for out-of-bound cells as is.
            # However, we still need to identify&mask those offsets, and we
            # need to care about periodic bounds.
            # We only need to care about the working dir here.
            mask_nib = np.copy(mask)
            if xyz_per[dim]:
                XYZm_offset[dim] = modclip(XYZm_offset[dim], xyz_min[dim], xyz_max[dim])
                # Above wrapping should have put all the dim coords inside the bounds.
                # However, we still need the inbounds field for technical reasons.
                inbounds = (xyz_min[dim]<XYZm_offset[dim]) * (XYZm_offset[dim]<xyz_max[dim])
                assert np.all(inbounds)
            else:
                inbounds = (xyz_min[dim]<XYZm_offset[dim]) * (XYZm_offset[dim]<xyz_max[dim])
                mask_nib[mask] *= inbounds
                # If all blocks are at the same domain bound then
                # smth went wrong (or the sim is rlevel 1)
                assert np.any(inbounds) 
            # Find cells belonging to the offset coordinates
            X_mo, Y_mo, Z_mo = XYZm_offset[:,inbounds]
            bl_mo, k_mo, j_mo, i_mo = octree.find(X_mo, Y_mo, Z_mo)
            # Write back mask-related results.
            bl_neigh_LUT[mask_nib,neigh] = bl_mo
            k_neigh_LUT[mask_nib,neigh] = k_mo
            j_neigh_LUT[mask_nib,neigh] = j_mo
            i_neigh_LUT[mask_nib,neigh] = i_mo
            #
            print(f' {('x','y','z')[dim]}{a}', end='')
            sys.stdout.flush()
    print(' OK')
    sys.stdout.flush()

    def neighbours(foo, idx):
        ''' Find adjacent cells for any given paramesh AMR cell indexed by idx. '''
        # Caution: astrodendro gives idx as list of indicies (of one cell)
        # but we actually need a tuple to select that specific index combo
        # from a multidim array (in recent versions of numpy).
        I = tuple(idx) 
        bl_neigh = bl_neigh_LUT[I]
        k_neigh = k_neigh_LUT[I]
        j_neigh = j_neigh_LUT[I]
        i_neigh = i_neigh_LUT[I]
        neighidx = list(zip(bl_neigh,k_neigh,j_neigh,i_neigh))
        return list(neighidx)

    return neighbours


# ==============================================================================
# ==== HELPER ==================================================================
# ==============================================================================

formulae = {
    'dens_log': ('dens', np.log10),
    'numdens_log': ('numdens', np.log10),
    'l_jeans_recp': (1.0, 'l_jeans', '/'),
    'l_jeans_recp_log': (1.0, 'l_jeans', '/', np.log10),
}


# ==============================================================================
# ==== DENDROGRAM LEAF DETECTION ===============================================
# ==============================================================================

def Dendronodes(leafdata, octree, mask, l1clist, var='dens_log', leavesonly=False, **ddgargs):
    dendroobjs = list()
    if not l1clist:
        return dendroobjs
    data_amr = leafdata[var]
    baseshape = data_amr.shape
    neighbours = AmrAdjacencyLUT(leafdata, octree)
    for iobj, objl in enumerate(l1clist):
        print(f'...Processing L1-object {iobj+1}/{len(l1clist)}...', end=' ')
        sys.stdout.flush()
        objmask = fof.GetMask(baseshape, objl)
        objmask *= mask
        data_masked = np.full_like(data_amr, np.nanmin(data_amr))
        data_masked[objmask] = data_amr[objmask]
        dendro = Dendrogram.compute(data_masked, neighbours=neighbours, **ddgargs)
        if leavesonly:
            structures = list(dendro.leaves)
            print(f'Saving {len(structures)} dendrogram leafs.')
        else:
            structures = list(dendro.all_structures)
            print(f'Saving {len(structures)} dendrogram nodes.')
        for il,dobj in enumerate(structures):
            nodemask = dobj.get_mask()
            nodewhere = np.where(nodemask)
            clist = fof.lzip(*nodewhere)
            dendroobjs.append(clist)
        # Clean up dendrogram data
        for key in dendro._structures_dict:
            dendro._structures_dict[key]._tree_index = None
            dendro._structures_dict[key] = None
        dendro._structures_dict.clear()
        dendro.index_map = None
        dendro.data = None
        dendro = None
    #
    print(f'...Detected {len(dendroobjs)} dendrogram objects from {len(l1clist)} L1-objects.')
    return deepcopy(dendroobjs)


def isolate_leafs(leafdata, octree, l1clist, **param):
    leafdata.learn(formulae)
    print('-> Selecting Cells...')
    sys.stdout.flush()
    cell_criteria = param.pop('criteria', dict())
    # mask will be true everywhere if mask_cells receives an empty cell_criteria dict
    mask = fof.mask_cells(leafdata, cell_criteria)
    print('-> Detecting Substructure Dendrogram Leafs...')
    sys.stdout.flush()
    object_list = Dendronodes(leafdata, octree, mask, l1clist, leavesonly=True, **param)
    return object_list

def get_nodes(leafdata, octree, l1clist, **param):
    leafdata.learn(formulae)
    print('-> Selecting Cells...')
    sys.stdout.flush()
    cell_criteria = param.pop('criteria', dict())
    # mask will be true everywhere if mask_cells receives an empty cell_criteria dict
    mask = fof.mask_cells(leafdata, cell_criteria)
    print('-> Detecting Substructure Dendrogram Nodes...')
    sys.stdout.flush()
    object_list = Dendronodes(leafdata, octree, mask, l1clist, **param)
    return object_list


# ==============================================================================
# ==== TEST ====================================================================    
# ==============================================================================

plotfile = '/home/mweis/scratch/czsample/b.Aa1-Aa4-2zone/out/CZ-b_hdf5_chk_2028'
testbl = 0

if __name__ == '__main__':
    import constants as cu

    leafdata, octree = cz.LoadPlotfile(plotfile)
    neighbours = AmrAdjacencyLUT(leafdata, octree)
    coords = octree.coords()
    X,Y,Z = (coords[:,dim] for dim in range(3))

    cells = (
        (testbl,4,4,4),
        (testbl,4,4,0),
        (testbl,4,0,4),
        (testbl,0,4,4),
        (testbl,4,4,7),
        (testbl,4,7,4),
        (testbl,7,4,4),
    )
    for idx in cells:
        nc = neighbours(None,idx)
        print()
        print(f'Neigh {idx} (coords (pc): {X[idx]/cu.pc} | {Y[idx]/cu.pc} | {Z[idx]/cu.pc}):')
        for entry in nc:
            pass
            print(tuple(entry))
