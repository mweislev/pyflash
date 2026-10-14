import os, sys
import numpy as np

import libmf.micflash as mflash
import libmf.tools as tools
import libmf.micplot as mplot

import cz_tools as cz

#===============================================================================
# ==== OBJECT-CELL RELATIONS ===================================================
#===============================================================================

def CellOwnershipObjLUT(objmasks, Moo_contain):
    n_obj = len(objmasks)
    dshape = objmasks[0].shape
    cell_owner_obj = np.full(dshape, -1, dtype=np.int16)
    for ia,a in enumerate(objmasks):
        objownmask = np.copy(a)
        masks_contained = [objmasks[ib] for ib in range(n_obj) if Moo_contain[ia,ib]]
        contained_mask = np.logical_or.reduce(masks_contained)
        objownmask *= np.logical_not(contained_mask)
        # Check against double assignment
        assert np.all(cell_owner_obj[objownmask]==-1)
        # Check against empty assignment
        assert np.any(objownmask)
        # Assign ownership
        cell_owner_obj[objownmask] = ia
    return cell_owner_obj

def CellOwnershipMasks(cell_owner_obj, objmasks=None):
    ownmasks = list()
    nobj = int(np.max(cell_owner_obj)) + 1
    for iobj in range(nobj):
        obj_owned_cells = (cell_owner_obj==iobj)
        ownmasks.append(obj_owned_cells)
    # Verify results if object masks are given
    if objmasks:
        assert len(ownmasks) == len(objmasks)
        for owned, member in zip(ownmasks,objmasks):
            assert np.any(owned) # Verify that each object owns at least one cell
            assert np.all(owned*member==owned) # Verify that all object owned cells are object cells
    return ownmasks


#===============================================================================
# ==== OBJECT-OBJECT RELATIONS =================================================
#===============================================================================

# We model two different kinds of relations, the combination of which
# is useful to understand the hierarchical structure.
# * o_i contains o_j: (j<>i) Inclusive relation:
#   All cells contained in o_j are also contained in o_i.
#   (but not vice versa, identical objects are not allowed and will be checked against)
# * o_i owns o_j: Exclusive relation:
#   o_i contains o_j and there is no intermediate o_k (k<>i,j)
#   such that o_i would contain o_k and o_k would contain o_j

def ObjObjContainmentMatrix(objmasks):
    """ Containment Matrix Moo[i,j]: Does object i contain object j (obj_j<obj_i)? """
    n_obj = len(objmasks)
    Moo_contain = np.zeros((n_obj,n_obj), dtype=bool)
    for ia,a in enumerate(objmasks):
        a_coverage_ = lambda b: np.sum(a*b) / np.sum(b)
        a_contains_ = lambda b: np.isclose(a_coverage_(b), 1.)
        Moo_contain[ia] = [a_contains_(b) for b in objmasks]
    # For practical reasons, do not track self-containment:
    Moo_contain *= np.logical_not(np.diag(n_obj*[True,]))
    # Check against identical objects:
    assert not np.any(Moo_contain*Moo_contain.T)
    #
    return Moo_contain

def ObjObjOwnershipMatrix(Moo_contain):
    """ Ownership matrix M[i,k]: Does object i directly contain object k (obj_j<obj_k),
    such that there is no object j with i contains j and j contains k? """
    # Step 1: Build matrix ijk_matryoshka[i,j,k] for o_i contains o_j and o_j contains o_k
    # Note that the matrix is False if i=j or j=k by construction of the containment matrix,
    # and that it is False for i=k (otherwise there would be j with i contains j and j contains k=i).
    i_contains_j = Moo_contain[:,:,None]
    j_contains_k = Moo_contain[None,:,:]
    ijk_matryoshka = i_contains_j*j_contains_k
    # Step 2: If ijk_matryoshka[i,j,k] is true for any intermediate object j then
    # object k cannot be directly owned by object i
    ik_indirect = np.any(ijk_matryoshka, axis=1)
    i_contains_k = Moo_contain
    i_owns_k = i_contains_k * np.logical_not(ik_indirect)
    # Check against double ownership (a given i may own multiple k, but not v.v.)
    n_parents = np.sum(i_owns_k, axis=0)
    assert np.all(n_parents <= 1)
    # 
    return i_owns_k

def ObjObjOwnershipLUT(Moo_own):
    """ Convert ownership matrix into lookup table (object a is owned by which parent) """
    pass

def ObjParent(i_owns_k, k_obj):
    n_objs = i_owns_k.shape[0]
    I = np.arange(n_objs)
    # Per construction, any object k can be either unowned or owned by exactly one object i
    ownermask = i_owns_k[:,k_obj]
    assert np.sum(ownermask) in (0,1)
    if np.any(ownermask):
        i_owner = I[ownermask].item()
        return i_owner
    else:
        return -1
    
def ObjChildren(i_owns_k, i_obj):
    """ 
    Find indicies of children for a given object index.
    """
    # Note that this can be used for child objects as well as child sink particles,
    # depending on which field is inserted for i_owns_k.
    n_objs = i_owns_k.shape[0]
    I = np.arange(n_objs)
    childmask = i_owns_k[i_obj,:]
    childindicies = I[childmask]
    return childindicies

def LeafObjMask(i_owns_k):
    n_children = np.sum(i_owns_k, axis=1)
    leafmask = (n_children==0)
    # In any hierarchy there must be a lowest level, or something went horribly wrong
    assert np.any(leafmask)
    # Return format: One bool for each object: Is it a leaf?
    return leafmask

def RootObjMask(i_owns_k):
    n_objs = i_owns_k.shape[1]
    has_parrent_ = [ObjParent(i_owns_k, k_obj)>=0 for k_obj in range(n_objs)]
    rmask = np.logical_not(has_parrent_)
    return rmask

def ObjDepth(i_owns_k, maxlevel=100):
    n_objs = i_owns_k.shape[1]
    # For each node, set nextparent to the nodeindex itself
    nextparent = np.arange(n_objs)
    depth = np.full(n_objs, -1, dtype=int)
    # Iterate nextparents up to their next parent level by level, until no parent if found
    for level in range(maxlevel):
        nextparent = np.array([ObjParent(i_owns_k, k_obj) for k_obj in nextparent])
        depth[(nextparent==-1)*(depth==-1)] = level
        if np.all(depth>=0): break
    else:
        raise RuntimeError('ObjDepth: Breached maximum recursion depth. Try to increase maxlevel!')
    return depth


#===============================================================================
# ==== OBJECT-SINK RELATIONS ===================================================
#===============================================================================

def ObjSinkContainmentMatrix(octree, objmasks, sinkdict):
    # Step 1: Find grid indicies for sink positions
    sink_x, sink_y, sink_z = sinkdict['posx'], sinkdict['posy'], sinkdict['posz']
    GrI = octree.find(sink_x, sink_y, sink_z)
    # Step 2: Find cell indicies at sink positions
    # return format: Matrix[i,s]: Object i contains sink s
    pass

def ObjSinkOwnershipMatrix(Moo_contain, Mos_contain):
    # Step 1: Build matrix ijs_matryoshka[i,j,s] for o_i contains o_j and o_j contains s_k
    i_contains_j = Moo_contain[:,:,None]
    j_contains_s = Mos_contain[None,:,:]
    ijs_matryoshka = i_contains_j*j_contains_s
    # Step 2: If ijs_matryoshka[i,j,s] is true for any intermediate object j then
    # sink s cannot be directly owned by object i
    is_indirect = np.any(ijs_matryoshka, axis=1)
    i_contains_s = Mos_contain
    i_owns_s = i_contains_s * np.logical_not(is_indirect)
    # Check against double ownership (a given i may own multiple s, but not v.v.)
    n_parents_sink = np.sum(i_owns_s, axis=0)
    assert np.all(n_parents_sink<=1)
    #
    return i_owns_s

#===============================================================================
# ==== OBJECT-TRACER RELATIONS =================================================
#===============================================================================

def ObjTracerContainmentMatrix(objmasks, trdata):
    # We probably will not have the memory for that!
    raise NotImplementedError('ObjTracerContainmentMatrix: Not implemented (mem. req. not workable)')

def ObjTracerContainmentMask(octree, objmask, trdata, trmask=None):
    ''' Mask the tracers that are contained in a specific object '''
    # Use trmask if only a subset of the available tracer list is to be processed.
    # In this case, the output mask will be a mask over the entire tracer field nonetheless.
    if trmask is None:
        trmask = np.ones(trdata.shape, dtype=bool)
    # Find grid indicies for particle positions
    part_x, part_y, part_z = trdata['posx'][trmask], trdata['posy'][trmask], trdata['posz'][trmask]
    GrI = octree.find(part_x, part_y, part_z)
    # Read object mask at given indicies at particle positions
    obj_tr_mask = np.zeros(trdata.shape, dtype=bool)
    obj_tr_mask[trmask] = objmask[GrI]
    # Return format: One bool for each tracer: Contained by given object?
    return obj_tr_mask

def TracerOwnershipObjLUT(octree, objmasks, Moo_contain, trdata, trmask=None):
    """ Which tracer is owned by which object? """
    # Use trmask if only a subset of the available tracer list is to be processed.
    # In this case, the output will have the format trdata[trmask]
    if trmask is None:
        trmask = np.ones(trdata.shape, dtype=bool)
    # Step 0: No objects, no problem
    if not objmasks:
        print('Warning: TracerOwnershipObjLUT: No objects!')
        tr_owner_obj = np.full(trdata[trmask].shape, -1, dtype=np.int16)
        return tr_owner_obj
    # Step 1: Which object owns which cell? (AMR Grid with object indicies, -1 = no owner)
    cell_owner_objs = CellOwnershipObjLUT(objmasks, Moo_contain)
    # Step 2: Find grid indicies for particle positions
    part_x, part_y, part_z = trdata['posx'][trmask], trdata['posy'][trmask], trdata['posz'][trmask]
    GrI = octree.find(part_x, part_y, part_z)
    # Step 3: Find out which object owns the cell at the tracers location
    tr_owner_obj = cell_owner_objs[GrI]
    # Check against mo owner assignments
    assert np.any(tr_owner_obj>=0)
    # Return format: One object index (owner) per tracer particle, -1 = no owner)
    return tr_owner_obj

def ObjTracerOwnershipMask(tr_owner_obj, iobj):
    tr_ownmask = (tr_owner_obj==iobj)
    return tr_ownmask

#===============================================================================
# ==== TEST ====================================================================
#===============================================================================

ppath = '/home/mweis/scratch/thesisdata/corezoom-work-wbox/b.Aa1-Aa4-2zone/out'
pf = os.path.join(ppath, 'CZ-b_hdf5_plt_cnt_0300')
pd = cz.PartfileFromPlotfile(pf)
criterion = 'COCxp30:adrnode:czfrag'

def GetNodeDepth(dendromasks):
    Moo_contain = ObjObjContainmentMatrix(dendromasks)
    i_owns_k = ObjObjOwnershipMatrix(Moo_contain)
    depth = ObjDepth(i_owns_k)
    is_root = RootObjMask(i_owns_k)
    assert np.all((depth==0)==is_root)
    return depth

def TestRelations():
    import cl_find_181 as clfind
    leafdata, octree = cz.LoadPlotfile(pf)
    objmasks = clfind.get_cmasks(leafdata, octree, criterion)
    depth = GetNodeDepth(objmasks)
    print(depth)


#===============================================================================
# ==== LOBBY ===================================================================
#===============================================================================

if __name__ == '__main__':
    TestRelations()