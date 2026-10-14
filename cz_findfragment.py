import os,sys
import numpy as np

import cz_tools as cz
import cl_gaussfit as clgf
import libmf.micflash as mflash

import cz_selfgrav as ptgr
import constants as cu

__author__ = "Michael Weis"

#===============================================================================
# ==== CONSTANTS ===============================================================
#===============================================================================

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKCYAN = '\033[96m'
    OKGREEN = '\033[92m'
    WARNYELL = '\033[93m'
    FAILRED = '\033[91m'
    ENDC = '\033[0m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'


#===============================================================================
# ==== FRAGMENT CANDIDATE HELPER ===============================================
#===============================================================================

def ObjCoords(leafdata, octree, objmask):
    objcenter = cz.ObjCenter(leafdata, octree, objmask)
    relcoords = cz.BG_RelCoords_D(leafdata, octree, objcenter)
    # Read coordinates of object cells
    objx, objy, objz = [relcoords[i][objmask] for i in range(3)]
    objxyz = objx,objy,objz
    return objcenter, objxyz

def MaskGpotMinima(leafdata, octree, **kwargs):
    height = -1.*leafdata['gpot']
    minmask = clgf.GetMaxima(height, octree, **kwargs)
    return minmask


#===============================================================================
# ==== DATA HELPER =============================================================
#===============================================================================

def Divergence(leafdata, octree, key):
    varx_grid = mflash.datagrid(leafdata[f'{key}x'], octree, ng=1)
    vary_grid = mflash.datagrid(leafdata[f'{key}y'], octree, ng=1)
    varz_grid = mflash.datagrid(leafdata[f'{key}z'], octree, ng=1)
    divvar = mflash.divergence(varx_grid,vary_grid,varz_grid)
    return divvar

#===============================================================================
# ==== FRAGMENT CANDIDATE CHECKER ==============================================
#===============================================================================

def ObjContainsSink(objmask, sinkcount, debug=True):
    nsinks_contained = np.sum(objmask*sinkcount)
    nsinks_total = np.sum(sinkcount)
    if debug:
        print(f'Contained sink particles {nsinks_contained}/{nsinks_total}')
    result = nsinks_contained # > 0
    return result

def ObjContainsPeak(objmask, peakmask, debug=False):
    """
    Check if the object contains a peak,
    usually a local minimum of gpot or a local maximum of dens
    """
    npeaks_contained = np.sum(objmask*peakmask)
    if debug:
        print(f'Contained peaks: {npeaks_contained}')
    result = npeaks_contained # > 0
    return result

def ObjIsBound(leafdata, octree, objmask, objcenter, debug=False, selfgrav=True):
    """
    Check if the sum of the volume energy terms of the object,
    Ekin+Etherm+Emag+Epot, is negative.
    """
    # Define field reader that only returns the object cells
    objdata = lambda key: leafdata[key][objmask]
    # Calculate kinetic volume energy, based on velocities relative to bulk motion
    mass = objdata('mass')
    velx, vely, velz = objdata('velx'), objdata('vely'), objdata('velz')
    velx_obj = np.average(velx, weights=mass)
    vely_obj = np.average(vely, weights=mass)
    velz_obj = np.average(velz, weights=mass)
    vx, vy, vz = velx-velx_obj, vely-vely_obj, velz-velz_obj
    v_squared = vx**2+vy**2+vz**2
    Ekin = .5*np.sum(mass*v_squared)
    # Calculate magnetic volume energy
    vol = objdata('vol')
    magx, magy, magz = objdata('magx'), objdata('magy'), objdata('magz')
    B_squared = magx**2+magy**2+magz**2
    Emag = 1./(8.*np.pi)*np.sum(vol*B_squared)
    # Calculate thermal volume energy
    Eth = np.sum(1.5*objdata('pres')*vol)
    # Calculate potential volume energy
    gp = leafdata['gpot'][objmask]
    Egrav = .5*np.sum(mass*gp)
    if selfgrav:
        gp_self = ptgr.ObjGpotSelf(leafdata, octree, objmask, center=objcenter)
    else:
        gp_self = gp
    Egrav_self = .5*np.sum(mass*gp_self)
    # Judgement day
    Etot = Ekin+Emag+Eth+Egrav_self
    if debug:
        print(f'Ekin: {Ekin}')
        print(f'Emag: {Emag}')
        print(f'Eth:  {Eth}')
        if selfgrav:
            print(f'Egs:  {Egrav_self} (Egg: {Egrav})')
        else:
            print(f'Egg:  {Egrav}')
        print(f'Etot: {Etot}')
    is_bound = (Etot<0.)
    return is_bound

def ObjIsContracting(leafdata, objmask, objxyz, divvel=None, debug=False):
    """
    Check if the velocity field has a bulk radially inwards motion
    """
    # Define field reader that only returns the object cells
    objdata = lambda key: leafdata[key][objmask]
    # Calculate velocities relative to center of mass
    mass = objdata('mass')
    velx, vely, velz = objdata('velx'), objdata('vely'), objdata('velz')
    velx_obj = np.average(velx, weights=mass)
    vely_obj = np.average(vely, weights=mass)
    velz_obj = np.average(velz, weights=mass)
    vx, vy, vz = velx-velx_obj, vely-vely_obj, velz-velz_obj
    # Unpack coordinates of object cells
    relx, rely, relz = objxyz
    # Calculate normalized radial vector.
    r = np.maximum(1e-40,np.sqrt(relx**2+rely**2+relz**2))
    nx = relx/r
    ny = rely/r
    nz = relz/r
    # Calculate radial velocity component of each cell (outwards positive)
    vr = nx*vx + ny*vy + nz*vz
    # Calculate the total radial momentum
    momr = np.sum(vr*mass)
    # Calculate velocity field divergence rating
    if divvel is not None:
        divv = np.average(divvel[objmask], weights=mass)
    else:
        # Divergence check off, just set smth that will pass the test 
        divv = -1.
    # Debug output
    if debug:
        print(f'Radial bulk velocity (km/s): {momr/np.sum(mass)*1e-5}')
        if divvel is not None:
            print(f'Weighted velocity divergence (km/s/cpc): {divv*1e-5*(.01*cu.pc)}')
    # Judgement day
    is_contracting = (momr<0.) *(divv<0.)
    return is_contracting

def ObjIsFragment(leafdata, octree, objmask, peakmask, sinkcount=0, divvel=None, debug=True):
    """
    Check if the given object qualifies as a fragment.
    For this, the following conditions must be satisfied:
    a) The object must be bound or at least strongly self-gravitating
    b) The object must contain a local gpot minimum or a sink particle
    c) The object must be locally contracting
    """
    OKSINK, OKGAS, REJECT = bcolors.OKBLUE, bcolors.OKGREEN, bcolors.FAILRED
    END = bcolors.ENDC
    # Pre-calculate fields that might be needed
    objcenter, objxyz = ObjCoords(leafdata, octree, objmask)
    # If the object contains a sink particle, we will assume that it is a fragment
    if ObjContainsSink(objmask, sinkcount):
        if debug: print(f'{OKSINK}Found sink fragment ({np.sum(objmask)} cells).{END}')
        return True
    if not ObjContainsPeak(objmask, peakmask):        
        if debug: print(f'{REJECT}Not a fragment: No sink, no peak.{END}')
        return False
    if not ObjIsBound(leafdata, octree, objmask, objcenter):
        if debug: print(f'{REJECT}Not a fragment: No sink, not bound.{END}')
        return False
    if not ObjIsContracting(leafdata, objmask, objxyz, divvel=divvel):
        if debug: print(f'{REJECT}Not a fragment: No sink, not contracting.{END}')
        return False
    if debug: print(f'{OKGAS}Found gas fragment ({np.sum(objmask)} cells).{END}')
    return True


#===============================================================================
# ==== HIERACHICAL FILTERING ===================================================
#===============================================================================

def FilterHierarchical(objmasks, debug=False):
    """
    Only keep masks that do not contain another mask
    """
    objms_filtered = list()
    for objmask in objmasks:
        # Guard against empty masks
        if not np.any(objmask):
            print('Warning: FilterHierarchical encountered empty mask')
            continue
        # Try to add new mask to list of filtered masks
        nc_obj = np.sum(objmask)
        for i,refmask in enumerate(objms_filtered):
            nc_test = np.sum(refmask)
            common = objmask*refmask
            # Check for masks that are intersecting, but not containing each other
            # (This should never happen), but allow for some slack:
            nc_common = np.sum(common)
            frac_common = nc_common / min(nc_test,nc_obj)
            if 0.05<frac_common and frac_common<.95:
                raise ValueError('FilterHierarchical: Partial mask intersection')
            if frac_common>=.95:
                # Drop the new mask if it contains the filtered mask:
                if nc_obj>=nc_test: break
                # Replace the filtered mask with the new mask if the filtered contains the new:
                objms_filtered[i] = objmask
                break
        else: # Inner loop not broken
            objms_filtered.append(objmask)
    #
    print(f'FilterHierarchical: Kept {len(objms_filtered)}/{len(objmasks)} fragments')
    # Check the result:
    for i,m1 in enumerate(objms_filtered):
        for j,m2 in enumerate(objms_filtered[i+1:]):
            if debug:
                print(f'i:{i} | j:{i+1+j} | #m1:{np.sum(m1)} | #m2:{np.sum(m2)} | #m1*m2:{np.sum(m1*m2)}')
            if np.sum(m1*m2)>=.05*min(np.sum(m1),np.sum(m2)):
                raise ValueError('FilterHierarchical: Result invalid, check algorithm')
    #
    return objms_filtered


#===============================================================================
# ==== FRAGMENT CANDIDATE FILTERING ============================================
#===============================================================================

def CountSinks(leafdata, octree, sinkdict):
    """
    Return an mhd-data integer field that counts the number of sink particles
    inside each cell volume
    """
    WARN, END = bcolors.WARNYELL, bcolors.ENDC
    dshape = leafdata.dshape()
    count = np.zeros(dshape, dtype=int)
    # Check for presence of sink particles
    nsink = sinkdict['nsink']
    if nsink == 0:
        print(f'{WARN}CountSinks: Warning: Found no sink particles!{END}')
        return count
    # Read sink position (1D-arrays, 1 entry per sink)
    X = sinkdict['x']
    Y = sinkdict['y']
    Z = sinkdict['z']
    # Find cell index for each set of sink position coordinates
    cells = octree.find(X,Y,Z)
    # Count occurence of each cell
    for idx,(bl,k,j,i) in enumerate(zip(*cells)):
       count[bl,k,j,i] += 1
    # Check result
    nlost = nsink-np.sum(count)
    assert nlost >=0
    if nlost > 0:
        print(f'{WARN}CountSinks: Warning: Lost {nlost}/{nsink} sinks!{END}')
    #
    return count

def FilterFragments(leafdata, octree, objmasks, sinkdata=None, hierarchical=True, checkpeak=True):
    print(f'## PARAM hierarchical: {hierarchical}')
    # Do preparation work
    sinkcount = CountSinks(leafdata, octree, sinkdata)
    # Note: Peaks need not be checked if the input data is already from a dendrogram.
    if checkpeak:
       peakmask = MaskGpotMinima(leafdata, octree)
    else:
      peakmask = np.ones(leafdata.dshape(), dtype=bool)
    divvel = Divergence(leafdata, octree, 'vel')
    #
    fragments = list()
    nobj = len(objmasks)
    for iobj,objmask in enumerate(objmasks):
        print(f'Object {iobj+1}/{nobj}...')
        is_fragment = ObjIsFragment(leafdata, octree, objmask, peakmask, sinkcount, divvel=divvel)
        if is_fragment: fragments.append(objmask)
    print(f'Qualified fragments: {len(fragments)}/{nobj}')
    if hierarchical:
        return FilterHierarchical(fragments)
    else:
        return fragments

