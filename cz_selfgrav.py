import os,sys
import numpy as np

import cz_tools as cz
import constants as cu

__author__ = "Michael Weis"

#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

potparam = dict(method='tree', parallel=True, theta=0.5, G=cu.G_g)


#===============================================================================
# ==== INTERFACE ===============================================================
#===============================================================================

def CenterCoordsDisplaced(leafdata, octree, center=None):
    if center is None:
        X,Y,Z = leafdata['posx'], leafdata['posy'], leafdata['posz']
    else:
        relcoords = cz.BG_RelCoords_D(leafdata, octree, center)
        X,Y,Z = [relcoords[dim] for dim in range(3)]
    return X,Y,Z

def CenterCoordsCells(leafdata, octree, center=None):
    if center is None:
        coords = octree.coords(force=True)
        X,Y,Z = [coords[:,dim] for dim in range(3)]
    else:
        relcoords = cz.BG_RelCoords(octree, center, force=True)
        X,Y,Z = [relcoords[dim] for dim in range(3)]
    return X,Y,Z


def ObjGpotSelf(leafdata, octree, objmask, center=None, rsmooth_cells=2.5, debug=False):
    from pytreegrav import Potential
    # Catch cases where the object mask only contains 0 or 1 cells.
    # Notes: 1.) The latter case actually occured properly even when using an object size
    #            of at least 30 cells. The reason is that less cells than 30 might be left
    #            when investigating the exclusive cell set of a dendrogram node that cleanly
    #            split into subnodes.
    #        2.) Pytreegrav will go wrong when inserting a single point even if the array shape
    #            conforms to the dimensionality of a multi-point input.
    if np.sum(objmask) < 2:
        dshape = objmask[objmask].shape
        gpot_objc = np.zeros(dshape)
        return gpot_objc
    # Center coordinate window on object center of mass,
    # so the object is as far from the periodic bound as possible.
    # (Note: The Potential func. is not aware of boundary conditions.)
    X,Y,Z = CenterCoordsDisplaced(leafdata, octree, center)
    xyz_objc = np.array((X[objmask], Y[objmask], Z[objmask])).T
    mass_objc = leafdata['mass'][objmask]
    dcell = np.cbrt(leafdata['vol']) #/(np.pi*4./3.)
    rsmooth = rsmooth_cells * dcell
    hrad_objc = rsmooth[objmask]
    if debug: print(f'Calculating self-gravity potential ({hrad_objc.size} cells)...', end='')
    sys.stdout.flush()
    try:
        gpot_objc = Potential(xyz_objc,mass_objc, hrad_objc, **potparam)
    except KeyError:
        raise
    except:
        print("#ED objmask:", np.sum(objmask), np.shape(objmask))
        print("#ED xyz_objc:", type(xyz_objc), np.shape(xyz_objc), getattr(xyz_objc, "dtype", None))
        print("#ED mass_objc:", type(mass_objc), np.shape(mass_objc), getattr(mass_objc, "dtype", None), mass_objc)
        print("#ED hrad_objc:", type(hrad_objc), np.shape(hrad_objc), getattr(hrad_objc, "dtype", None), hrad_objc)
        raise
    if debug: print(f'OK')
    sys.stdout.flush()
    return gpot_objc

def ObjGpotGlob(leafdata, octree, objmask, center=None, rsmooth_cells=2.5, ng=0, verbose=True, debug=False):
    from pytreegrav import PotentialTarget
    mass_objc = leafdata['mass'][objmask]
    dcell = np.cbrt(leafdata['vol']) #/(np.pi*4./3.)
    rsmooth = rsmooth_cells * dcell
    hrad_objc =rsmooth[objmask]
    hrad_rgrid = rsmooth.ravel()
    # Use the displaced cell center of mass coordinates as sources of the gravity field
    # (Note that the coordinates are relative to the objects center of mass to treat periodic bc as well as possible)
    Xg,Yg,Zg = CenterCoordsDisplaced(leafdata, octree, center)
    xyz_objc = np.array((Xg[objmask], Yg[objmask], Zg[objmask])).T
    # Use the non-displaced cell center coordinates to sample the field at:
    # (Note that these must be relative to the center of mass, too)
    Xc,Yc,Zc = CenterCoordsCells(leafdata, octree, center)
    dshape = Xc.shape
    xyz_rgrid = np.array((Xc.ravel(), Yc.ravel(), Zc.ravel())).T
    if verbose:
        print(f'Calculating partial gravitational potential ({hrad_objc.size}/{hrad_rgrid.size} cells)...', end='')
        sys.stdout.flush()
    if debug:
        print(f'grid datashape:', dshape)
        print(f'xyz_rgrid.shape:', xyz_rgrid.shape)
        print(f'xyz_objc.shape:', xyz_objc.shape)
    gpot_rgrid = PotentialTarget(xyz_rgrid,xyz_objc,mass_objc,softening_target=hrad_rgrid, softening_source=hrad_objc, **potparam)
    if verbose:
        print(f'OK')
        sys.stdout.flush()
    gpot = gpot_rgrid.reshape(dshape)
    return gpot
