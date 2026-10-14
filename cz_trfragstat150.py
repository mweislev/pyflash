import os, sys
import numpy as np

import libmf.micflash as mflash
import libmf.tools as tools
from libmf.tools import fprint

import cl_find_181 as clfind

import cz_tools as cz
import cz_tracermatch as cztrmatch
import cz_tracerown as cztrown
import cz_findfragment as czfifra

import constants as cu

#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

# ==== I/O SETTINGS ============================================================
fragdata_basename = 'cz_fragstat_150_' #'cz_fragstat_015_'
miscdata_basename = 'cz_miscstat_150_' #'cz_fragstat_015_'
fdc_basename = 'cz_fdc_150_'
datadir = f'/home/mweis/scratch/external/fragdata'

# ==== INPUT FILE SETTINGS =====================================================
#plotfile_include_patterns = ['*hdf5_plt*_??00',]
plotfile_include_patterns = ['*hdf5_chk_????',]
#plotfile_include_patterns = ['*hdf5_plt*_??00', '*hdf5_plt*_??20', '*hdf5_plt*_??40',
#                             '*hdf5_plt*_??60', '*hdf5_plt*_??80',] 


#===============================================================================
# ==== TECH HELPER =============================================================
#===============================================================================

def Pick(namespace, *names):
    return {name: namespace[name] for name in names}

#===============================================================================
# ==== STAT HELPER =============================================================
#===============================================================================

def InertiaSpectrum(x,y,z,mass):
    import numpy.linalg as la
    # Build Inertia Matrix
    I11 = np.sum(mass*(y**2 +z**2))
    I12 = np.sum(mass*(-x*y))
    I23 = np.sum(mass*(-y*z))
    I13 = np.sum(mass*(-x*z))
    I22 = np.sum(mass*(z**2 +x**2))
    I33 = np.sum(mass*(x**2 +y**2))
    I = np.array([[I11, I12, I13],
                    [I12, I22, I23],
                    [I13, I23, I33]])
    # Evaluate Inertia Matrix for normalized eigenvectors & eigenvalues
    # eigh output: w,v with:
    # w: The eigenvalues in ascending order, each repeated according to its multiplicity.
    # v: The column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i].
    Iw, Iv = la.eigh(I)
    return Iw, Iv

#===============================================================================
# ==== NODE STATISTICS: REGION INFO ============================================
#===============================================================================

def NodeStatReg(leafdata, octree, objmask, regmasks, ddict=None):
    # --- Check for each region if it intersects and fully contains the objmask
    intersecting_regflags = 0 # i.e. there is some overlap
    containing_regflags = 0 # i.e. the object is fully contained
    n_intersecting_regions = 0
    n_containing_regions = 0
    for ireg, regmask in enumerate(regmasks):
        overlap = regmask[objmask]
        if np.any(overlap):
            intersecting_regflags += (1<<ireg)
            n_intersecting_regions += 1
        if np.all(overlap):
            containing_regflags += (1<<ireg)
            n_containing_regions += 1
    # --- Refinement statistics
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    # Read refinement level
    rlevel = objcells('rlevel')
    obj_refine_rmin = np.max(rlevel)
    obj_refine_rmax = np.min(rlevel)
    rmax_global = np.max(leafdata['rlevel'])
    is_maxlevel = np.isclose(rlevel,rmax_global)
    mass = objcells('mass')
    vol = objcells('vol')
    massfraction_rmax = np.sum(mass*is_maxlevel)
    volfraction_rmax = np.sum(vol*is_maxlevel)
    # --- Pick up the results that we want to keep
    data = Pick(locals(),
                'intersecting_regflags', 'containing_regflags',
                'n_intersecting_regions', 'n_containing_regions',
                'obj_refine_rmin', 'obj_refine_rmax', 'rmax_global',
                'massfraction_rmax', 'volfraction_rmax',
                )
    if ddict is not None: ddict.update(data)
    return data


#===============================================================================
# ==== NODE STATISTICS: GRAVITATIONAL ==========================================
#===============================================================================

def PrepareHess(leafdata, octree, varname):
    fprint(f'Calculating hessian({varname})...', end='')
    var_grid = mflash.datagrid(leafdata[varname], octree, ng=2)
    var_hess, var_grad = var_grid.hessian(backdim=True, gradient=True)
    # Store calculated fields in leafdata cache
    leafdata.cache(f'{varname}_hess', var_hess)
    leafdata.cache(f'{varname}_grad', var_grad)
    fprint('OK')
    return None

def PrepareObjsGpot(leafdata, octree, objmasks, verbose=True):
    import cz_selfgrav as ptgr
    if verbose: fprint(f'Calculating gpot ({len(objmasks)} objects)...', end='')
    for iobj, objmask in enumerate(objmasks):
        # Prepare object gpot
        gpot_obj = ptgr.ObjGpotGlob(leafdata, octree, objmask, rsmooth_cells=2.5, verbose=False)
        assert gpot_obj.shape == leafdata.dshape()
        leafdata.cache(f'obj{iobj}_gpot', gpot_obj)
        if verbose: fprint(f' {iobj}', end='')
    if verbose: fprint(f' ...OK')
    return None

def PrepareObjsTidal(leafdata, octree, objmasks, verbose=True):
    nobj = len(objmasks)
    if verbose: fprint(f'Calculating object(i) tidal tensors at object(j) locations ({nobj} objects) (get some snacks)...', end='')
    for iobj in range(nobj):
        # Prepare hessian of object gpot
        gpot_obj = leafdata[f'obj{iobj}_gpot']
        gpot_obj_grid = mflash.datagrid(gpot_obj, octree, ng=2)
        gpot_obj_hessian, gpot_obj_gradient = gpot_obj_grid.hessian(backdim=True, gradient=True)
        for jobj, objmask in enumerate(objmasks):
            # Only store the tensor at all the objects locations, since we do not have the memory
            # to hold (nobj x gridsize x 9 entries/tensor) values.
            gpot_i_hessian_at_j = gpot_obj_hessian[objmask]
            gpot_i_gradient_at_j = gpot_obj_gradient[objmask]
            leafdata.cache(f'obj{iobj}_gpot_hess_atobj{jobj}', gpot_i_hessian_at_j)
            leafdata.cache(f'obj{iobj}_gpot_grad_atobj{jobj}', gpot_i_gradient_at_j)
        #
        if verbose: fprint(f' {iobj}', end='')
    if verbose: fprint(f' ...OK')
    return None

def NodeStatGravBound(objinfo, ddict=None):
    leafdata, octree, objmask, objcenter, objxyz, iobj, nobj = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    # Calculate kinetic volume energy, based on velocities relative to bulk motion
    mass = objcells('mass')
    velx, vely, velz = objcells('velx'), objcells('vely'), objcells('velz')
    velx_obj = np.average(velx, weights=mass)
    vely_obj = np.average(vely, weights=mass)
    velz_obj = np.average(velz, weights=mass)
    vx, vy, vz = velx-velx_obj, vely-vely_obj, velz-velz_obj
    v_squared = vx**2+vy**2+vz**2
    Ekin = .5*np.sum(mass*v_squared)
    # Calculate magnetic volume energy
    vol = objcells('vol')
    magx, magy, magz = objcells('magx'), objcells('magy'), objcells('magz')
    B_squared = magx**2+magy**2+magz**2
    Emag = 1./(8.*np.pi)*np.sum(vol*B_squared)
    # Calculate thermal volume energy
    Eth = np.sum(1.5*objcells('pres')*vol)
    # Calculate potential volume energy (global potential)
    gpot_glob = objcells('gpot')
    Egrav_glob = .5*np.sum(mass*gpot_glob)
    # Calculate potential volume energy (self-gravity)
    gpot_self = objcells(f'obj{iobj}_gpot')
    Egrav_self = .5*np.sum(mass*gpot_self)
    # Sum of self-binding energies
    Etot_self = Ekin+Emag+Eth+Egrav_self
    # Calculate simple 3D virial alpha
    alpha_vir = -2*Ekin/Egrav_self
    # Calculate mass concentration alpha_m, so that Egrav_self = -alpha_m*G*M^2/R,
    # which is a numerical factor that measures how strongly the mass is concentrated in the center.
    relx, rely, relz = objxyz
    rsq = relx**2 + rely**2 + relz**2
    clIrad = np.sqrt(np.average(rsq, weights=mass))
    clmass = np.sum(mass)
    alpha_m = -Egrav_self*clIrad/(cu.G_g*clmass**2)
    # Pick up the results that we want to keep
    data = Pick(locals(), 'Ekin', 'Emag', 'Eth', 'Egrav_glob', 'Egrav_self', 'Etot_self',
                'alpha_vir', 'alpha_m')
    if ddict is not None: ddict.update(data)
    return data

def _EVScale(eigvals, rtol=1e-3):
    # Calculate tolerance for deciding whether an eigenvalue is zero
    scale = np.nanmedian(np.max(np.abs(eigvals), axis=1))
    tol = rtol * scale if scale > 0 else rtol
    return tol

def _GTCountEV(Tg, mass, vol, rtol=1e-3):
    assert Tg.ndim == 3 # Tg shape: (Nobjcells,i,j)
    # --- Indicator 1: Count vol/mass fractions of tidal signs ---
    #Compute eigen values of Tg for every cell.
    eigvals = np.linalg.eigvalsh(Tg) # eigvals shape: (Nobjcells,3)
    assert eigvals.ndim == 2
    # Calculate the number of positive eigenvalues for each cell
    npos = np.sum(eigvals > _EVScale(eigvals, rtol=rtol), axis=1)
    # Calculate the fractions holding k=0/1/2/3 positive eigenvalues
    objmass = np.sum(mass)
    fmass_compressive = np.sum(mass[npos==0] / objmass)
    fmass_1dstretch = np.sum(mass[npos==1] / objmass)
    fmass_2dstretch = np.sum(mass[npos==2] / objmass)
    fmass_extensive = np.sum(mass[npos==3] / objmass)
    objvol = np.sum(vol)
    fvol_compressive = np.sum(vol[npos==0] / objvol)
    fvol_1dstretch = np.sum(vol[npos==1] / objvol)
    fvol_2dstretch = np.sum(vol[npos==2] / objvol)
    fvol_extensive = np.sum(vol[npos==3] / objvol)        
    # --- Pick up the results that we want to keep
    data = Pick(locals(),
                'fmass_compressive', 'fmass_1dstretch', 'fmass_2dstretch', 'fmass_extensive',
                'fvol_compressive', 'fvol_1dstretch', 'fvol_2dstretch', 'fvol_extensive',
                )
    return data

def _GTAverageEV(Tg, mass, vol):
    # --- Indicator 2: Large scale tidal field ---
    # Calculate the weighted average tensor
    Tg_massavg = np.sum(Tg*mass[:,None,None], axis=0) / np.sum(mass)
    Tg_volavg = np.sum(Tg*vol[:,None,None], axis=0) / np.sum(vol)
    eigvals_mass, eigvecs_mass = np.linalg.eigh(Tg_massavg)
    stretchdim_mass = np.sum(eigvals_mass > 0.) # XXX TOLERANCE?
    eigvals_vol, eigvecs_vol = np.linalg.eigh(Tg_volavg)
    stretchdim_vol = np.sum(eigvals_vol > 0.) # XXX TOLERANCE?
    # --- Pick up the results that we want to keep
    data = Pick(locals(),
                'eigvals_mass', 'eigvecs_mass', 'stretchdim_mass',
                'eigvals_vol', 'eigvecs_vol', 'stretchdim_vol',
                )
    return data

def _GTAverageAcc(accg, mass, vol):
    # Calculate the weighted average acceleration vector
    accg_vec_massavg = np.sum(accg*mass[:,None], axis=0) / np.sum(mass)
    accg_vec_volavg =  np.sum(accg*vol[:,None],  axis=0) / np.sum(vol)
    accg_abs_massavg = np.sqrt(np.sum(accg_vec_massavg**2, axis=-1))
    accg_abs_volavg =  np.sqrt(np.sum(accg_vec_volavg**2,  axis=-1))
    # --- Pick up the results that we want to keep
    data = Pick(locals(),
                'accg_vec_massavg', 'accg_vec_volavg', 'accg_abs_massavg', 'accg_abs_volavg',
                )
    return data

def _GTVirial(Tg, mass, vol, objxyz):
    objmass = np.sum(mass)
    # --- Indicator 3: Virial indicator ---
    # Calculate Wtide = sum_cells (m r*T*r)
    # Wtide > 0 increases Inertia, i.e. disruptive/stretching
    r = np.array(objxyz).T
    W_tide = np.einsum("n,ni,nij,nj->", mass, r, Tg, r)
    Irad_sq = np.average(np.sum(r**2,axis=1), weights=mass)
    # Tidal indicator om_rate_sq:
    # unit 1/s^2
    # om_rate_sq > 0  → globally disruptive / extensive
    # om_rate_sq < 0  → globally compressive
    # om_rate_sq ≈ 0  → balanced or mostly shear-like
    om_rate_sq = W_tide / (objmass*Irad_sq)
    # Tidal indicator q_tide:
    # q_tide ≳ 1  → external tide dynamically important
    # q_tide ≪ 1  → self-gravity/internal dynamics dominate
    # q_tide < 0  → external tide helps confine/compress
    q_tide = om_rate_sq / (cu.G_g*objmass/(Irad_sq**(1.5)))
    # --- Pick up the results that we want to keep
    data = Pick(locals(),
                'W_tide', 'Irad_sq', 'om_rate_sq', 'q_tide',
                )
    return data

def NodeStatGravAccel(objinfo, ddict=None, debug=False, rtol=1e-3):
    leafdata, octree, objmask, objcenter, objxyz, iobj, nobj = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    # Get global tidal tensor at object position
    accg_glob = -1.*objcells('gpot_grad')
    # Get self and outer tidal tensor at object position
    #accg_self = -1.*objcells(f'obj{iobj}_gpot_grad')
    accg_self = -1.*leafdata[f'obj{iobj}_gpot_grad_atobj{iobj}']
    accg_outer = accg_glob - accg_self
    # Get by-object tidal tensors at object position
    #accg_objs = [-1.*objcells(f'obj{j}_gpot_grad') for j in range(nobj)]
    accg_objs = [-1.*leafdata[f'obj{j}_gpot_grad_atobj{iobj}'] for j in range(nobj)]
    #
    mass = objcells('mass')
    vol = objcells('vol')
    data = dict()
    # Acceleration from total outer field and global field
    data.update(AddKeyPrefix(_GTAverageAcc(accg_outer, mass, vol), 'outer_'))
    data.update(AddKeyPrefix(_GTAverageAcc(accg_glob, mass, vol), 'glob_'))
    # Per-object acceleration
    for jobj,accg_j in enumerate(accg_objs):
        data.update(AddKeyPrefix(_GTAverageAcc(accg_j, mass, vol), f'obj{jobj}_'))
    # Self-acceleration (For checking consistency, this one is supposed to vanish)
    data.update(AddKeyPrefix(_GTAverageAcc(accg_self, mass, vol), 'self_'))
    #
    if ddict is not None: ddict.update(data)
    return data

def NodeStatGravTide(objinfo, ddict=None, debug=False, rtol=1e-3):
    leafdata, octree, objmask, objcenter, objxyz, iobj, nobj = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    # Get global tidal tensor at object position
    Tg_glob = -1.*objcells('gpot_hess')
    # Get self and outer tidal tensor at object position
    #Tg_self = -1.*objcells(f'obj{iobj}_gpot_hess')
    Tg_self = -1.*leafdata[f'obj{iobj}_gpot_hess_atobj{iobj}']
    Tg_outer = Tg_glob - Tg_self
    # Get by-object tidal tensors at object position
    #Tg_objs = [-1.*objcells(f'obj{j}_gpot_hess') for j in range(nobj)]
    Tg_objs = [-1.*leafdata[f'obj{j}_gpot_hess_atobj{iobj}'] for j in range(nobj)]
    #
    mass = objcells('mass')
    vol = objcells('vol')
    data = dict()
    # Characterize total outer field (this should usually be used to measure the tidal
    # influence of the outer mass distribution)
    data.update(AddKeyPrefix(_GTCountEV(Tg_outer, mass, vol), 'outer_'))
    data.update(AddKeyPrefix(_GTAverageEV(Tg_outer, mass, vol), 'outer_'))
    data.update(AddKeyPrefix(_GTVirial(Tg_outer, mass, vol, objxyz), 'outer_'))
    # Characterize per-object outer field (this should usually be used to measure the tidal
    # influence of the outer mass distribution)
    for jobj,Tg_j in enumerate(Tg_objs):
        data.update(AddKeyPrefix(_GTCountEV(Tg_j, mass, vol), f'obj{jobj}_'))
        data.update(AddKeyPrefix(_GTAverageEV(Tg_j, mass, vol), f'obj{jobj}_'))
        data.update(AddKeyPrefix(_GTVirial(Tg_j, mass, vol, objxyz), f'obj{jobj}_'))
    # Characterize self (For checking / measuring quaility of core detection)
    data.update(AddKeyPrefix(_GTCountEV(Tg_self, mass, vol), 'self_'))
    data.update(AddKeyPrefix(_GTAverageEV(Tg_self, mass, vol), 'self_'))
    data.update(AddKeyPrefix(_GTVirial(Tg_self, mass, vol, objxyz), 'self_'))
    # Characterize global field (still here for traditional reasons, usefulness questionable)
    data.update(AddKeyPrefix(_GTCountEV(Tg_glob, mass, vol), 'glob_'))
    data.update(AddKeyPrefix(_GTAverageEV(Tg_glob, mass, vol), 'glob_'))
    data.update(AddKeyPrefix(_GTVirial(Tg_glob, mass, vol, objxyz), 'glob_'))
    #
    if ddict is not None: ddict.update(data)
    return data


#===============================================================================
# ==== NODE STATISTICS: MISC ===================================================
#===============================================================================

def NodeStatMisc(leafdata, octree, objm, regmasks):
    miscdata = dict()
    NodeStatReg(leafdata, octree, objm, regmasks, ddict=miscdata)
    return miscdata

def miscdata_main(leafdata, octree, objmasks, regmasks, debug=True):
    # Every statistic for which sink mapping is irrelevant may go here!
    objmasks_exclusive = ExclusiveNodeMasks(objmasks)
    fprint(f'...Computing inclusive node statistics...', end='')
    nodestat_inclusive = [NodeStatMisc(leafdata, octree, objm, regmasks) for objm in objmasks]
    fprint('OK')
    fprint(f'...Computing exclusive node statistics...', end='')
    nodestat_exclusive = [NodeStatMisc(leafdata, octree, objm, regmasks) for objm in objmasks_exclusive]
    fprint('OK')
    assert len(nodestat_inclusive) == len(nodestat_exclusive)
    # Package the data from list(dict) to dict(array).
    # While at it, pack inclusive and exclusive data in the same dict,
    # prefixing the exclusive data keys with excl_.
    arrdict_inclusive = RepackageResult(nodestat_inclusive)
    arrdict_exclusive = RepackageResult(nodestat_exclusive)
    data_out = arrdict_inclusive | AddKeyPrefix(arrdict_exclusive, 'excl')
    return data_out

def sinkregdata_main(leafdata, octree, regmasks, debug=True):
    if not regmasks:
        print('WARNING! No zoom regions detected, skipping region sink data')
    sinkdict = cz.SinkData(leafdata)
    cz.BG_SinkMass(leafdata, octree, sinkdict)
    objinfo = lambda regmask: (leafdata, octree, regmask, None, None)
    if regmasks:
        if debug: fprint(f'...Computing region sink statistics...', end='')
        sinkstat_reg = [NodeStatSinks(objinfo(regmask)) for regmask in regmasks]
        arrdict_sinkreg = RepackageResult(sinkstat_reg)
        if debug: fprint('OK')
    else:
        arrdict_sinkreg = dict()
    if debug: fprint(f'...Computing global sink statistics...', end='')
    domainmask = np.ones(leafdata.dshape(), dtype=bool)
    sinkstat_glob = [NodeStatSinks(objinfo(domainmask)),]
    arrdict_sinkglob = RepackageResult(sinkstat_glob)
    if debug: fprint('OK')
    arrdict_srg = AddKeyPrefix(arrdict_sinkreg, 'reg') | AddKeyPrefix(arrdict_sinkglob, 'glob')
    return arrdict_srg

def determine_miscdata(leafdata, octree, objmasks):
    # --- Compute node statistics, ignoring sinks
    fprint(f'Computing node misc. data: ({len(objmasks)} nodes):')
    if len(objmasks)==0: return dict() # No objects? We are done!
    # Open region data
    simdir = leafdata.dirname().removesuffix('/out')
    try:
        regtable = cz.RegionTable(simdir, domainreg=False)
        stepreg = cz.StepRegions(leafdata, regtable)
        regmasks = [cz.RegionMask(leafdata, octree, rd) for rd in stepreg]
    except OSError:
        regmasks = list()
    # Note: since this uses the global potential from file,
    # the sink particle mapping would be pointless here
    miscdata = miscdata_main(leafdata, octree, objmasks, regmasks)
    regdata = sinkregdata_main(leafdata, octree, regmasks)
    miscdata.update(regdata)
    return miscdata


#===============================================================================
# ==== NODE STATISTICS: VIRIAL SURFACE TERMS ===================================
#===============================================================================

def PrepareGuardVars(leafdata, octree):
    fprint('Calculating guarded fields... ', end='')
    gcdata = dict()
    # Note that momx = dens*velx (specific momentum),
    # which per-block is a conserved quantity (since vol is constant per-block),
    # and as such is also suitable for per-block interpolation.
    fieldvars = ('pres','dens','momx','momy','momz','magx','magy','magz')
    for var in fieldvars:
        vardata = leafdata[var]
        gcdata[var] = ReconstructGuardcells(leafdata, octree, vardata, ng=1)
        fprint(f'{var} ', end='')
        sys.stdout.flush
    fprint('OK')
    return gcdata

def ReconstructGuardcells(leafdata, octree, data, ng=1):
    assert data.shape == leafdata.dshape()
    datagrid = mflash.datagrid(data, octree, ng=ng)
    return datagrid.gbld

def LRFaceIndicies(fccmask):
    '''
    Pre-calculate the cell indicies needed to calculate the LR-averaged face
    values for the given set of faces
    '''
    # --- Extract the indicies of the faces marked by the fccmask.
    # Each cell face is located by the cell indicies (bl,k,j,i),
    # followed by the face axis ax (0|1|2 equating x|y|z) and the face
    # orientation (0|1 equating to left|right, ie.
    # towards the next lower|higher index).
    bl, k,j,i, ax,lr = np.where(fccmask)
    nfaces = len(bl) #np.sum(fccmask)
    fc = np.arange(nfaces)
    # --- Calculate cell indicies of the cells left of each given face, respectively ---
    # First offset i,j,k by one so that those subgrid indicies point to the corresponding
    # cell when accounting for one guard cell layer in the input fields:
    ijk_left = np.array((i,j,k), dtype=np.int16) + 1
    # The cell left of the face is now the next-left cell in case of a left face,
    # otherwise it is the cell that the index already points to.
    # Note that we separated out above guard cell offset calculation since now
    # we touch only the index corresponding to the axis direction (selected by ax index).
    ijk_left[ax,fc] += lr-1
    # --- Calculate cell indicies of the cells left of each given face, respectively ---
    # Analogous for the cell right of the face:
    ijk_right = np.array((i,j,k), dtype=np.int16) + 1
    ijk_right[ax,fc] = ijk_left[ax,fc] + 1
    # Package and return indicies
    fccind = bl, ijk_left, ijk_right
    return fccind

def FaceValues(data_1gc, fccind, weights_1gc=None):
    # --- Decode pre-computed Indicies
    bl, ijk_left, ijk_right = fccind
    # --- Calculate the weighted average of the left and right cell ---
    i0,j0,k0 = ijk_left
    i1,j1,k1 = ijk_right
    if weights_1gc is None:
        faceval = (data_1gc[bl,k0,j0,i0]+data_1gc[bl,k1,j1,i1])*.5
    else:
        val0, val1 = data_1gc[bl,k0,j0,i0], data_1gc[bl,k1,j1,i1]
        w0, w1 = weights_1gc[bl,k0,j0,i0], weights_1gc[bl,k1,j1,i1]
        faceval = (val0*w0+val1*w1)/(w0+w1)
    return faceval

def FaceNormals(leafdata, octree, fccmask):
    ''' Calculate the direction of outward pointing cell face normals '''
    fccn_shape = (3,2)
    fcc_nx = np.zeros(fccn_shape)
    fcc_ny = np.zeros(fccn_shape)
    fcc_nz = np.zeros(fccn_shape)
    fcc_nx[0,0] = -1.
    fcc_nx[0,1] =  1.
    fcc_ny[1,0] = -1.
    fcc_ny[1,1] =  1.
    fcc_nz[2,0] = -1.
    fcc_nz[2,1] =  1.
    objfacedir = np.where(fccmask)[4:]
    face_normal_x = fcc_nx[objfacedir]
    face_normal_y = fcc_ny[objfacedir]
    face_normal_z = fcc_nz[objfacedir]
    return face_normal_x, face_normal_y, face_normal_z

def FaceAreas(leafdata, octree, fccmask):
    vcc_shape = leafdata.dshape()
    fcc_shape = vcc_shape+(3,2)
    lenx = leafdata['lenx']
    leny = leafdata['leny']
    lenz = leafdata['lenz']
    Ax, Ay, Az = leny*lenz, lenz*lenx, lenx*leny
    fcc_A = np.zeros(fcc_shape)
    fcc_A[...,0,:] = Ax[...,None]
    fcc_A[...,1,:] = Ay[...,None]
    fcc_A[...,2,:] = Az[...,None]
    face_area = fcc_A[fccmask]
    return face_area

def FaceRadials(leafdata, octree, objcenter, fccind):
    relx_1gc, rely_1gc, relz_1gc = cz.BG_RelCoords(octree, objcenter, ng=1, force=True)
    face_relx = FaceValues(relx_1gc, fccind)
    face_rely = FaceValues(rely_1gc, fccind)
    face_relz = FaceValues(relz_1gc, fccind)
    return face_relx, face_rely, face_relz

def OuterFaceMask(leafdata, octree, volmask, debug=True):
    '''
    Mark all per-block faces of the object marked by the volume mask volmask.
    CAUTION: Note that this has two downsides:
    a) ReconstructGuardcells is technically not correct at coarse-fine block boundaries.
        The completely correct solution here would be to calculate a face coverage factor
        for each selected face.
        However, since the number of faces will be large anyways I hope this to be
        an acceptable approximation.
    b) This is already slow as-is, which also makes reconciling above point all but unfeasible.
    '''
    assert np.any(volmask)
    vcc_shape = leafdata.dshape()
    fcc_shape = vcc_shape+(3,2)
    volmask_1gc_f = ReconstructGuardcells(leafdata, octree, 1.*volmask, ng=1)
    volmask_1gc = volmask_1gc_f > .5
    fcc_surfmask = np.zeros(fcc_shape, dtype=bool)
    fcc_surfmask[...,0,0] = volmask_1gc[:,1:-1,1:-1,1:-1] * ~volmask_1gc[:,1:-1,1:-1,0:-2]
    fcc_surfmask[...,0,1] = volmask_1gc[:,1:-1,1:-1,1:-1] * ~volmask_1gc[:,1:-1,1:-1,2:  ]
    fcc_surfmask[...,1,0] = volmask_1gc[:,1:-1,1:-1,1:-1] * ~volmask_1gc[:,1:-1,0:-2,1:-1]
    fcc_surfmask[...,1,1] = volmask_1gc[:,1:-1,1:-1,1:-1] * ~volmask_1gc[:,1:-1,2:  ,1:-1]
    fcc_surfmask[...,2,0] = volmask_1gc[:,1:-1,1:-1,1:-1] * ~volmask_1gc[:,0:-2,1:-1,1:-1]
    fcc_surfmask[...,2,1] = volmask_1gc[:,1:-1,1:-1,1:-1] * ~volmask_1gc[:,2:  ,1:-1,1:-1]
    n_cells = np.sum(volmask)
    n_faces = np.sum(fcc_surfmask)
    if debug:
        fprint(f'#c{n_cells}#f{n_faces}',end='')
    return fcc_surfmask

def OuterFaceMaskSimple(leafdata, octree, volmask, debug=False):
    '''
    Mark all per-block faces of the object marked by the volume mask volmask.
    CAUTION: Note that this version is much faster, but will also mark
             all contained cell faces at block boundaries.
             This is fine for integration over scalar products of the kind
             (v,n) with the oriented boundary normal n, but will not work
             when used in other calculations, e.g. face-area or abs(face_flux).
    '''
    assert np.any(volmask)
    vcc_shape = leafdata.dshape()
    fcc_shape = vcc_shape+(3,2)
    fcc_surfmask = np.zeros(fcc_shape, dtype=bool)
    untouched = np.ones(fcc_shape, dtype=bool)
    # Set inner volume by checking against neighbouring cells
    fcc_surfmask[:, :  , :  ,1:  ,0,0] = volmask[:, :  , :  ,1:  ] * ~volmask[:, :  , :  , :-1]
    fcc_surfmask[:, :  , :  , :-1,0,1] = volmask[:, :  , :  , :-1] * ~volmask[:, :  , :  ,1:  ]
    fcc_surfmask[:, :  ,1:  , :  ,1,0] = volmask[:, :  ,1:  , :  ] * ~volmask[:, :  , :-1, :  ]
    fcc_surfmask[:, :  , :-1, :  ,1,1] = volmask[:, :  , :-1, :  ] * ~volmask[:, :  ,1:  , :  ]
    fcc_surfmask[:,1:  , :  , :  ,2,0] = volmask[:,1:  , :  , :  ] * ~volmask[:, :-1, :  , :  ]
    fcc_surfmask[:, :-1, :  , :  ,2,1] = volmask[:, :-1, :  , :  ] * ~volmask[:,1:  , :  , :  ]
    # Treat block boundaries by checking if the touching cell is included in the object
    fcc_surfmask[:, :, :, 0,0,0] = volmask[:, :, :, 0]
    fcc_surfmask[:, :, :,-1,0,1] = volmask[:, :, :,-1]
    fcc_surfmask[:, :, 0, :,1,0] = volmask[:, :, 0, :]
    fcc_surfmask[:, :,-1, :,1,1] = volmask[:, :,-1, :]
    fcc_surfmask[:, 0, :, :,2,0] = volmask[:, 0, :, :]
    fcc_surfmask[:,-1, :, :,2,1] = volmask[:,-1, :, :]
    # Sanity check
    n_cells = np.sum(volmask)
    n_faces = np.sum(fcc_surfmask)
    assert n_faces > 0
    if debug:
        fprint(f'#c{n_cells}#f{n_faces}',end='')
    return fcc_surfmask

# We want to weigh variables in a way that averaging is done on conserved values.
# Note however that the volume of all cells used in subgrid averaging is equal,
# so e.g. using uniform weights instead of vol and density instead of mass is sufficient.
# If a var is not listed uniform weighting will be used.
SurfWeighVar = {'velx':'dens', 'vely':'dens', 'velz':'dens'}

def _SurfCountStat(Objfaces, face_data, vel_obj, v_partition=None, v_slow=2e+4, v_med=1e+5, v_fast=5e+5, debug=True):
    if v_partition is None:
        v_partition = [v_slow,v_med,v_fast]
    # Unpack data
    face_nornal, face_radial, face_area = face_data
    face_normal_x, face_normal_y, face_normal_z = face_nornal
    face_radial_x, face_radial_y, face_radial_z = face_radial
    velx_obj, vely_obj, velz_obj = vel_obj
    # Calculate scalar product (vec(r)*d_vec(A))
    face_rproj = face_normal_x*face_radial_x + face_normal_y*face_radial_y + face_normal_z*face_radial_z
    RdA = face_area * face_rproj
    # --- Calculate flow direction face fractions (bulk motion corrected)
    # Calculate Eulerian velocities at faces
    face_dens = Objfaces('dens')
    face_momx = Objfaces('momx')
    face_momy = Objfaces('momy')
    face_momz = Objfaces('momz')
    face_velx = face_momx / face_dens
    face_vely = face_momy / face_dens
    face_velz = face_momz / face_dens
    # Calculate bulk motion corrected velocities at faces
    face_relvx = face_velx - velx_obj
    face_relvy = face_vely - vely_obj
    face_relvz = face_velz - velz_obj
    # Calculate mass flux through faces (outwards positive)
    face_relv_out = face_relvx*face_normal_x + face_relvy*face_normal_y + face_relvz*face_normal_z
    face_mfluxdens_bulk = face_relv_out*face_dens
    face_mflux_bulk = face_mfluxdens_bulk * face_area
    # Sort velocities according to thresholds
    face_relv_partition = [-v for v in v_partition[::-1]]+[0,]+[v for v in v_partition]
    face_relv_nbins = len(face_relv_partition)+1
    face_relv_bins = np.searchsorted(face_relv_partition, face_relv_out)
    # Count faces per class unweighted, area weighted and flow weighted
    face_relv_counts = np.bincount(face_relv_bins, minlength=face_relv_nbins)
    face_relv_areas = np.bincount(face_relv_bins, weights=face_area, minlength=face_relv_nbins)
    face_relv_flows = np.bincount(face_relv_bins, weights=face_mflux_bulk, minlength=face_relv_nbins)
    # Calculate fractions
    face_relv_countfrac = face_relv_counts / np.sum(face_relv_counts)
    face_relv_areafrac = face_relv_areas / np.sum(face_relv_areas)
    face_relv_flowfrac = face_relv_flows / np.sum(face_relv_flows) # Note that this should still sum to 1, but will contain negative values
    # --- Pick up the results that we want to keep
    data = Pick(locals(),
                'face_relv_counts', 'face_relv_areas', 'face_relv_flows',
                'face_relv_countfrac', 'face_relv_areafrac', 'face_relv_flowfrac',
                )
    return data
    

def _SurfNormStat(Objfaces, face_data, vel_obj, debug=True):
    # Unpack data
    face_nornal, face_radial, face_area = face_data
    face_normal_x, face_normal_y, face_normal_z = face_nornal
    face_radial_x, face_radial_y, face_radial_z = face_radial
    velx_obj, vely_obj, velz_obj = vel_obj
    # Calculate scalar product (vec(r)*d_vec(A))
    face_rproj = face_normal_x*face_radial_x + face_normal_y*face_radial_y + face_normal_z*face_radial_z
    RdA = face_area * face_rproj
    # --- Calculate thermal surface term
    face_pres = Objfaces('pres')
    EthS_amr = .5*np.sum(face_pres*RdA)
    # --- Calculate kinetic surface term
    face_dens = Objfaces('dens')
    face_momx = Objfaces('momx')
    face_momy = Objfaces('momy')
    face_momz = Objfaces('momz')
    face_Kxx = .5*face_momx**2 / face_dens
    face_Kyy = .5*face_momy**2 / face_dens
    face_Kzz = .5*face_momz**2 / face_dens
    face_Kxy = .5*face_momx*face_momy / face_dens
    face_Kxz = .5*face_momx*face_momz / face_dens
    face_Kyz = .5*face_momy*face_momz / face_dens
    EKinS_amr = np.sum(face_area*(
        +face_radial_x*face_Kxx*face_normal_x
        +face_radial_x*face_Kxy*face_normal_y
        +face_radial_x*face_Kxz*face_normal_z
        +face_radial_y*face_Kxy*face_normal_x
        +face_radial_y*face_Kyy*face_normal_y
        +face_radial_y*face_Kyz*face_normal_z
        +face_radial_z*face_Kxz*face_normal_x
        +face_radial_z*face_Kyz*face_normal_y
        +face_radial_z*face_Kzz*face_normal_z
    ))
    if debug: fprint('*', end='')
    # --- Calculate magnetic surface term
    face_magx = Objfaces('magx')
    face_magy = Objfaces('magy')
    face_magz = Objfaces('magz')
    face_Txx = (1./(4.*np.pi))*.5*face_magx**2
    face_Tyy = (1./(4.*np.pi))*.5*face_magy**2
    face_Tzz = (1./(4.*np.pi))*.5*face_magz**2
    face_Txy = (1./(4.*np.pi))*face_magx*face_magy
    face_Txz = (1./(4.*np.pi))*face_magx*face_magz
    face_Tyz = (1./(4.*np.pi))*face_magy*face_magz
    EMagS_amr = np.sum(face_area*(
        +face_radial_x*face_Txx*face_normal_x
        +face_radial_x*face_Txy*face_normal_y
        +face_radial_x*face_Txz*face_normal_z
        +face_radial_y*face_Txy*face_normal_x
        +face_radial_y*face_Tyy*face_normal_y
        +face_radial_y*face_Tyz*face_normal_z
        +face_radial_z*face_Txz*face_normal_x
        +face_radial_z*face_Tyz*face_normal_y
        +face_radial_z*face_Tzz*face_normal_z
    ))
    if debug: fprint('*', end='')
    # --- Calculate surface area (direct, radially projected)
    face_R = np.sqrt(face_radial_x**2+face_radial_y**2+face_radial_z**2)
    face_cos = face_rproj / face_R
    faceS_proj = np.sum(face_cos*face_area)
    if debug: fprint('*', end='')
    # --- Calculate eulerian accretion rate
    face_velx = face_momx / face_dens
    face_vely = face_momy / face_dens
    face_velz = face_momz / face_dens
    face_vel_out = face_velx*face_normal_x + face_vely*face_normal_y + face_velz*face_normal_z
    face_mfluxdens_euler = face_vel_out*face_dens
    accrateS_euler = -1.*np.sum(face_mfluxdens_euler*face_area)
    if debug: fprint('*', end='')
    # --- Calculate accretion rate (bulk motion corrected)
    # Calculate corrected velocities at faces
    face_relvx = face_velx - velx_obj
    face_relvy = face_vely - vely_obj
    face_relvz = face_velz - velz_obj
    # Calculate mass flux through faces
    face_relv_out = face_relvx*face_normal_x + face_relvy*face_normal_y + face_relvz*face_normal_z
    face_mfluxdens_bulk = face_relv_out*face_dens
    accrateS_bulk = -1.*np.sum(face_mfluxdens_bulk*face_area)
    if debug: fprint('*', end='')
    # --- Calculate rotational inertia accretion (I-flux) through surface
    IfluxS_euler = np.sum(face_dens*face_R**2*face_vel_out*face_area)
    IfluxS_bulk = np.sum(face_dens*face_R**2*face_relv_out*face_area)
    if debug: fprint('*', end='')
    #diag_nfaces = len(fccind[0])
    # --- Pick up the results that we want to keep
    data = Pick(locals(), 'EthS_amr', 'EKinS_amr', 'EMagS_amr',
                'faceS_proj',
                'accrateS_euler', 'accrateS_bulk',
                'IfluxS_euler', 'IfluxS_bulk',)
    return data

def _FaceData(leafdata, octree, fccmask, fccind, objcenter, debug=True):
    # Calculate face geometry data
    face_area = FaceAreas(leafdata, octree, fccmask)
    face_normal_x, face_normal_y, face_normal_z = FaceNormals(leafdata, octree, fccmask)
    face_radial_x, face_radial_y, face_radial_z = FaceRadials(leafdata, octree, objcenter, fccind)
    if debug: fprint('.', end='')
    # Package face geometry data
    face_normal = face_normal_x, face_normal_y, face_normal_z
    face_radial = face_radial_x, face_radial_y, face_radial_z
    face_data = face_normal, face_radial, face_area
    #
    return face_data


def NodeStatSurface(objinfo, ddict=None, gcdata=dict(), debug=False):
    leafdata, octree, objmask, objcenter, objxyz = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    # Calculate center of mass velocity
    mass = objcells('mass')
    velx, vely, velz = objcells('velx'), objcells('vely'), objcells('velz')
    velx_obj = np.average(velx, weights=mass)
    vely_obj = np.average(vely, weights=mass)
    velz_obj = np.average(velz, weights=mass)
    vel_obj = velx_obj, vely_obj, velz_obj

    data = dict()

    ## XXX WARNING: PROBABLE BUG IN OuterFaceMaskSimple! #
    ## XXX ## Calculate simplified face values (also view block bounds as faces)
    ## XXX ## Note that this only works for value*normal style sums.
    ## XXX # Get face cell mask
    ## XXX # CAUTION: PLEASE NOTE THE COMMENT ON THE LIMITATION OF THE OuterFaceMaskSimple FUNCTION!
    ## XXX fccmask_simple = OuterFaceMaskSimple(leafdata, octree, objmask)
    ## XXX if debug: fprint('.', end='')
    ## XXX # Define field reader that only returns the object cells
    ## XXX fccind_simple = LRFaceIndicies(fccmask_simple)
    ## XXX fccdata_simple = _FaceData(leafdata, octree, fccmask_simple, fccind_simple, objcenter)
    ## XXX #
    ## XXX def Objfaces_simple(var):
    ## XXX     data_1gc = gcdata[var]
    ## XXX     wvar = SurfWeighVar.get(var,None)
    ## XXX     weights_1gc = gcdata[wvar] if wvar is not None else None
    ## XXX     return FaceValues(data_1gc, fccind_simple, weights_1gc=weights_1gc)
    ## XXX snstat_simple = _SurfNormStat(Objfaces_simple, fccdata_simple, vel_obj, debug=True)
    ## XXX data.update(snstat_simple)
    ## XXX if debug: fprint('.', end='')

    ## Calculate full face values (also view block bounds as faces)
    ## Note that this only works for value*normal style sums.
    # Get face cell mask
    # CAUTION: PLEASE NOTE THE COMMENT ON THE LIMITATION OF THE OuterFaceMaskSimple FUNCTION!
    fccmask_full = OuterFaceMask(leafdata, octree, objmask)
    if debug: fprint('.', end='')
    # Define field reader that only returns the object cells
    fccind_full = LRFaceIndicies(fccmask_full)
    fccdata_full = _FaceData(leafdata, octree, fccmask_full, fccind_full, objcenter)
    #
    def Objfaces_full(var):
        data_1gc = gcdata[var]
        wvar = SurfWeighVar.get(var,None)
        weights_1gc = gcdata[wvar] if wvar is not None else None
        return FaceValues(data_1gc, fccind_full, weights_1gc=weights_1gc)
    snstat_full = _SurfNormStat(Objfaces_full, fccdata_full, vel_obj, debug=True)
    data.update(AddKeyPrefix(snstat_full, 'snfull'))
    if debug: fprint('.', end='')
    snstat_counts = _SurfCountStat(Objfaces_full, fccdata_full, vel_obj, debug=True)
    data.update(AddKeyPrefix(snstat_counts, 'snfull'))
    if debug: fprint('.', end='')

    # --- Pick up the results that we want to keep
    if ddict is not None: ddict.update(data)
    return data


#===============================================================================
# ==== NODE STATISTICS =========================================================
#===============================================================================

def NodeStatSinks(objinfo, ddict=None, debug=False):
    leafdata, octree, objmask, objcenter, objxyz = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    # Calculate total sink mass and number of sink cells in object
    msink = objcells('msink')
    sinkmass = np.sum(msink)
    sinkcells = np.sum(msink>0.)
    # Calculate actual number of sinks in object
    # (Note that two sinks might occupy the same cell e.g. by dynamically passing there)
    sinkcount = np.sum(objcells('nsink'))
    # Pick up the results that we want to keep
    data = Pick(locals(), 'sinkmass', 'sinkcells', 'sinkcount')
    if ddict is not None: ddict.update(data)
    return data

def NodeStatAngmom(objinfo, ddict=None, debug=False):
    leafdata, octree, objmask, objcenter, objxyz = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    # Unpack coordinates of object cells
    x, y, z = objxyz
    rsq = x**2 + y**2 + z**2
    r = np.maximum(1e-40,np.sqrt(rsq))
    nx, ny, nz = x/r, y/r, z/r
    # Calculate velocities relative to center of mass
    mass = objcells('mass')
    velx, vely, velz = objcells('velx'), objcells('vely'), objcells('velz')
    velx_obj = np.average(velx, weights=mass)
    vely_obj = np.average(vely, weights=mass)
    velz_obj = np.average(velz, weights=mass)
    vx, vy, vz = velx-velx_obj, vely-vely_obj, velz-velz_obj
    # Evaluate Inertia Matrix for normalized eigenvectors & eigenvalues
    # eigh output: w,v with:
    # w: The eigenvalues in ascending order, each repeated according to its multiplicity.
    # v: The column v[:, i] is the normalized eigenvector corresponding to the eigenvalue w[i].
    Iw, Iv = InertiaSpectrum(x,y,z, mass)
    clIvec0, clIval0 = Iv[:,0], Iw[0]
    clIvec1, clIval1 = Iv[:,1], Iw[1]
    clIvec2, clIval2 = Iv[:,2], Iw[2]
    # Determine angular velocities (cartesian, per cell)
    omx, omy, omz = (1./r**2)*np.cross([x,y,z], [vx, vy, vz], axis=0)
    Ic = mass*r**2
    # Determine total angular momenta (cartesian)
    clLx, clLy, clLz = np.sum(Ic*omx), np.sum(Ic*omy), np.sum(Ic*omz)
    clLc = np.sqrt(clLx**2 +clLy**2 +clLz**2)
    # Determine angular velocities (principal, per cell)
    om0 = Iv[0,0]*omx +Iv[1,0]*omy +Iv[2,0]*omz
    om1 = Iv[0,1]*omx +Iv[1,1]*omy +Iv[2,1]*omz
    om2 = Iv[0,2]*omx +Iv[1,2]*omy +Iv[2,2]*omz
    # Determine total angular momenta (principal)
    clL0, clL1,clL2 = np.sum(Ic*om0), np.sum(Ic*om1), np.sum(Ic*om2)
    clLp = np.sqrt(clL0**2 +clL1**2 +clL2**2)
    # Determine pincipal axis velocities
    v0 = Iv[0,0]*vx +Iv[1,0]*vy +Iv[2,0]*vz
    v1 = Iv[0,1]*vx +Iv[1,1]*vy +Iv[2,1]*vz
    v2 = Iv[0,2]*vx +Iv[1,2]*vy +Iv[2,2]*vz
    # Determine principal component velocity dispersion
    cldisp0 = np.sqrt(np.average(v0**2, weights=mass))
    cldisp1 = np.sqrt(np.average(v1**2, weights=mass))
    cldisp2 = np.sqrt(np.average(v2**2, weights=mass))
    # Determine average angular speeds
    om0_obj = np.average(om0, weights=mass)
    om1_obj = np.average(om1, weights=mass)
    om2_obj = np.average(om2, weights=mass)
    # Determine principal component angular velocity dispersion
    clomdisp0 = np.sqrt(np.average(om0**2, weights=mass) -om0_obj**2)
    clomdisp1 = np.sqrt(np.average(om1**2, weights=mass) -om1_obj**2)
    clomdisp2 = np.sqrt(np.average(om2**2, weights=mass) -om2_obj**2)
    # Determine total angular momentum direction
    clLabs = max(1e-40,clLc)
    clLnx, clLny, clLnz = clLx/clLabs, clLy/clLabs, clLz/clLabs
    # Calculate cylinder directions (per cell) (XXX CHECK NRX,NRY,NRZ AFTER GETTING SOME SLEEP! XXX)
    nhx, nhy, nhz = clLnx, clLny, clLnz # Height direction (=direction of angmom vector)
    hv = nx*nhx + ny*nhy + nz*nhz # Height component value of coordinate direction
    i = np.argmax(hv**2)
    if debug: print('#D Ni', nx[i], ny[i], nz[i])
    if debug: print('#D NH', nhx, nhy, nhz)
    assert np.all(hv**2<=1.0001)
    hx, hy, hz = hv*nhx, hv*nhy, hv*nhz # Height component vector
    rx, ry, rz = nx-hx, ny-hy, nz-hz # Axis distance component vector direction
    rv = np.maximum(1e-40,np.sqrt(np.maximum(0.,1.-hv**2))) # Axis distance component value of coordinate direction
    assert np.all(rv**2<=1.0001)
    nrx, nry, nrz = rx/rv, ry/rv, rz/rv # Axis distance direction (coordinate direction - its height projection)
    if debug: print('#D NRi', nrx[i], nry[i], nrz[i])
    npx, npy, npz = np.cross([nhx,nhy,nhz], [nrx,nry,nrz], axis=0)
    if debug: print('#D NPi', npx[i], npy[i], npz[i])
    # Sanity check (did we get an orthogonal system?)
    assert np.all(np.isclose(nhx*nrx+nhy*nry+nhz*nrz, 0.))
    assert np.all(np.isclose(nrx*npx+nry*npy+nrz*npz, 0.))
    assert np.all(np.isclose(npx*nhx+npy*nhy+npz*nhz, 0.))
    # Calculate velocity (cylindrical coordinates)
    vh = vx*nhx + vy*nhy + vz*nhz # Height velocity
    vr = vx*nrx + vy*nry + vz*nrz # Axis distance velocity
    vp = vx*npx + vy*npy + vz*npz # Angular velocity
    # Calculate mass weighted velocity dispersion (cylindrical)
    cldispvcylh = np.sqrt(np.average(vh**2, weights=mass))
    cldispvcylr = np.sqrt(np.average(vr**2, weights=mass))
    cldispvcylp = np.sqrt(np.average(vp**2, weights=mass))
    cldispvcyl = np.sqrt(cldispvcylh**2 +cldispvcylr**2 +cldispvcylp**2)
    # Pick up the results that we want to keep
    data = Pick(locals(), 
                'clIvec0', 'clIvec1', 'clIvec2', 'clIval0', 'clIval1', 'clIval2',
                'clLx', 'clLy', 'clLz', 'clLc', 'clLnx', 'clLny', 'clLnz',
                'clL0', 'clL1','clL2', 'clLp', 'om0_obj', 'om1_obj', 'om2_obj',
                'cldisp0', 'cldisp1', 'cldisp2', 'clomdisp0', 'clomdisp1', 'clomdisp2',
                'cldispvcylh', 'cldispvcylr', 'cldispvcylp', 'cldispvcyl')
    if ddict is not None: ddict.update(data)
    return data

def NodeStatVelDispersion(objinfo, ddict=None):
    leafdata, octree, objmask, objcenter, objxyz = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    # Calculate velocities relative to center of mass
    mass = objcells('mass')
    velx, vely, velz = objcells('velx'), objcells('vely'), objcells('velz')
    velx_obj = np.average(velx, weights=mass)
    vely_obj = np.average(vely, weights=mass)
    velz_obj = np.average(velz, weights=mass)
    vx, vy, vz = velx-velx_obj, vely-vely_obj, velz-velz_obj
    # Unpack coordinates of object cells
    relx, rely, relz = objxyz
    # Calculate mass weighted velocity dispersion (cartesian)
    cldispx = np.sqrt(np.average(vx**2, weights=mass))
    cldispy = np.sqrt(np.average(vy**2, weights=mass))
    cldispz = np.sqrt(np.average(vz**2, weights=mass))
    cldisp = np.sqrt(cldispx**2 +cldispy**2 +cldispz**2)
    # Calculate a safe normalized radial vector.
    r = np.maximum(1e-40,np.sqrt(relx**2+rely**2+relz**2))
    nx, ny, nz = relx/r, rely/r, relz/r
    # Calculate radial velocity component of each cell (outwards positive)
    vr = nx*vx + ny*vy + nz*vz # To square (from above) or not to square (projection) (CONFUSED!)
    vrx, vry, vrz = vr*nx, vr*ny, vr*nz
    # Calculate velocity dispersion (radial component)
    vrx_obj = np.average(vrx, weights=mass)
    vry_obj = np.average(vry, weights=mass)
    vrz_obj = np.average(vrz, weights=mass)
    cldispvrx = np.sqrt(np.average((vrx-vrx_obj)**2, weights=mass))
    cldispvry = np.sqrt(np.average((vry-vry_obj)**2, weights=mass))
    cldispvrz = np.sqrt(np.average((vrz-vrz_obj)**2, weights=mass))
    cldispvr = np.sqrt(cldispvrx**2 +cldispvry**2 +cldispvrz**2)
    # Calculate spherical prograde velocity vector (non-radial velocity component)
    vpx, vpy, vpz = vx-vrx, vy-vry, vz-vrz
    vp = np.sqrt(vpx**2+vpy**2+vpz**2)
    # Calculate velocity dispersion (prograde component)
    vpx_obj = np.average(vpx, weights=mass)
    vpy_obj = np.average(vpy, weights=mass)
    vpz_obj = np.average(vpz, weights=mass)
    cldispvpx = np.sqrt(np.average((vpx-vpx_obj)**2, weights=mass))
    cldispvpy = np.sqrt(np.average((vpy-vpy_obj)**2, weights=mass))
    cldispvpz = np.sqrt(np.average((vpz-vpz_obj)**2, weights=mass))
    cldispvp = np.sqrt(cldispvpx**2 +cldispvpy**2 +cldispvpz**2)
    # Calculate total momenta
    mom_radial = np.sum(vr*mass)
    mom_prograde = np.sum(vp*mass)
    # Pick up the results that we want to keep
    data = Pick(locals(), 'velx_obj', 'vely_obj', 'velz_obj',
                'vrx_obj', 'vry_obj', 'vrz_obj',
                'vpx_obj', 'vpy_obj', 'vpz_obj',
                'cldispx', 'cldispy', 'cldispz', 'cldisp',
                'cldispvrx', 'cldispvry', 'cldispvrz', 'cldispvr',
                'cldispvpx', 'cldispvpy', 'cldispvpz', 'cldispvp',
                'mom_radial', 'mom_prograde')
    if ddict is not None: ddict.update(data)
    return data

def VarInData(hdfdata, varname):
    try:
        ld = hdfdata[varname]
    except KeyError:
        return False
    else:
        return True

def NodeStatCooling(objinfo, ddict=None, debug=False):
    import cz_chemmap_eq as czchem
    leafdata, octree, objmask, objcenter, objxyz = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    # Check if important fields are in the plotfile.
    # If not, we might get some estimates, but they will not be as meaningful.
    irxe_present = VarInData(leafdata, 'irxe')
    cdh2_present = VarInData(leafdata, 'cdh2')
    cdco_present = VarInData(leafdata, 'cdco')
    cdto_present = VarInData(leafdata, 'cdto')
    chid_present = VarInData(leafdata, 'irxe')
    # Run chemistry for given object
    czchem.InitChem(leafdata)
    dt_sim = leafdata['real scalars/dt']
    dt_chem = dt_sim/2
    resfields = ('cool_rates', 'cool_rates_chem')
    chparam = dict(mask=objmask, cachemode=False, dictout=False, resfields=resfields)
    try:
        crp, crc = czchem.EvolveChem(leafdata, dt_chem, **chparam)
    except KeyError:
        dshape = objmask[objmask].shape
        if debug: tools.fprint('WARNING: CHEMISTRY DATA UNAVAILABLE')
        crp = np.full((30,)+dshape, np.nan)
        crc = np.full((8,)+dshape, np.nan)
    # Compute cooling rate (masked cells)
    dens = objcells('dens')
    vol = objcells('vol')
    n_hx = objcells('n_hx')
    hxcount = n_hx*vol
    mass = dens*vol
    coolrate_per_hx = np.sum(crp, axis=0) + np.sum(crc, axis=0) # Cooling per h-atom
    coolrate_per_vol = coolrate_per_hx * n_hx
    coolrate_per_mass = coolrate_per_vol / dens
    # Compute totals
    avgcool_per_hx = np.average(coolrate_per_hx, weights=hxcount)
    avgcool_per_vol = np.average(coolrate_per_vol, weights=vol)
    avgcool_per_mass = np.average(coolrate_per_mass, weights=mass)
    coolrate_obj = np.sum(coolrate_per_mass*mass)
    coolrates_crp_obj = np.sum(crp*hxcount, axis=1)
    coolrates_crc_obj = np.sum(crc*hxcount, axis=1)
    hxcount_total = np.sum(hxcount)
    # Pick up the results that we want to keep
    data = Pick(locals(), 'avgcool_per_hx', 'avgcool_per_vol', 'avgcool_per_mass',
                'coolrate_obj', 'coolrates_crp_obj', 'coolrates_crc_obj', 'hxcount_total')
    if ddict is not None: ddict.update(data)
    return data

def NodeStatMagAlign(objinfo, ddict=None):
    leafdata, octree, objmask, objcenter, objxyz = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    # Calculate direction of magnetic field (per cell)
    magx, magy, magz = objcells('magx'), objcells('magy'), objcells('magz')
    Bmag = np.maximum(1e-40,np.sqrt(magx**2+magy**2+magz**2))
    Bnx, Bny, Bnz = magx/Bmag, magy/Bmag, magz/Bmag
    # Calculate cos of angle between grad(dens) and mag (per cell)
    dgnx, dgny, dgnz = objcells('dgnx'), objcells('dgny'), objcells('dgnz')
    cos_phi = dgnx*Bnx + dgny*Bny + dgnz*Bnz
    acp = np.abs(cos_phi)
    # Calculate object acp averages.
    # One should probably use the vol-weighted average acp_avg_vol,
    # since Emag =~ sum(B*vol) is the conserved quantity.
    acp_avg_mass = np.average(acp, weights=objcells('mass'))
    acp_avg_vol = np.average(acp, weights=objcells('vol'))
    # Calculate weighted acp variance
    acp_var_mass = np.average((acp-acp_avg_mass)**2, weights=objcells('mass'))
    acp_var_vol = np.average((acp-acp_avg_vol)**2, weights=objcells('vol'))
    acp_std_mass, acp_std_vol = np.sqrt(acp_var_mass), np.sqrt(acp_var_vol)
    # Pick up the results that we want to keep
    data = Pick(locals(), 'acp_avg_mass', 'acp_avg_vol',
                'acp_var_mass', 'acp_var_vol', 'acp_std_mass', 'acp_std_vol')
    if ddict is not None: ddict.update(data)
    return data

def NodeStatVelAlign(objinfo, ddict=None):
    leafdata, octree, objmask, objcenter, objxyz = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    # Calculate direction of magnetic field (per cell)
    velx, vely, velz = objcells('velx'), objcells('vely'), objcells('velz')
    vel = np.maximum(1e-40,np.sqrt(velx**2+vely**2+velz**2))
    vnx, vny, vnz = velx/vel, vely/vel, velz/vel
    # Calculate cos of angle between grad(dens) and mag (per cell)
    dgnx, dgny, dgnz = objcells('dgnx'), objcells('dgny'), objcells('dgnz')
    cos_velphi = dgnx*vnx + dgny*vny + dgnz*vnz
    acvp = np.abs(cos_velphi)
    # Calculate object acvp averages.
    # One should probably use the mass-weighted average acvp_avg_mass.
    acvp_avg_mass = np.average(acvp, weights=objcells('mass'))
    acvp_avg_vol = np.average(acvp, weights=objcells('vol'))
    # Calculate weighted acvp variance
    acvp_var_mass = np.average((acvp-acvp_avg_mass)**2, weights=objcells('mass'))
    acvp_var_vol = np.average((acvp-acvp_avg_vol)**2, weights=objcells('vol'))
    acvp_std_mass, acvp_std_vol = np.sqrt(acvp_var_mass), np.sqrt(acvp_var_vol)
    # Pick up the results that we want to keep
    data = Pick(locals(), 'acvp_avg_mass', 'acvp_avg_vol',
                'acvp_var_mass', 'acvp_var_vol', 'acvp_std_mass', 'acvp_std_vol')
    if ddict is not None: ddict.update(data)
    return data

def NodeStatAverages(objinfo, ddict=None):
    leafdata, octree, objmask, objcenter, objxyz = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    ## Determine weighted average statistics for following vars:
    avgvars = ('dens', 'mag', 'temp', 'tdus', 'numdens', 'cdto',
                'velx', 'vely', 'velz', 'vel',
                'magx', 'magy', 'magz', 'mag',
                'pres', 'magp', 'velp', 'gravp', 'rlevel', 'jeans_res',
                'massfrac_co', 'massfrac_h2', 'beta', 't_ff', )
    # Determine mass and volume weighted average statistics
    data = dict()
    mass = objcells('mass')
    vol = objcells('vol')
    for var in avgvars:
        data['cl'+var] = np.average(objcells(var), weights=mass)
        data['clV'+var] = np.average(objcells(var), weights=vol)
    # Send results
    if ddict is not None: ddict.update(data)
    return data

def NodeStatExtrema(objinfo, ddict=None):
    leafdata, octree, objmask, objcenter, objxyz = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    ## Determine extrema for following vars:
    extvars = ('dens', 'numdens', 'temp', 'tdus', 'cdto', 'magp', 'velp',
                'pres', 'gravp', 'massfrac_co', 'massfrac_h2', 'gpot',
                'l_jeans', 'mu', 'beta', 'rlevel', 'jeans_res', 't_ff', )
    # Determine mass and volume weighted average statistics
    data = dict()
    for var in extvars:
        data['cl'+var+'_min'] = np.min(objcells(var))
        data['cl'+var+'_max'] = np.max(objcells(var))
    # Send results
    if ddict is not None: ddict.update(data)
    return data

def NodeStatBase(objinfo, ddict=None):
    leafdata, octree, objmask, objcenter, objxyz = objinfo
    # Define field reader that only returns the object cells
    objcells = lambda key: leafdata[key][objmask]
    #
    mass = objcells('mass')
    vol = objcells('vol')
    clmass = np.sum(mass)
    clvol = np.sum(vol)
    clvolrad = np.cbrt(3./(4.*np.pi)*clvol)
    # Compute object size from mass-weighted r^2
    relx, rely, relz = objxyz
    rsq = relx**2 + rely**2 + relz**2
    clIrad = np.sqrt(np.average(rsq, weights=mass))
    # Decode center of mass
    center_x, center_y, center_z = objcenter
    # --- Put together some diagnostic telemetry
    diag_ncells = np.sum(objmask)
    # Send results
    data = Pick(locals(), 'clmass', 'clvol', 'clvolrad', 'clIrad',
                'center_x', 'center_y', 'center_z',
                'diag_ncells',)
    if ddict is not None: ddict.update(data)
    return data

#===============================================================================

def NodeStatCollection(leafdata, octree, objmask, iobj, nobj, gcdata=dict(), ddict=None, sfb=True):
    ''' 
    Collect all statistics offered by the NODE STATISTICS section in cz_trfragstat.
    '''
    objdata = dict()
    # Prepare input data
    objcenter, objxyz = czfifra.ObjCoords(leafdata, octree, objmask)
    objinfo = leafdata, octree, objmask, objcenter, objxyz
    objinfo_ext = leafdata, octree, objmask, objcenter, objxyz, iobj, nobj 
    #
    NodeStatBase(objinfo, ddict=objdata)
    if sfb: fprint('b', end='')
    NodeStatAngmom(objinfo, ddict=objdata)
    if sfb: fprint('A', end='')
    NodeStatVelDispersion(objinfo, ddict=objdata)
    if sfb: fprint('D', end='')
    NodeStatCooling(objinfo, ddict=objdata)
    if sfb: fprint('C', end='')
    NodeStatMagAlign(objinfo, ddict=objdata)
    if sfb: fprint('M', end='')
    NodeStatVelAlign(objinfo, ddict=objdata)
    if sfb: fprint('V', end='')
    NodeStatGravBound(objinfo_ext, ddict=objdata)
    if sfb: fprint('G1', end='')
    NodeStatGravTide(objinfo_ext, ddict=objdata)
    if sfb: fprint('2', end='')
    NodeStatGravAccel(objinfo_ext, ddict=objdata)
    if sfb: fprint('3', end='')
    NodeStatSurface(objinfo, ddict=objdata, gcdata=gcdata)
    if sfb: fprint('S', end='')
    NodeStatAverages(objinfo, ddict=objdata)
    if sfb: fprint('a', end='')
    NodeStatExtrema(objinfo, ddict=objdata)
    if sfb: fprint('e', end='')
    NodeStatSinks(objinfo, ddict=objdata)
    if sfb: fprint('s', end='')
    #
    if ddict is not None: ddict.update(objdata)
    return objdata


#===============================================================================
# ==== NODE STATISTICS COLLECTOR ===============================================
#===============================================================================

# --- Will want the following points:
# Note: Some of them might not make sense for the exclusive part
# * (X) Simple averages
# * (X) Simple extrema # XXX percentiles (10/50/90) might be more informative!
# * (X) Angular momentum
# * (1/2) Velocity dispersion (w/wo. radial component)
# * (X) Mass, Volume, Radius
# * (X) Aspect ratio
# * (X) Cooling
# * (X) Magnetic Alignment
# * (X) Grav. boundedness
# * (X) Virial volume terms
# * (-) Virial surface terms
# * (X) Virial grav terms
# XXX What about contained sink particles?

def AddKeyPrefix(d, prefix):
    '''
    Add prefix to all keys in a dictionary to avoid key confusion
    when combining dict outputs of multiple functions
    '''
    prefix_d = {f'{prefix}_{k}':v for k,v in d.items()}
    return prefix_d

def ObjDataPhasesize(leafdata, octree, objmask, ddict=None):
    import cl_phasesize_ar as clps
    import pl_dendroplot as dpl
    objvector = dpl.FakeVector(leafdata)
    objdata = clps.ObjData(leafdata, octree, objvector, objmask)
    if ddict is not None:
        clps_objdata = AddKeyPrefix(objdata, 'clps')
        ddict.update(clps_objdata)
    return objdata

def ObjDataPhaseshape(clps_objdata, ddict=None): # TODO
    import pl_dendrostat as dds
    r_sphere, r_sheet, r_filament = dds.ShapeRatios(clps_objdata, 'stdr')
    r_shapedim = dds.ShapeVal(clps_objdata, 'stdr') # int, 0=Mixed
    v_sphere, v_sheet, v_filament = dds.ShapeRatios(clps_objdata, 'stdv')
    v_shapedim = dds.ShapeVal(clps_objdata, 'stdv') # int, 0=Mixed
    objdata = Pick(locals(),
                   'r_sphere', 'r_sheet', 'r_filament', 'r_shapedim',
                   'v_sphere', 'v_sheet', 'v_filament', 'v_shapedim')
    if ddict is not None:
        dds_objdata = AddKeyPrefix(objdata, 'dds')
        ddict.update(dds_objdata)
    return objdata

def Nodestat(leafdata, octree, objmask, iobj, nobj, gcdata=dict(), sfb=True):
    assert np.any(objmask) # If an empty object got here then something is fubar.
    objdata_collected = dict()
    data_clps = ObjDataPhasesize(leafdata, octree, objmask, ddict=objdata_collected)
    if sfb: fprint('r', end='')
    ObjDataPhaseshape(data_clps, ddict=objdata_collected)
    if sfb: fprint('s', end='')
    NodeStatCollection(leafdata, octree, objmask, iobj, nobj, gcdata=gcdata, ddict=objdata_collected, sfb=sfb)
    if sfb: fprint(' ', end='')
    return objdata_collected


#===============================================================================
# ==== RESULT PROCESSING =======================================================
#===============================================================================

shapewarn_ignore = ()

def RepackageResult(list_of_dicts, debug=True):
    '''
    Repackage a uniform list of dictionaries into a dictionary of arrays
    '''
    if len(list_of_dicts) == 0:
        raise RuntimeError('RepackageResult: Empty object list')
    # Note: A uniform list of dictionaries is expected to contain
    # exactly the same set of dictionary keys for each list entry.
    keys0 = set(list_of_dicts[0].keys())
    assert np.all([set(d.keys())==keys0 for d in list_of_dicts])
    # Repackage along the keys listed in keys0:
    result = dict()
    for key in keys0:
        if debug:
            fprint(f'Repackaging {key}')
        result[key] = np.array([d[key] for d in list_of_dicts])
    # Check the arrays. Warn if dimension>1 (perhaps forgotten sum/average)
    for key in keys0:
        arr = result[key]
        if (not arr.ndim == 1) and (not key in shapewarn_ignore):
            print(f'WARNING: {key} array shape: {arr.shape}')
    #
    return result

def PrepareGradDens(leafdata, octree):
    fprint('Calculating grad(rho)...', end='')
    dens_grid = mflash.datagrid(leafdata['dens'], octree, ng=1)
    dgx, dgy, dgz = dens_grid.gradient()
    dgabs = np.maximum(1e-200,np.sqrt(dgx**2+dgy**2+dgz**2))
    dgnx, dgny, dgnz = dgx/dgabs, dgy/dgabs, dgz/dgabs # Gradient direction
    # Store calculated fields in leafdata cache
    fields = Pick(locals(), 'dgx', 'dgy', 'dgz', 'dgabs', 'dgnx', 'dgny', 'dgnz')
    leafdata.cache_fields(fields)
    fprint('OK')
    return

def ExclusiveNodeMasks(nodemaks):
    fprint('Calculating exclusive object masks... ', end='')
    Moo_contain = cztrown.ObjObjContainmentMatrix(nodemaks)
    cell_ownerobj = cztrown.CellOwnershipObjLUT(nodemaks, Moo_contain)
    objmasks_exclusive = cztrown.CellOwnershipMasks(cell_ownerobj, objmasks=nodemaks)
    fprint('OK')
    return objmasks_exclusive

def fragdata_main(leafdata, octree, objmasks, debug=True):
    #
    PrepareGradDens(leafdata, octree)
    PrepareHess(leafdata, octree, 'gpot')
    gcdata = PrepareGuardVars(leafdata, octree)
    #
    fprint(f'Preparing inclusive node statistics..')
    nobj = len(objmasks)
    PrepareObjsGpot(leafdata, octree, objmasks, verbose=True)
    PrepareObjsTidal(leafdata, octree, objmasks, verbose=True)
    fprint(f'...Computing inclusive node statistics...', end='')
    nodestat_inclusive = [Nodestat(leafdata, octree, objm, iobj, nobj, gcdata=gcdata) for iobj,objm in enumerate(objmasks)]
    arrdict_inclusive = RepackageResult(nodestat_inclusive)
    fprint('OK')
    #
    fprint(f'Preparing exclusive node statistics..', end='')
    objmasks_exclusive = ExclusiveNodeMasks(objmasks)
    nobj_exclusive = len(objmasks_exclusive)
    assert nobj_exclusive == nobj
    PrepareObjsGpot(leafdata, octree, objmasks_exclusive, verbose=True)
    PrepareObjsTidal(leafdata, octree, objmasks_exclusive, verbose=True)
    fprint(f'...Computing exclusive node statistics...', end='')
    nodestat_exclusive = [Nodestat(leafdata, octree, objm, iobj, nobj_exclusive, gcdata=gcdata) for iobj,objm in enumerate(objmasks_exclusive)]
    arrdict_exclusive = RepackageResult(nodestat_exclusive)
    fprint('OK')
    #
    assert len(nodestat_inclusive) == len(nodestat_exclusive)
    # Package the data from list(dict) to dict(array).
    # While at it, pack inclusive and exclusive data in the same dict,
    # prefixing the exclusive data keys with excl_.
    data_out = arrdict_inclusive | AddKeyPrefix(arrdict_exclusive, 'excl')
    return data_out

def determine_fragdata(leafdata, octree, objmasks, **kwargs):
    sinkdict = cz.SinkData(leafdata)
    # --- Step 1: Compute node statistics, ignoring sinks
    fprint(f'Computing node statistics ({len(objmasks)} nodes wo. sinkdata):')
    if len(objmasks)==0: return dict() # No objects? We are done!
    cz.BG_SinkMass(leafdata, octree, sinkdict)
    fragdata_nosink = fragdata_main(leafdata, octree, objmasks, **kwargs)
    # --- Step 2: Compute node statistics while mapping sink data
    if sinkdict['nsink'] == 0:
        fprint('WARNING: No sink particles found, skipping sinkmap stats.')
        return fragdata_nosink
    else:
        fprint(f'Computing node statistics ({len(objmasks)} sink-mapped nodes):')
    cz.BG_MapSinksToGrid(leafdata, octree, sinkdict)
    fragdata_sinkmap = fragdata_main(leafdata, octree, objmasks, **kwargs)
    # --- Step 3: Combine data
    fragdata = fragdata_nosink | AddKeyPrefix(fragdata_sinkmap, 'sinkmap')
    return fragdata

def determine_basedata(leafdata, octree, criterion):
    filename = leafdata.filename()
    filebase = os.path.basename(filename)
    simdir = os.path.dirname(filename)
    basedir = os.path.basename(simdir)
    simtime = leafdata['real scalars/time']
    simstep = leafdata['integer scalars/nstep']
    fields = Pick(locals(), 'filename', 'filebase', 'simdir', 'basedir', 'criterion', 'simtime', 'simstep')
    #parfile = os.path.join(simdir, 'flash.par')
    #pardict = tools.readpar(parfile)
    #pardict.update(fields)
    return fields


#===============================================================================
# ==== MISCDATA I/O ============================================================    
#===============================================================================

def get_miscdata_path(hdfdata1, criterion, debug=True):
    hdf1_fp = tools.get_plotfile_fingerprint(hdfdata1)
    crit_fp = clfind.get_criterion_fingerprint(criterion)
    prefix = miscdata_basename +crit_fp +'_' +hdf1_fp +'_'
    filepath = tools.get_datafile_path(hdfdata1, prefix, outdir=datadir)
    if debug>0:
        fprint(f'MISCDATA PATH: {filepath}')
    return filepath

def prepare_miscdata(hdfdata1, octree1, criterion, objmasks=None):
    ''' Compile and save a cluster misc. data to criterion if not existing yet '''
    filepath = get_miscdata_path(hdfdata1, criterion)
    # Check if the list is already in place
    if os.path.exists(filepath) and not 'test' in criterion:
        fprint(f'-> Cluster misc. data already exists: {filepath}')
    else:
        if objmasks is None:
            objmasks = clfind.get_cmasks(hdfdata1, octree1, criterion)
        miscdata = determine_miscdata(hdfdata1, octree1, objmasks)
        tools.save_data(miscdata, filepath, force=False)
    return filepath, objmasks

def get_miscdata(hdfdata1, octree1, criterion, save=True):
    ''' Obtain a cluster misc. data according to criterion; compile if necessary '''
    filepath = get_miscdata_path(hdfdata1, criterion)
    #
    force = False    
    if os.path.exists(filepath):
        fprint(f'-> Loading cluster misc. data ({criterion})')
        try:
            miscdata = tools.load_data(filepath)
        except:
            print('!! Warning: Could not load cluster misc. data !!')
            force = True
        else:
            return miscdata
    objmasks = clfind.get_cmasks(hdfdata1, octree1, criterion)
    miscdata = determine_miscdata(hdfdata1, octree1, objmasks)
    if save:
        fprint(f'-> Saving misc. data ({criterion})')
        try:
            tools.save_data(miscdata, filepath, force=force)
        except:
            print('!! Warning: Could not save misc. fragment data !!')
    return miscdata


#===============================================================================
# ==== FRAGDATA I/O ============================================================    
#===============================================================================

def get_fragdata_path(hdfdata1, criterion, debug=True):
    hdf1_fp = tools.get_plotfile_fingerprint(hdfdata1)
    crit_fp = clfind.get_criterion_fingerprint(criterion)
    prefix = fragdata_basename +crit_fp +'_' +hdf1_fp +'_'
    filepath = tools.get_datafile_path(hdfdata1, prefix, outdir=datadir)
    if debug>0:
        fprint(f'FRAGDATA PATH: {filepath}')
    return filepath

def prepare_fragdata(hdfdata1, octree1, criterion, objmasks=None):
    ''' Compile and save a cluster fragment data to criterion if not existing yet '''
    filepath = get_fragdata_path(hdfdata1, criterion)
    # Check if the list is already in place
    if os.path.exists(filepath) and not 'test' in criterion:
        fprint(f'-> Cluster fragment data already exists: {filepath}')
    else:
        if objmasks is None:
            objmasks = clfind.get_cmasks(hdfdata1, octree1, criterion)
        fragdata = determine_fragdata(hdfdata1, octree1, objmasks)
        tools.save_data(fragdata, filepath, force=False)
    return filepath, objmasks

def get_fragdata(hdfdata1, octree1, criterion, save=True):
    ''' Obtain a cluster fragment data according to criterion; compile if necessary '''
    filepath = get_fragdata_path(hdfdata1, criterion)
    #
    force = False    
    if os.path.exists(filepath):
        fprint(f'-> Loading cluster fragment data ({criterion})')
        try:
            fragdata = tools.load_data(filepath)
        except:
            print('!! Warning: Could not load cluster fragment data !!')
            force = True
        else:
            return fragdata
    objmasks = clfind.get_cmasks(hdfdata1, octree1, criterion)
    fragdata = determine_fragdata(hdfdata1, octree1, objmasks)
    if save:
        fprint(f'-> Saving cluster fragment data ({criterion})')
        try:
            tools.save_data(fragdata, filepath, force=force)
        except:
            print('!! Warning: Could not save cluster fragment data !!')
    return fragdata


#===============================================================================
# ==== RESULTLIST I/O ==========================================================    
#===============================================================================

def compile_fragdata_collection(filelist_sorted, criterion):
    fragdata_t = list()
    basedata_t = list()
    for fn in filelist_sorted:
        leafdata, octree = cz.LoadPlotfile(fn)
        fragdata = get_fragdata(leafdata, octree, criterion)
        basedata = determine_basedata(leafdata, octree, criterion)
        miscdata = get_miscdata(leafdata, octree, criterion)
        fragdata_t.append(fragdata|miscdata)
        basedata_t.append(basedata)
        del(octree)
        leafdata.close()
    return fragdata_t, basedata_t

def get_fdc_path(filelist_sorted, criterion, debug=True):
    pf0 = filelist_sorted[0]
    simdir = os.path.dirname(pf0)
    filelist_fp = tools.get_strlist_fingerprint(filelist_sorted)
    crit_fp = clfind.get_criterion_fingerprint(criterion)
    prefix = fdc_basename +'collection_' +crit_fp +'_' +filelist_fp
    filepath = tools.get_collection_path(simdir, prefix, outdir=datadir)
    if debug>0:
        print(f'FRAGDATA COLLECTION PATH: {filepath}')
        sys.stdout.flush()
    return filepath

def prepare_fdc(filelist_sorted, criterion):
    ''' Compile and save a fragment data collection if not existing yet '''
    filepath = get_fdc_path(filelist_sorted, criterion)
    # Check if the list is already in place
    if os.path.exists(filepath) and not 'test' in criterion:
        fprint(f'-> Fragment data collection already exists: {filepath}')
    else:
        fdc = compile_fragdata_collection(filelist_sorted, criterion)
        tools.save_data(fdc, filepath, force=False)
    return filepath

def get_fdc(filelist_sorted, criterion, save=True):
    ''' Obtain a fragment data collection; compile if necessary '''
    filepath = get_fdc_path(filelist_sorted, criterion)
    #
    force = False    
    if os.path.exists(filepath):
        fprint(f'-> Loading fragment data collection ({criterion})')
        try:
            fdc = tools.load_data(filepath)
        except:
            print('!! Warning: Could not load fragment data collection !!')
            force = True
        else:
            return fdc
    fragdata = compile_fragdata_collection(filelist_sorted, criterion)
    if save:
        fprint(f'-> Saving fragment data collection ({criterion})')
        try:
            tools.save_data(fragdata, filepath, force=force)
        except:
            print('!! Warning: Could not save fragment data collection !!')
    return fragdata


#===============================================================================
# ==== TEST ====================================================================
#===============================================================================

test_ppath = '/home/mweis/scratch/thesisdata/corezoom-work-wbox/b.Aa1-Aa4-2zone/out'
test_pf = os.path.join(test_ppath, 'CZ-b_hdf5_plt_cnt_0400')
test_criterion = 'COCxp30:adrnode'
#test_ppath = '/home/mweis/scratch/thesisdata/birka_judata/update/rfl9_update'
#test_pf = os.path.join(test_ppath, 'SpitzerTest_hdf5_plt_cnt_0271')
#test_criterion = 'COCxp30:adrnode'

def Test():
    leafdata, octree = cz.LoadPlotfile(test_pf)
    mass0 = np.sum(leafdata['mass'])
    sinkdict = cz.SinkData(leafdata)
    cz.BG_MapSinksToGrid(leafdata, octree, sinkdict)
    print('--------------------------------')
    mass_ld = np.sum(leafdata['mass'])
    print('#D1', mass_ld)
    print('   ', np.sum(leafdata['msink']), np.sum(leafdata['nsink']))
    print('--------------------------------')
    mp = leafdata.flush('mass')
    print('#D2', np.sum(mp))
    print('--------------------------------')
    mass_ld = np.sum(leafdata['mass'])
    print('#D3', mass_ld)
    print('--------------------------------')
    densvol_ld = np.sum(leafdata['dens']*leafdata['vol'])
    print('#D4', densvol_ld)
    print('--------------------------------')
    #objmasks = clfind.get_cmasks(leafdata, octree, test_criterion)
    #ld = determine_fragdata(leafdata, octree, objmasks)
    #objmass0 = ld['clmass']
    #objmass1 = ld['sinkmap_clmass']
    #objmass2 = ld['excl_clmass']
    #objmass3 = ld['sinkmap_excl_clmass']
    #print(np.sum(objmass0))
    #print(np.sum(objmass1))
    #print(np.sum(objmass2))
    #print(np.sum(objmass3))


#===============================================================================
# ==== MAIN ====================================================================
#===============================================================================

def Main(filelist_sorted, criterion):
    for ifn, fn in enumerate(filelist_sorted):
        leafdata, octree = cz.LoadPlotfile(fn)
        objmasks = None # Load later only if required
        
        # Skip execution if fragment data already exists
        fdp = get_fragdata_path(leafdata, criterion)
        if os.path.exists(fdp):
            fprint(f'-> Fragment statistics already prepped: {fdp}')
        else:
            fprint(f'-> Running fragment statistics prep: {fdp}')
            fp_frag, objmasks = prepare_fragdata(leafdata, octree, criterion)

        leafdata.flush()
        cz.LoadCellPos(leafdata, octree)

        # Skip execution if misc. data already exists
        mdp = get_miscdata_path(leafdata, criterion)
        if os.path.exists(mdp):
            fprint(f'-> Misc. statistics already prepped: {mdp}')
        else:
            fprint(f'-> Running misc. statistics prep: {mdp}')
            fp_misc, objmasks = prepare_miscdata(leafdata, octree, criterion, objmasks=objmasks)

        # Cleanup
        del(octree)
        leafdata.close()


#===============================================================================
# ==== LOBBY ===================================================================
#===============================================================================

if __name__ == '__main__':
    if len(sys.argv) < 2:
        Test()
    else:
        criterion = sys.argv[1]
        try:
            stride = int(sys.argv[2])
            offset = int(sys.argv[3])
        except ValueError:
            dirs = sys.argv[2:]
            fprint('Collecting plotfiles...')
            plotfiles = cztrmatch.CollectPlotfiles(dirs, include=plotfile_include_patterns)
            fprint('Sorting plotfiles...')
            plotfiles_sorted = cztrmatch.OrderPlotfiles(plotfiles)
            fprint(plotfiles_sorted)
            get_fdc(plotfiles_sorted, criterion)
        else:
            dirs = sys.argv[4:]
            fprint('Collecting plotfiles...')
            plotfiles = cztrmatch.CollectPlotfiles(dirs, include=plotfile_include_patterns)
            fprint('Sorting plotfiles...')
            plotfiles_sorted = cztrmatch.OrderPlotfiles(plotfiles)
            fprint(plotfiles_sorted[offset::stride])
            Main(plotfiles_sorted[offset::stride], criterion)
    print('FINISHED!')
