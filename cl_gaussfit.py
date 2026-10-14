import sys
import numpy as np
import libmf.micflash as mflash
import cz_tools as cz
import constants as cu

_fw_chunksize = 7e+7

# ==============================================================================
# ==== HELPER ==================================================================
# ==============================================================================

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
    
def GetMask(baseshape, clist):
    cl_where = select(clist)
    mask = np.zeros(baseshape, dtype=bool)
    mask[cl_where] = True
    return mask

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


# ==============================================================================
# ==== ITERATIVE MAXIMUM FINDER ================================================
# ==============================================================================

def _blocksearch_piecewise(func):
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


def _stack_neighborhood_indices(blshape):
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

@_blocksearch_piecewise
def _blocksearch_maxneigh(height_g, cellid_g):
    ''' SEARCH BLOCK-WISE FOR MAXIMA IN 1-NEIGHBORHOOD OF CELLS '''
    # Check input
    assert height_g.shape == cellid_g.shape
    # Append extra dimension stacking the celldata panned through the neighborhood
    K, J, I = _stack_neighborhood_indices(height_g.shape)
    height_gn = height_g[:,K,J,I]
    cellid_gn = cellid_g[:,K,J,I]
    # Search for o-argument of height maxima in 1-neighborhood
    o = np.argmax(height_gn, axis=-1)
    # Get height and cellid from found maximum by inserting found o-argument
    bl, k, j, i = np.meshgrid(*lmap(range, height_gn.shape[:-1]), indexing='ij')
    heightmax_height_gn = height_gn[bl,k,j,i,o]
    heightmax_cellid_gn = cellid_gn[bl,k,j,i,o]
    return heightmax_height_gn, heightmax_cellid_gn

def GetMaxima(height, octree, nneigh=4):
    ''' Get local maxima inside an exlusion radius of nneigh cells. '''
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
    gbld_cellid = np.zeros(gbld_shape, dtype=np.int64)
    gbld_cellid[select_data]  = +1 +np.arange(np.sum(select_data))
    gbld_cellid[select_guard] = -1 -np.arange(np.sum(select_guard))
    cellID = gbld_cellid[...,ng:-ng,ng:-ng,ng:-ng]
    # Use the ng-radius jump finder algorithm from the fellwalk implementation
    jumpID = np.copy(cellID)
    heightmax_gni, linkedid_gni = height_gbld, gbld_cellid
    ngnj = ng
    for n_jumps in range(1, ng+1):
        print(f'Searching ascension jumps in {n_jumps} cell radius...')
        sys.stdout.flush()
        # Search height maxima in 2-Neighborhood of cells
        # by recursively searching 1-Neighborhood of 1-Neighborhood results
        heightmax_gnj, linkedid_gnj = _blocksearch_maxneigh(heightmax_gni, linkedid_gni)
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
    return maxmask


# ==============================================================================
# ==== FIT GAUSSIAN (GENERAL, ONE AT A TIME) ===================================
# ==============================================================================

def F_Gaussian(A,B):
    def func(relpos):
        # Unpack radial coordinates
        # Here we assume that relpos has the format (3,...)
        rx, ry, rz = [relpos[i] for i in range(3)]
        # Calculate exponential argument r.T*B*r (B symmetric)
        xi = B[0,0]*rx**2 + B[1,1]*ry**2 + B[2,2]*rz**2
        xi += 2.*B[0,1]*rx*ry + 2.*B[0,2]*rx*rz + 2.*B[1,2]*ry*rz
        # Calculate exponential function
        values = A*np.exp(-.5*xi)
        return values
    return func

def F_Background(b0, gx, gy, gz):
    b0, gx, gy, gz
    def func(relpos):
        # Unpack radial coordinates
        # Here we assume that relpos has the format (3,...)
        rx, ry, rz = [relpos[i] for i in range(3)]
        values = b0 + gx*rx + gy*ry + gz*rz
        return values
    return func

def F_GaussianWithBackground(B,coeffs):
    Ac, b0, gx, gy, gz = coeffs
    Gaussian = F_Gaussian(Ac,B)
    Background = F_Background(b0, gx, gy, gz)
    def func(relpos):
        values = Gaussian(relpos) + Background(relpos)
        return values
    return func

def GaussianCovarianceMatrix(B, debug=False):
    # Invert to get covariance matrix
    Sigma = np.linalg.inv(B)
    # Enforce symmetry numerically
    Sigma = .5 * (Sigma + Sigma.T)
    #
    if debug>0:
        print('#Sigma:')
        print(Sigma)
    #
    return Sigma

def FitAnisotropicGaussian(values, relpos, debug=True):
    """
    Fit A * exp(-0.5 r^T B r) with known center mu=0.
    Returns A, B.
    """
    # Make sanity checks on the input
    assert values.shape == relpos.shape[1:]
    assert relpos.shape[0] == 3
    if debug>0:
        print(f'# fit values: {values.shape}')
    # Unpack radial coordinates
    # Here we assume that relpos has the format (3,...), and will ravel
    rx, ry, rz = [relpos[i].ravel() for i in range(3)]
    # Build design matrix for log(values)
    ones = np.ones_like(rx)
    M_design = np.column_stack([ones, -.5*rx**2, -.5*ry**2, -.5*rz**2, -rx*ry, -rx*rz, -ry*rz])
    # Linear list squares fit in log space
    logval = np.log(values).ravel()
    p, *_ = np.linalg.lstsq(M_design, logval, rcond=None)
    # Unpack result
    A = np.exp(p[0])
    B = np.array([
        [p[1], p[4], p[5]],
        [p[4], p[2], p[6]],
        [p[5], p[6], p[3]] ])
    if debug>0:
        print('#B:')
        print(B)
    #
    return A, B

def FitAmplitudeAndBackground(values, relpos, B):
    """
    Given fixed center mu=0 and precision matrix B,
    Fit corrected amplitude A and linear background b+(g,r):
      f = Ac * exp(-0.5 r^T B r) + (b0 + gx * rx + gy * ry + gz * rz)
    via linear least squares.
    Returns Ac, b0, gx, gy, gz.
    """
    # Make sanity checks on the input
    assert values.shape == relpos.shape[1:]
    assert relpos.shape[0] == 3
    # Unpack radial coordinates
    # Here we assume that relpos has the format (3,...), and will ravel
    rx, ry, rz = [relpos[i].ravel() for i in range(3)]
    # Calculate exponential argument r.T*B*r (B symmetric)
    xi = B[0,0]*rx**2 + B[1,1]*ry**2 + B[2,2]*rz**2
    xi += 2.*B[0,1]*rx*ry + 2.*B[0,2]*rx*rz + 2.*B[1,2]*ry*rz
    # Calculate exponential factor
    phi = np.exp(-.5*xi)
    # Design matrix: [phi, 1, rx, ry, rz]
    ones = np.ones_like(phi)
    M_design = np.column_stack([phi,ones,rx,ry,rz])
    # Linear list squares fit in original (not log) space
    coeffs, *_ = np.linalg.lstsq(M_design, values, rcond=None)
    Ac, b0, gx, gy, gz = coeffs
    return Ac, b0, gx, gy, gz

def FitFWHM(B):
    # Compute eigenvectors and eigenvalues of the covariance matrix
    Sigma = GaussianCovarianceMatrix(B)
    eigvals, eigvecs = np.linalg.eigh(Sigma)
    # Calculate the FWHMs from the eigen-covariances
    stdev = np.sqrt(np.fabs(eigvals))
    fwhm = 2.0 * np.sqrt(2.0 * np.log(2.0)) * stdev
    return fwhm, eigvecs

def PrintFitInformation(center, maxval, fwhm):
    center = np.asarray(center)
    # Print values
    print(f'Center position (pc): {center/cu.pc}')
    print(f'Center value (cgs): {maxval}')
    print(f'FWHM mainax sizes (pc): {fwhm/cu.pc}')
    sys.stdout.flush()
    #
    return None


# ==============================================================================
# ==== FIT GAUSSIAN (V2, NON-LINEAR) ===========================================
# ==============================================================================

from scipy.optimize import least_squares

# ---------- Utilities ----------

def pack_cholesky_params(L):
    """Pack lower-triangular L (3x3) into 6 params: (a,b,c,l21,l31,l32)
       with L = [[e^a, 0,   0],
                 [l21, e^b, 0],
                 [l31, l32, e^c]]
    """
    a = np.log(L[0,0]); b = np.log(L[1,1]); c = np.log(L[2,2])
    return np.array([a, b, c, L[1,0], L[2,0], L[2,1]])

def unpack_cholesky_params(theta):
    """Turn 6 params into SPD precision matrix B = L L^T."""
    a, b, c, l21, l31, l32 = theta
    L = np.array([[np.exp(a), 0.0,      0.0     ],
                  [l21,        np.exp(b), 0.0     ],
                  [l31,        l32,        np.exp(c)]])
    B = L @ L.T
    return B

def gaussian_phi(B, r):
    """phi = exp(-0.5 * r^T B r) for all points (N,3)."""
    # q = sum_i,j B_ij r_i r_j; do it efficiently
    rx, ry, rz = r[:,0], r[:,1], r[:,2]
    q = (B[0,0]*rx*rx + B[1,1]*ry*ry + B[2,2]*rz*rz
         + 2*B[0,1]*rx*ry + 2*B[0,2]*rx*rz + 2*B[1,2]*ry*rz)
    return np.exp(-0.5 * q)

def solve_linear_coeffs(phi, r, vals):
    """Given phi (N,), r (N,3), vals (N,), solve for [A, b0, gx, gy, gz] by LS."""
    rx, ry, rz = r[:,0], r[:,1], r[:,2]
    M = np.column_stack([phi, np.ones_like(phi), rx, ry, rz])   # (N,5)
    coeffs, *_ = np.linalg.lstsq(M, vals, rcond=None)
    return coeffs  # A, b0, gx, gy, gz

# ---------- Main fitter ----------

def FitGaussianAndBG(values, relpos, robust_loss="soft_l1", nstep_max=1000,
                     f_scale=1.0, posunit=1., debug=False):
    """
    Fit in one go:
      f(x) = A * exp(-0.5 r^T B r) + (b0 + g·r)
    Unknowns (nonlinear): B (via Cholesky 6 params)
    Solved exactly each iteration (linear): A, b0, gx, gy, gz

    Returns dict with A, b0, g(3,), B
    """

    if debug>0:
        print(f'# fit values: {values.shape}')

    val = np.asarray(values).ravel()
    rx,ry,rz = np.asarray(relpos) / posunit
    rpos = np.array((rx.ravel(),ry.ravel(),rz.ravel())).T

    # Initial guess for B (choose unit accordingly!)
    B0 = np.eye(3)
    # Initial guess for center offset (choose unit accordingly!)
    off0 = np.zeros(3)

    # Cholesky parameterization
    # Start from chol(B0 + eps I)
    w0, V0 = np.linalg.eigh(0.5*(B0+B0.T))
    w0 = np.clip(w0, 1e-8, None)
    B0 = V0 @ np.diag(w0) @ V0.T
    L0 = np.linalg.cholesky(B0)

    # Parameter vector
    theta0 = np.concatenate([pack_cholesky_params(L0), off0])

    # ---- Residual function with variable projection ----
    def residuals(theta):
        B = unpack_cholesky_params(theta)
        phi = gaussian_phi(B, rpos)
        A, b0, gx, gy, gz = solve_linear_coeffs(phi, rpos, val)
        # Compute residuals in original space
        model = A*phi + (b0 + gx*rx + gy*ry + gz*rz)
        return model - val

    def residuals_off(theta):
        theta_B = theta[:6]
        offset = theta[6:]
        r = rpos - offset  # center is unknown -> recompute each iteration
        B = unpack_cholesky_params(theta_B)
        phi = gaussian_phi(B, r)
        A, b0, gx, gy, gz = solve_linear_coeffs(phi, r, val)
        rx, ry, rz = r[:,0], r[:,1], r[:,2]
        model = A*phi + (b0 + gx*rx + gy*ry + gz*rz)
        return model - val

    res = least_squares(residuals_off, theta0,
        loss=robust_loss, f_scale=f_scale, method="trf", max_nfev=nstep_max)

    # Final parameters
    theta_B = res.x[:6]
    B = unpack_cholesky_params(theta_B)  # Solve linear part one last time with final B
    offset = res.x[6:]
    r = rpos - offset
    phi = gaussian_phi(B, rpos)
    A, b0, gx, gy, gz = solve_linear_coeffs(phi, r, val)

    if debug>0:
        print('#B:')
        print(B)

    B_cgs = B*(1./posunit)**2
    gx_cgs = gx*(1./posunit)
    gy_cgs = gy*(1./posunit)
    gz_cgs = gz*(1./posunit)
    coeffs_cgs = A, b0, gx_cgs, gy_cgs, gz_cgs
    offset_cgs = offset * posunit
    status = {
        "success": res.success,
        "message": res.message,
    }
    
    return A, B_cgs, coeffs_cgs, offset_cgs, status



# ==============================================================================
# ==== FIT GAUSSIANS (AMR HELPER) ==============================================
# ==============================================================================

def DownselectForGaussian(values, relpos, maxval=None, minfrac=.01, radius=.05*cu.pc, debug=True):
    relpos = np.asarray(relpos)
    dshape = values.shape
    dsmask = np.ones(dshape, dtype=bool)
    assert dshape == relpos.shape[1:]
    # Downselect by radius
    if radius is not None:
        relradius_sq = np.sum(relpos**2, axis=0)
        radselect = relradius_sq < radius**2
        if debug>0:
            print(f'Downselect (radius): {np.sum(radselect*dsmask)}/{np.sum(dsmask)} cells')
        dsmask *= radselect
    # Calculate max value if not given
    if maxval is None:
        i = np.argmin(relradius_sq)
        maxval = values.ravel()[i]
    # Downselect by value
    if minfrac is not None:
        valselect = values >= maxval*minfrac
        if debug>0:
            print(f'Downselect (values): {np.sum(valselect*dsmask)}/{np.sum(dsmask)} cells')
        dsmask *= valselect
    # Do sanity checks
    assert np.any(dsmask)
    assert np.max(values[dsmask]) > .99*maxval
    if debug>0:
        print(f'Downselect (total): {np.sum(dsmask)}/{np.prod(dshape)} cells')
    return dsmask

def MaskGaussianFWHM(leafdata, octree, B, center, threshold=.5, debug=True):
    # TODO: This should really use subsampling
    relpos = cz.BG_RelCoords(octree, center)
    phi = F_Gaussian(1.,B)(relpos)
    mask = phi>=threshold
    if debug>0:
        fwhm_vol = np.sum(leafdata['vol'][mask]) / (cu.pc)**3
        print(f'FWHM Volume (pc^3): {fwhm_vol} ({np.sum(mask)} cells)')
        sys.stdout.flush()
    return mask

def closest(a, val=0.):
    """ Find the value in a that is closest to val, but not equal """
    absdiff = np.abs(a-val)
    mask = val!=a
    im_closest = np.argmin(absdiff[mask])
    closeval = a[mask][im_closest]
    return closeval

def F_Gaussian_Localized(octree,A,B,center, coeffs=None):
    x_per, y_per, z_per = octree.periodic()
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    x_size, y_size, z_size = xmax -xmin, ymax -ymin, zmax -zmin
    if coeffs is None:
        RelGaussian = F_Gaussian(A,B)
    else:
        RelGaussian = F_GaussianWithBackground(B,coeffs)
    def func(X,Y,Z):
        relx = cz.PeriodicDiff(X, center[0], x_size, x_per)
        rely = cz.PeriodicDiff(Y, center[1], y_size, y_per)
        relz = cz.PeriodicDiff(Z, center[2], z_size, z_per)
        relpos = (relx,rely,relz)
        values = RelGaussian(relpos)
        return values
    return func

def F_Gaussian_Params(octree, gsparams):
    A,B,coeffs,center = gsparams
    func = F_Gaussian_Localized(octree,A,B,center, coeffs=coeffs)
    return func

def MaxRadii(octree, maxmask, maxrad=.1*cu.pc, fraction=.5, debug=True):
    coords = octree.coords()
    X, Y, Z = [coords[:,i] for i in range(3)]
    maxwhere = np.where(maxmask)
    cellcoords = lambda cell: np.array((X[cell],Y[cell],Z[cell]))
    max_xyz_ = np.array([cellcoords(cell) for cell in zip(*maxwhere)])
    # Calculate raw coordinate difference matrices (not considering periodic bc)
    dist_xraw = max_xyz_[:,0,None] - max_xyz_[None,:,0]
    dist_yraw = max_xyz_[:,1,None] - max_xyz_[None,:,1]
    dist_zraw = max_xyz_[:,2,None] - max_xyz_[None,:,2]
    # Trim down coordinate differences to periodic value range if applicable
    x_per, y_per, z_per = octree.periodic()
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    x_size, y_size, z_size = xmax -xmin, ymax -ymin, zmax -zmin
    dist_x = cz.PeriodicDiff(dist_xraw, 0., x_size, x_per)
    dist_y = cz.PeriodicDiff(dist_yraw, 0., y_size, y_per)
    dist_z = cz.PeriodicDiff(dist_zraw, 0., z_size, z_per)
    # Calculate radial squared distances
    dist_sq = dist_x**2 + dist_y**2 + dist_z**2
    # For each column in the distance matrix find the radial sq distance that is closest to 0
    mindist = np.sqrt(np.array([closest(col) for col in dist_sq]))
    #
    max_rad_ = np.minimum(fraction*mindist, maxrad)
    #
    if debug>0:
        print('Closest Distances (pc):')
        print(mindist/cu.pc)
        print('MaxRadii (pc):')
        print(max_rad_/cu.pc)
    #
    return max_xyz_, max_rad_


# ==============================================================================
# ==== FIT GAUSSIANS (INTERFACE) ===============================================
# ==============================================================================

def FitFragmentsGauss(leafdata, octree, maxvar='gpot', fitvar='numdens',
                      mask=None, threshold=None, invertdir=False,
                      maxrad_cells=4, maxrad_sample=None,
                      initrad_fit=None, autorad_fit=True,
                      minfraction_fit=0.01,
                      max_vol=None, max_aspect=None, max_elong=None,
                      debug=True, **param):
    # --- Sanitize input data ---
    assert maxrad_cells in (0,1,2,3,4)
    # Maximum radius used to downselect samples for fitting:
    if maxrad_sample is None: maxrad_sample = 0.1*cu.pc
    # Radius used as initial guess for fitting the gaussian:
    if initrad_fit is None: initrad_fit = .05*cu.pc
    #
    if param:
        print('Warning: FitFragmentsGauss received unknown arguments:')
        print(param)
    # --- Create height field to map to data ---
    values = leafdata[maxvar]
    if invertdir:
        if threshold is None: threshold = np.max(values) # TODO: Feels wrong, not needed now (check on good day)
        height = np.clip(threshold-values, 0., None)
    else:
        if threshold is None: threshold = np.min(values)
        height = np.clip(values-threshold, 0., None)
    if debug>0:
        print(f'MinHeight, MaxHeight: {np.min(height), np.max(height)}')
    # --- Find local maxima ---
    print(f'-> Searching local maxima in {maxrad_cells} cell radius...')
    sys.stdout.flush()
    maxmask = GetMaxima(height, octree, nneigh=maxrad_cells)
    assert maxmask.shape == leafdata.dshape()
    # If mask is given: only accept values inside
    maxmask *= height>0.
    if mask is not None:
        maxmask *= mask
    #D#
    if debug>0:
        npeak = np.sum(maxmask)
        print(f'Retained {npeak} peaks after selection')
        sys.stdout.flush()
    # --- Loop over maxima to fit Gaussians ---
    # Read fit variable data
    if fitvar is None:
        fitvar = maxvar
        fitval = values
    else:
        fitval = leafdata[fitvar]
    max_values = fitval[maxmask]
    # Limit downselect radii by distance to closest next local max
    max_coords, max_radii = MaxRadii(octree, maxmask, maxrad=maxrad_sample)
    # Prepare output fields
    fragmasks = list()
    fraginfo = list()
    loopdata = zip(max_coords, max_radii, max_values)
    for i, (maxpos, maxrad, maxval) in enumerate(loopdata):
        relcoords = cz.BG_RelCoords(octree, maxpos)
        dsmask = DownselectForGaussian(fitval, relcoords, maxval=maxval, radius=maxrad, minfrac=minfraction_fit)
        samplepos = np.array(relcoords)[:,dsmask]
        sampleval = fitval[dsmask]
        initrad = .5*maxrad if autorad_fit else initrad_fit
        if debug>0:
            print(f'Fragment Gaussian {i+1}/{npeak}')
            print(f'Sample radius (pc): {maxrad/cu.pc}')
            print(f'Fit init. radius (pc): {initrad/cu.pc}')
        A, B, coeffs, offset, status = FitGaussianAndBG(sampleval, samplepos, posunit=initrad)
        gspos = maxpos + offset
        fwhm, eigvecs = FitFWHM(B)
        assert len(fwhm)==3
        #D#
        if debug>0:
            print('Fit status:')
            print(status)
            PrintFitInformation(gspos, maxval, fwhm)
            print(f'Offset (pc): {offset/cu.pc}')
            sys.stdout.flush()
        # Compute FWHM fragment mask
        gsmask = MaskGaussianFWHM(leafdata, octree, B, gspos)
        # Reject fragment if beyond max_vol
        if max_vol is not None:
            fragment_vol = np.sum(leafdata['vol'][gsmask])
            if fragment_vol > max_vol:
                print('FRAGMENT REJECTED (VOLUME)!')
                print()
                continue
        # Reject fragment if beyond max_aspect
        if max_aspect is not None:
            fragment_aspect = fwhm.max()/fwhm.min()
            assert fragment_aspect > 0.
            if debug>0:
                print(f'Aspect ratio: {fragment_aspect}')
            if fragment_aspect > max_aspect:
                print('FRAGMENT REJECTED (ASPECT)!')
                print()
                continue
        # Reject fragment if beyond max_elong
        if max_elong is not None:
            fragment_elong = fwhm.max()/np.median(fwhm)
            assert fragment_elong > 0.
            if debug>0:
                print(f'Elongation: {fragment_elong}')
            if fragment_elong > max_elong:
                print('FRAGMENT REJECTED (ELONGATION)!')
                print()
                continue
        print()
        # Save return parameters
        gsdict = dict(type='gaussfrag')
        infodict = dict(A=A,B=B,coeffs=coeffs,maxpos=maxpos,gspos=gspos,
                        fwhm=fwhm,eigvecs=eigvecs,initrad=initrad,maxrad=maxrad)
        gsdict.update(infodict)
        fraginfo.append(gsdict)
        # Save return fragment mask
        fragmasks.append(gsmask)
        
    if debug>0:
        print('FitFragmentsGauss: Returning...')
        sys.stdout.flush()
    assert len(fragmasks) == len(fraginfo)
    return fragmasks, fraginfo
