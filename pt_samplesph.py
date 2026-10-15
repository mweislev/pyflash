import os, sys
import numpy as np
import matplotlib.pylab as plt

import libmf.micflash as mflash
import libmf.micplot as mplot
import libmf.tools as tools

import cl_objcol as objcol
import pt_partcol as partcol

import constants as cu

__author__ = "Michael Weis"
__version__ = "0.0.1.0"


#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

# ==== VISUALIZATION SETTINGS ==================================================
do_plot = True
do_coldens = True
do_partcol = True
fs = mplot.set_fig_preset('impress_14')
mplot.fig_outdir = 'samplesph'
col_dgrid_ng = 0
scatterargs = dict(marker='x', color='green', lw=.8, s=1., alpha=.2)
proj_spec = 'coldens'
npart_plot = 5000
colres = None, None, None
#colres = 2048,512,512

# ==== BASE SETTINGS ===========================================================
var_ch = mflash.var_ch5 # XXX TRIPLE-CHECK ! XXX

# ==== SPH SAMPLING ============================================================
npart_default = None
partmass_default = 1e-3*cu.M_sol
csv_outdir = '/home/mweis/scratch/particles/'
  
sph_settings = {
    'glass': dict(),
    'gas': dict(densvar='dens'),
    'center': dict(densvar='dens',
                 x_extent=[-16.*cu.pc,16.*cu.pc]),
    'reg1': dict(densvar='dens',
                 x_extent=[-4.*cu.pc,4.*cu.pc]),
    'reg2': dict(densvar='dens_co',
                 x_extent=[-1.5e+19,0.0],
                 y_extent=[-2.39e+19,-1.6e+19],
                 z_extent=[2.29e+18,1e+19]),
    'CoreAa01_gas': dict(densvar='dens',
                 x_extent=[1.1*cu.pc,2.3*cu.pc],
                 y_extent=[-13.3*cu.pc,-12.6*cu.pc],
                 z_extent=[-4.*cu.pc,-3.3*cu.pc]),
    'CoreAa04_gas': dict(densvar='dens',
                 x_extent=[-3.3*cu.pc,-2.5*cu.pc],
                 y_extent=[-6.9*cu.pc, -6.4*cu.pc],
                 z_extent=[2.35*cu.pc, 2.85*cu.pc]),
    'CoreAa01_1915': dict(densvar='dens',
                 x_extent=[+0.49419824E+19-0.70*cu.pc, +0.49419824E+19+0.70*cu.pc],
                 y_extent=[-0.39947188E+20-0.35*cu.pc, -0.39947188E+20+0.35*cu.pc],
                 z_extent=[-0.11208298E+20-0.45*cu.pc, -0.11208298E+20+0.45*cu.pc]),
    'CoreAa04_1915': dict(densvar='dens',
                 x_extent=[-0.91393729E+19-0.50*cu.pc, -0.91393729E+19+0.50*cu.pc],
                 y_extent=[-0.20264328E+20-0.25*cu.pc, -0.20264328E+20+0.25*cu.pc],
                 z_extent=[+0.77448241E+19-0.35*cu.pc, +0.77448241E+19+0.35*cu.pc]),
    'CoreAb09_gas': dict(densvar='dens',
                 x_extent=[-0.0231*cu.pc,0.1670*cu.pc],
                 y_extent=[11.0779*cu.pc,11.2679*cu.pc],
                 z_extent=[-14.6841*cu.pc,-14.4940*cu.pc]),
    'CoreAb_lm1_gas': dict(densvar='dens',
                 x_extent=[-.3*cu.pc, 1.1*cu.pc],
                 y_extent=[-2.75*cu.pc, -2.0*cu.pc],
                 z_extent=[.9*cu.pc, 2.1*cu.pc]),
    'CoreAb_lm2_gas': dict(densvar='dens',
                 x_extent=[-1.65*cu.pc,+0.2*cu.pc],
                 y_extent=[+10.7*cu.pc,+12.4*cu.pc],
                 z_extent=[-15.32*cu.pc,-13.25*cu.pc]),
    'CoreCb_lm1_gas': dict(densvar='dens',
                 x_extent=[0.0*cu.pc,0.95*cu.pc],
                 y_extent=[1.5*cu.pc,2.3*cu.pc],
                 z_extent=[4.1*cu.pc,5.3*cu.pc]),
    'CoreCb_hm1_gas': dict(densvar='dens',
                 x_extent=[-.1*cu.pc,1.25*cu.pc],
                 y_extent=[-5.55*cu.pc,-4.95*cu.pc],
                 z_extent=[-6.4*cu.pc,-5.4*cu.pc]),
    'CoreCb_hm2_gas': dict(densvar='dens',
                 x_extent=[0.2*cu.pc,0.7*cu.pc],
                 y_extent=[7.7*cu.pc,8.25*cu.pc],
                 z_extent=[-5.2*cu.pc,-4.75*cu.pc]),
    'Reg6D_gas': dict(densvar='dens',
                 x_extent=[-0.10478374E+19-0.93E+18,-0.10478374E+19+0.93E+18],
                 y_extent=[0.35590111E+20-0.93E+18,0.35590111E+20+0.93E+18],
                 z_extent=[-0.44844279E+20-0.93E+18,-0.44844279E+20+0.93E+18]),
}


#===============================================================================
# ==== PLOTTING ================================================================
#===============================================================================

def FormatAxis(ax, extent, axis=0):
    iax, iay = objcol.ax_xy_order[axis]
    axunit, axulabel = cu.pc, 'pc'
    xs_extent = extent[iax]
    ys_extent = extent[iay]
    if xs_extent is not None:
        ax.set_xlim(np.array(extent[iax]) / axunit)
    if ys_extent is not None:
        ax.set_ylim(np.array(extent[iay]) / axunit)
    ax.set_aspect('equal')
    objcol.axFormat(ax, axis, None, None, axulabel, show_title=False)
    return ax

def ExtentOfMask(leafdata, octree, mask):
    if not np.any(mask):
        return ((None,None),(None,None),(None,None))
    block_bbox = leafdata['bounding box']
    bbox_min = block_bbox.min(axis=2)
    bbox_max = block_bbox.max(axis=2)
    domain_min = bbox_min.min(axis=0)
    domain_max = bbox_max.max(axis=0)
    if np.all(mask):
        x_extent = (domain_min[0], domain_max[0])
        y_extent = (domain_min[1], domain_max[1])
        z_extent = (domain_min[2], domain_max[2])
        return x_extent, y_extent, z_extent
    domain_radius = .5*(domain_max-domain_min)
    drx, dry, drz = domain_radius
    csx, csy, csz = leafdata['lenx'], leafdata['leny'], leafdata['lenz']
    coords = octree.coords()
    x, y, z = (coords[:,i] for i in range(3))
    x_avg = np.median(x[mask])
    y_avg = np.median(y[mask])
    z_avg = np.median(z[mask])
    ip = np.argmin((x[mask]-x_avg)**2+(y[mask]-y_avg)**2+(z[mask]-z_avg)**2)
    x_pivot = np.ravel(x[mask])[ip]
    y_pivot = np.ravel(y[mask])[ip]
    z_pivot = np.ravel(z[mask])[ip]
    x_wrapped = mflash.modclip(x, x_pivot-drx, x_pivot+drx)
    y_wrapped = mflash.modclip(y, y_pivot-dry, y_pivot+dry)
    z_wrapped = mflash.modclip(z, z_pivot-drz, z_pivot+drz)
    xmin, xmax = x_wrapped -.5*csx, x_wrapped +.5*csx
    ymin, ymax = y_wrapped -.5*csy, y_wrapped +.5*csy
    zmin, zmax = z_wrapped -.5*csz, z_wrapped +.5*csz
    x_extent = (np.min(xmin[mask]), np.max(xmax[mask]))
    y_extent = (np.min(ymin[mask]), np.max(ymax[mask]))
    z_extent = (np.min(zmin[mask]), np.max(zmax[mask]))
    return x_extent, y_extent, z_extent

def PlotColumn(ax, leafdata, octree, extent, axis=0, plotvar=None, title=None, objmask=None, **kwargs):
    # Process parameters
    if plotvar is None:
        plotvar = 'dens::coldens'
    # Prepare column plot data
    objdata = leafdata[plotvar]
    if objmask is not None:
        objdata *= objmask
    # Prepare data grid
    objgrid = mflash.datagrid(objdata, octree, ng=col_dgrid_ng)
    # Prepare plot parameters
    x_extent, y_extent, z_extent = extent
    param = dict(x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
    # Plot panel
    param.update(dict(axis=axis,show_cbar=True))
    param.update(kwargs)
    pl = objcol.axVarSurfdens(ax, objgrid, plotvar, **param)
    #
    mplot.ax_format_ticks(ax)
    if title is not None:
        ax.set_title(title)
    #
    return pl

def PlotParticles(ax, partpos, axis=0, **kwargs):
    iax, iay = objcol.ax_xy_order[axis]
    Xs = np.array(partpos[iax])/cu.pc
    Ys = np.array(partpos[iay])/cu.pc
    npart = len(Xs)
    stride = npart//npart_plot + 1
    scparam = dict()
    scparam.update(scatterargs)
    scparam.update(kwargs)
    sc = ax.scatter(Xs[::stride], Ys[::stride], **scparam)
    return sc

axa = [0,2,1]

def InspectParticles(leafdata, octree, mask, partpos, partmass, setname):
    sphset = sph_settings[setname]
    if do_partcol:
        fig, axs2 = plt.subplots(2,2)
        axs = axs2[0]
        axq = axs2[1]
    else:
        fig, axs = plt.subplots(1,2)
        axq = list()
    x_extent = sphset.get('x_extent', None)
    y_extent = sphset.get('y_extent', None)
    z_extent = sphset.get('z_extent', None)
    extent = x_extent, y_extent, z_extent
    #extent = ExtentOfMask(leafdata, octree, mask)
    #print('Mask x_extent:', np.array(extent[0])/cu.pc)
    #print('Mask y_extent:', np.array(extent[1])/cu.pc)
    #print('Mask z_extent:', np.array(extent[2])/cu.pc)
    for ia,ax in enumerate(axs):
        axchar = ('x','y','z')[axa[ia]]
        if do_coldens:
            field = sphset.get('densvar', 'dens')
            plvar = f'{field}::{proj_spec}'
            colargs = dict(axis = axa[ia], title=None, objmask=None, plotvar=plvar, res=colres)
            ax.set_title(f'Surface density ({field},{axchar})')
            PlotColumn(ax, leafdata, octree, extent, **colargs)
        scargs = dict(axis = axa[ia])
        PlotParticles(ax, partpos, **scargs)
        FormatAxis(ax, extent, axis=axa[ia])
    for ia,ax in enumerate(axq):
        axchar = ('x','y','z')[axa[ia]]
        ax.set_title(f'Tracer surface density ({axchar})')
        partcol.axPlotTracerColumn(ax, partpos, partmass=partmass, axis=axa[ia], extent=extent)

    simname, filename = ID(leafdata)
    plname = f'{filename}-sphsample-{setname}'
    mplot.savefig(fig, plname, subpath=simname)


#===============================================================================
# ==== SPH SAMPLING ============================================================
#===============================================================================

def SampleSph(leafdata, octree, densvar=None, mask=None, npart=None, **kwargs):
    vol = leafdata['vol']
    if densvar is None:
        dens = np.ones_like(vol)
    else:
        dens = leafdata[densvar]
    if mask is None:
        mask = np.ones_like(vol)
    massFlat = (dens*vol*mask).ravel()
    # Decide how many particles to place
    mass_total = np.sum(massFlat)
    if npart is None:
        partmass = kwargs.get('partmass', partmass_default)
        npart = round(mass_total/partmass)
        if not npart>0:
            return np.array([]), np.array([]), np.array([])
    else:
        partmass = mass_total / npart
    # Decide in which cells to put the sample particles
    massCumsum = np.cumsum(massFlat)
    partNorm = np.random.rand(npart)
    partIMass = partNorm * massCumsum[-1]
    partCellIndex = np.searchsorted(massCumsum, partIMass)
    Bl, K, J, I = np.unravel_index(partCellIndex, dens.shape)
    # Determine cell center positions for each sample particle
    coords = octree.coords()
    X, Y, Z = [coords[:,i] for i in range(3)]
    partX = X[Bl,K,J,I]
    partY = Y[Bl,K,J,I]
    partZ = Z[Bl,K,J,I]
    # Determine the size of each cell
    blocksize = leafdata['block size']
    nxb = leafdata['integer scalars/nxb']
    nyb = leafdata['integer scalars/nyb']
    nzb = leafdata['integer scalars/nzb']
    cellsize = blocksize / np.array([nxb, nyb, nzb]) # Format: [Block,Dir]
    # Randomize particle positions inside cells
    xyzRelOffset = np.random.rand(npart,3) -.5  # Format: [Particle,Dir]
    xyzOffset = xyzRelOffset*cellsize[Bl]
    partX += xyzOffset[:,0]
    partY += xyzOffset[:,1]
    partZ += xyzOffset[:,2]
    #
    return (partX, partY, partZ), partmass


#===============================================================================
# ==== FILE I/O ================================================================
#===============================================================================

def ID(leafdata):
    filepath = leafdata.filename()
    simdir, filename = os.path.split(filepath)
    simname = os.path.basename(simdir)
    return simname, filename

def ParticleFileloc(leafdata):
    simname, fn = ID(leafdata)
    csvfile = f'{fn}-particles.csv'
    csvdir = os.path.join(csv_outdir,simname)
    return csvdir, csvfile

def WriteParticles(leafdata, partpos, istart=0, partinfo=None):
    partX, partY, partZ = partpos
    # Sanity check: Fields equally sized?
    npart = len(partX)
    assert npart > 0
    assert len(partY) == npart
    assert len(partZ) == npart
    #
    csvdir, csvfile = ParticleFileloc(leafdata)
    tools.makedir(csvdir)
    csvpath = os.path.join(csvdir,csvfile)
    #
    mode = 'w' if istart == 0 else 'a'    
    with open(csvpath,mode) as f:
        # Write informational hints
        if partinfo is not None:
            f.write('! ')
            for ie,(key,val) in enumerate(partinfo.items()):
                if ie: f.write(' | ')
                f.write(f'{key}:{val}')
            f.write('\n')
        # Write header
        f.write('# partindex')
        f.write('           partposx_cm')
        f.write('           partposy_cm')
        f.write('           partposz_cm')
        f.write('\n')
        # Write particles
        for i,pos in enumerate(zip(*partpos)):
            ipart = istart + i
            f.write(f'{ipart:11}')
            f.write(f'{pos[0]:22.13e}')
            f.write(f'{pos[1]:22.13e}')
            f.write(f'{pos[2]:22.13e}')
            f.write('\n')
        f.close()
    #
    return csvpath

def LoadPlotfile(dirpath, filename, verbose=False):
    plotfile = os.path.join(dirpath, filename)
    if verbose > 0:
        print(f'\n>> Processing {plotfile}')
    assert os.path.isfile(plotfile)
    # Open plotfile data
    leafdata = mflash.plotfile(plotfile, memorize=False)
    leafdata.learn(mflash.var_mhd)
    leafdata.learn(mflash.var_grid)
    leafdata.learn(var_ch)
    # Get octree from plotfile
    octree = mflash.pm3dgrid(leafdata)
    return leafdata, octree


#===============================================================================
# ==== MASKING FUNCTIONS =======================================================
#===============================================================================

def BG_RelCoords(octree, center):
    ''' Determine radial vector from center to cells,
    while rotating the coordinate system window to center the center '''
    coords = octree.coords()
    x_raw, y_raw, z_raw = [coords[:,i,:,:,:] for i in range(3)]
    x_ctr, y_ctr, z_ctr = center
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    xsize, ysize, zsize = xmax -xmin, ymax -ymin, zmax -zmin
    x_rel = tools.modclip(x_raw -x_ctr, -.5*xsize, +.5*xsize)
    y_rel = tools.modclip(y_raw -y_ctr, -.5*ysize, +.5*ysize)
    z_rel = tools.modclip(z_raw -z_ctr, -.5*zsize, +.5*zsize)
    return x_rel, y_rel, z_rel

def MaskExtent(octree, x_extent=None, y_extent=None, z_extent=None, center=None, debug=False, **kwargs):
    ''' Mark the parts of the simulation box that are inside the extent '''
    # This version marks marks a cell if and only if its center
    # lies inside, or exactly at the lower boundary of, the given extent.
    # This is done in a somewhat inderect way to make it easier to
    # account for periodic boundary conditions, at which the extent
    # needs to be wrapped around the boundary.
    if center is None:
        center = 0., 0., 0.
    x_rel, y_rel, z_rel = BG_RelCoords(octree, center)
    boxmask = np.ones_like(x_rel, dtype=bool)
    x_center, y_center, z_center = center
    if x_extent is not None:
        x_relmin, x_relmax = x_extent[0]-x_center, x_extent[1]-x_center
        boxmask *= (x_relmin<=x_rel) * (x_rel<x_relmax)
    if y_extent is not None:
        y_relmin, y_relmax = y_extent[0]-y_center, y_extent[1]-y_center
        boxmask *= (y_relmin<=y_rel) * (y_rel<y_relmax)
    if z_extent is not None:
        z_relmin, z_relmax = z_extent[0]-z_center, z_extent[1]-z_center
        boxmask *= (z_relmin<=z_rel) * (z_rel<z_relmax)
    if debug:
        print('# Debug [MaskExtent]:')
        print(f'center: [{x_center},{y_center},{z_center}]')
        print(f'x-extent: [{x_extent}]')
        print(f'y-extent: [{y_extent}]')
        print(f'z-extent: [{z_extent}]')
        print(f'# marked cells: [{np.sum(boxmask)}]')
        if kwargs:
            print('Ignored arguments:')
            print(kwargs)
    return boxmask


#===============================================================================
# ==== MAIN ====================================================================
#===============================================================================

def WriteParticleSet(leafdata, octree, setname, istart=0):
    settings = sph_settings[setname]
    npart = settings.pop('npart', npart_default)
    print(f'Setname: {setname}')
    xmin, xmax = settings.get('x_extent', (np.nan,np.nan))
    ymin, ymax = settings.get('y_extent', (np.nan,np.nan))
    zmin, zmax = settings.get('z_extent', (np.nan,np.nan))
    print(f'Pos: {.5*(xmax+xmin)}, {.5*(ymax+ymin)}, {.5*(zmax+zmin)}')
    print(f'Rad: {.5*(xmax-xmin)}, {.5*(ymax-ymin)}, {.5*(zmax-zmin)}')
    print('>< Masking extent...', end=' ')
    sys.stdout.flush()
    exmask = MaskExtent(octree, **settings)
    print('OK')
    print('>< Computing particle positions...', end=' ')
    sys.stdout.flush()
    partpos, partmass = SampleSph(leafdata, octree, mask=exmask, npart=npart, **settings)
    npart_set = len(partpos[0])
    print(f'OK, placed {npart_set} particles of {partmass/cu.M_sol} M_sol')
    print('<- Saving particle list...', end=' ')
    sys.stdout.flush()
    pltfile = os.path.basename(leafdata.filename())
    partinfo = dict(plotfile=pltfile, setname=setname)
    csvpath = WriteParticles(leafdata, partpos, istart=istart, partinfo=partinfo)
    print(csvpath)
    npart = istart + npart_set
    if do_plot:
        print('>< Plotting particle positions...')
        InspectParticles(leafdata, octree, exmask, partpos, partmass, setname)
    return csvpath, npart

def CreateParticleFile(plotfile, sph_sample_sets):
    npart = 0
    leafdata, octree = LoadPlotfile('', plotfile)
    for setname in sph_sample_sets:
        print(f'-- Processing {setname} particle set')
        csvpath, npart = WriteParticleSet(leafdata, octree, setname, istart=npart)
    return csvpath
    

#===============================================================================
# ==== LOBBY ===================================================================
#===============================================================================

if __name__ == '__main__':
    plotfile = sys.argv[1]
    sph_sample_sets = sys.argv[2:]
    CreateParticleFile(plotfile, sph_sample_sets)
