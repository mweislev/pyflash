# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt

import libmf.micflash as mflash
import libmf.micplot as mplot
import libmf.tools as tools

import cl_objcol as objcol
import cl_find_182 as clfind

from matplotlib.colors import LogNorm, Normalize
from matplotlib.gridspec import GridSpec
from mpl_toolkits.axes_grid1 import make_axes_locatable
from var_settings_v3 import GetVarSettings, intvar

#===============================================================================
# ==== CONSTANTS ===============================================================
#===============================================================================

from constants import *
version_str = 'centertracks017b'
homedir = os.path.expanduser('~')


#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

core_key = 'COCex8p30'
var_ch = mflash.var_ch5
plotfile_pattern = '*hdf5_plt_cnt_????'
plotfile_tmin = 13.05*Myr

# ==== FEATURE SWITCHES ========================================================
create_coldens_fig = False#True
create_perobj_res_fig = False#True
create_coldens_fig_singleax = True

# ==== LINE FORMAT SETTINGS ====================================================
std_cmap = mplot.get_cmap('jet')
paramset = { # [units, norm, cmap, label]
    'sim_mag_x': [1e-6, LogNorm(.625, 6.25), std_cmap, r'CF-B%01.2f'],
    'lrefine_max_jeans': [1, Normalize(4, 14), std_cmap, r'R$\,$%i'],
    'rho_ambient': [m_a, LogNorm(.5, 500.), std_cmap, r'%.3g$\,$u/cm$^3$'],
    'lrefine_max': [1, Normalize(4, 7), std_cmap, r'R$\,$%i'],
    }

# ==== OBJECT FORMAT SETTINGS ==================================================
mass_min, mass_max = [1e-1*M_sol, 3e+3*M_sol]
mass_unit, mass_utext = [M_sol, '$\mathrm{M}_{\odot}$']
mobj_cmap = mplot.get_cmap('jet')
mobj_norm = LogNorm(mass_min/mass_unit, mass_max/mass_unit)
mobj_label = r'Core Mass [%s]'%mass_utext

# ==== COLUMN PLOTS ============================================================
col_dgrid_ng = 0
col_plotvar = 'dens::coldens'
col_show_cbar = True

# ==== FIGURE SETTINGS =========================================================
fs = mplot.set_fig_preset('impress_col')
mplot.fig_outdir = f'coreana/{version_str}-{core_key}-{mplot.fig_target}'


#===============================================================================
# ==== HELPER ==================================================================
#===============================================================================

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

def Select(cell_list):
    if cell_list:
        return tuple(map(np.array, zip(*cell_list)))
    else:
        return tuple(4*[np.array([], dtype=np.int64),])
    
def SimPlotParam(dirpath):
    parfile = os.path.join(dirpath, 'flash.par')
    pardict = tools.readpar(parfile)
    paramkey = 'sim_mag_x'
    cunit = paramset[paramkey][0]
    cnorm = paramset[paramkey][1]
    ccmap = paramset[paramkey][2]
    setvalue = pardict[paramkey] / cunit
    simcolor = ccmap(cnorm(setvalue))
    pldict = dict(color=simcolor)
    return pldict


#===============================================================================
# ==== OBJECT BULK TRANSFORMATIONS =============================================
#===============================================================================

def ObjCenter(hdfdata, octree, objwhere):
    # Get object cell raw positions
    coords = octree.coords()
    posx_raw = coords[:,0,:,:,:][objwhere]
    posy_raw = coords[:,1,:,:,:][objwhere]
    posz_raw = coords[:,2,:,:,:][objwhere]
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
    center_unw = BG_UnwindPeriodic(octree, center)
    return np.array(center_unw)

def ObjRadius(hdfdata, octree, objwhere, objcenter):
    relcoords = BG_RelCoords(octree, objcenter)
    x, y, z = [relcoords[i][objwhere] for i in range(3)] 
    rsq = x**2 + y**2 + z**2
    radius = np.sqrt(np.max(rsq))
    return radius
    
def ObjVel(hdfdata, octree, objwhere):
    cldata = lambda key: hdfdata[key][objwhere]
    # Get object cell velocities
    velx = cldata('velx')
    vely = cldata('vely')
    velz = cldata('velz')
    # Calculate the objects bulk motion
    mass = cldata('mass')
    clvelx = np.average(velx, weights=mass)
    clvely = np.average(vely, weights=mass)
    clvelz = np.average(velz, weights=mass)
    objvel = clvelx, clvely, clvelz
    return np.array(objvel)


#===============================================================================
# ==== KERNEL ==================================================================
#===============================================================================

def KernelEpanechnikov(u):
   usq = u**2
   support = usq < 1.
   value = .75*(1.-usq)
   return support*value

def KernelGaussian(u):
   usq = u**2
   support = usq < 1.
   value = 1./np.sqrt(2.*np.pi)*np.exp(-.5*usq)
   return support*value

def KernelUniform(u):
   usq = u**2
   support = usq < 1.
   value = 1.
   return support*value

KernelFunctions = {
   'Epanechnikov':KernelEpanechnikov,
   'Gaussian':KernelGaussian,
   'Uniform':KernelUniform,
}

def BG_UnwindPeriodic(octree, coords_raw):
    x_raw, y_raw, z_raw = coords_raw
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    x_unw = tools.modclip(x_raw, xmin, xmax)
    y_unw = tools.modclip(y_raw, ymin, ymax)
    z_unw = tools.modclip(z_raw, zmin, zmax)
    return x_unw, y_unw, z_unw

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

def BG_Kernel(relcoords, radius, kernel=None):
    ''' Apply kernel function to radial grid vector '''
    KernelF = KernelFunctions.get(kernel, KernelEpanechnikov)
    x_rel, y_rel, z_rel = relcoords
    Hx = KernelF(x_rel/radius)
    Hy = KernelF(y_rel/radius)
    Hz = KernelF(z_rel/radius)
    Hk = Hx*Hy*Hz
    return Hk


#===============================================================================
# ==== Center of Mass Tracking =================================================
#===============================================================================

def TrackingWeights(hdfdata, relcoords, radius):
    w_mass = hdfdata['mass']
    assert np.all(w_mass>0.)
    w_kernel = BG_Kernel(relcoords, radius)
    dens = hdfdata['dens']
    reldens = dens/np.min(dens) + 1e-2
    w_dens = np.log10(reldens)
    assert np.all(w_dens>0.)
    try:
        assert np.any(w_kernel>0.)
        assert np.all(w_kernel>=0.)
    except AssertionError:
        print( 'WARNING! [TrackingWeights]: Kernel weights yielded all 0!')
        print( '         This could be a resolution effect.')
        print( '         Re-applying kernel with increased radius.')
        w_kernel = BG_Kernel(relcoords, 4.*radius)
        try:
            assert np.any(w_kernel>0.)
        except:
            axrel, ayrel, azrel = relcoords
            arrel = np.sqrt(axrel**2+ayrel**2+azrel**2)
            simres = hdfdata['length']
            print(f'Minimum relpos: ({axrel.min()/pc},{ayrel.min()/pc},{azrel.min()/pc})')
            print(f'Maximum relpos: ({axrel.max()/pc},{ayrel.max()/pc},{azrel.max()/pc})')
            print(f'Minimum relrad: {arrel.min()/pc} pc')
            print(f'Initial radius: {radius/pc} pc')
            minrelrad = arrel == arrel.min()
            localres = simres[minrelrad].min()
            print(f'Local simres: {localres/pc} pc')
            print(f'Worst simres: {simres.max()/pc} pc')
            print( 'WARNING! [TrackingWeights]: Increased radius, but still no weights.')
            print( '         Extending radius to 1-percentile of relative radii.')
            print( '         (This is really bad!).')
            p1radius = np.percentile(arrel, 1.)
            print(f'1-Perc. radius: {p1radius/pc} pc')
            assert p1radius > radius
            w_kernel = BG_Kernel(relcoords, p1radius)
    finally:
        weights = w_mass * w_kernel * w_dens
    assert np.all(weights>=0.)
    assert np.any(weights>0)
    return weights

def Autoguide(hdfdata, octree, center, radius):
    ''' Calculate weighted center of mass inside the given radius,
    with additional weights added to facilitate guiding along
    a dense object close to the center '''
    relcoords = BG_RelCoords(octree, center)
    # TODO: TAKE CARE OF MAKING THE RADIUS LARGE ENOUGH!
    weights = TrackingWeights(hdfdata, relcoords, radius)
    offset = [np.average(relcoords[i], weights=weights) for i in range(3)]
    center_new_raw = [center[i]+offset[i] for i in range(3)]
    center_new = BG_UnwindPeriodic(octree, center_new_raw)
    return np.array(center_new), radius

def CenterVelocity(hdfdata, octree, center, radius):
    ''' Calculate weighted velocity inside the given radius
    around the given center,
    with additional weights added to facilitate guiding along
    a dense object close to the center '''
    relcoords = BG_RelCoords(octree, center)
    weights = TrackingWeights(hdfdata, relcoords, radius)
    vel = [hdfdata[f'vel{s}'] for s in ['x','y','z']]
    vel_center = [np.average(vel[i], weights=weights) for i in range(3)]
    return np.array(vel_center)

def Track(hdfdata, octree, pos_old, vel_old, t_old, radius):
    t_new = hdfdata['real scalars/time']
    dt = t_new - t_old
    pos_expected = pos_old + dt*vel_old
    pos_new, r_new  = Autoguide(hdfdata, octree, pos_expected, radius)
    vel_new = CenterVelocity(hdfdata, octree, pos_new, radius)
    return pos_new, vel_new, t_new, r_new


#===============================================================================
# ==== Track Analysis / Visualization ==========================================
#===============================================================================

def DecodeTrack(objtrack):
    pos = [objvec[0] for objvec in objtrack]
    vel = [objvec[1] for objvec in objtrack]
    time = [objvec[2] for objvec in objtrack]
    radius = [objvec[3] for objvec in objtrack]
    arrays = tuple(map(np.array,(pos,vel,time,radius)))
    return arrays

def axPlotPositions(ax, objtrack):
    pos, vel, time, radius = DecodeTrack(objtrack)
    xyzcolors = ('red','green','blue')
    plparam = dict()
    for xyz in range(3):
        axl = ('X','Y','Z')[xyz]
        plparam['label'] = axl
        plparam['color'] = xyzcolors[xyz]
        pl = ax.plot(time/Myr, pos[:,xyz]/pc, **plparam)
    ax.set(xlabel='Time [Myr]')
    ax.set(ylabel='Center Position [pc]')
    mplot.ax_legend(ax)
    return pl

def axPlotMovement(ax, objtrack):
    pos, vel, time, radius = DecodeTrack(objtrack)
    tmid = .5 * (time[1:]+time[:-1])
    dt = time[1:]-time[:-1]
    dx = pos[1:]-pos[:-1]
    movement   = dx / dt[:,None]
    eulervel   = vel[:-1]
    correction = movement - eulervel
    xyzcolors = (['red','green','blue'])
    for xyz in range(3):
        axl = ('X','Y','Z')[xyz]
        plparam = dict(color=xyzcolors[xyz])
        pl1 = ax.plot(tmid/Myr, movement[:,xyz]/km,   linestyle='-',  label=axl, **plparam)
        pl2 = ax.plot(tmid/Myr, eulervel[:,xyz]/km,   linestyle='--', label='_', **plparam)
        pl3 = ax.plot(tmid/Myr, correction[:,xyz]/km, linestyle=':',  label='_', **plparam)
    ax.set(xlabel='Time [Myr]')
    ax.set(ylabel='Center Movement [km/s]')
    mplot.axFakeLabel(ax, 'Total', 'k-')
    mplot.axFakeLabel(ax, 'Euler', 'k--') 
    mplot.axFakeLabel(ax, 'Correction', 'k:')
    mplot.ax_legend(ax)
    return pl1, pl2, pl3

def InspectTracks(dirpath, objtracks, criterion):
    simname = os.path.basename(dirpath)
    for i,_ in enumerate(objtracks[0]):
        fig, (ax0,ax1) = plt.subplots(1,2)
        #
        objtrack = [objvectors[i] for objvectors in objtracks]
        axPlotPositions(ax0, objtrack)
        axPlotMovement(ax1, objtrack)
        #
        subpath = f'{simname}-{criterion}_{i}'
        fig.suptitle(f'Object Movement\n{simname}\n{criterion}({i})')
        mplot.savefig(fig, 'track', subpath)


#===============================================================================
# ==== TRACK FILE I/O ==========================================================
#===============================================================================

def RecordTracks(dirpath, plotfiles, objtracks, criterion, descr_prefix=None):
    simname = os.path.basename(dirpath)
    figpath = os.path.join(mplot.fig_basepath, mplot.fig_outdir)
    tools.makedir(figpath)
    #
    if descr_prefix is None:
        csvbasename = f'objtr-{simname}-{criterion}-'
    else:
        csvbasename = f'objtr-{descr_prefix}'
    # Read simulation extent from plotfile
    pf_start = plotfiles[0]
    leafdata, octree = LoadPlotfile(dirpath, pf_start, verbose=True)
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    #
    outfiles = list()
    for iobj,_ in enumerate(objtracks[0]):
        # Prepare CSV File (will hold table of object positions)
        csvname = f'{csvbasename}{iobj}.csv'
        csvpath = os.path.join(figpath,csvname)
        with open(csvpath,'w') as f:
            # WRITE HEADER
            f.write('plotfile; t_Myr; posx_pc; posy_pc; posz_pc; radius_pc; xmin_pc; ymin_pc; zmin_pc; xmax_pc; ymax_pc; zmax_pc; wrap_x; wrap_y; wrap_z')
            f.write('\n')
            # Write object position table
            objtrack = [objvectors[iobj] for objvectors in objtracks[::-1]]
            pos, vel, time, radius = DecodeTrack(objtrack)
            posx, posy, posz = [pos[:,xyz] for xyz in range(3)]
            for i,t in enumerate(time):
                #
                plotfile = os.path.join(dirpath,plotfiles[::-1][i])
                line = f'{plotfile};{t/Myr:5.2f}'
                line += f';{posx[i]/pc:8.4f};{posy[i]/pc:8.4f};{posz[i]/pc:8.4f}'
                line += f';{radius[i]/pc:8.4f}'
                #
                box_xmin = tools.modclip(posx[i]-radius[i], xmin, xmax)
                box_ymin = tools.modclip(posy[i]-radius[i], ymin, ymax)
                box_zmin = tools.modclip(posz[i]-radius[i], zmin, zmax)
                box_xmax = tools.modclip(posx[i]+radius[i], xmin, xmax)
                box_ymax = tools.modclip(posy[i]+radius[i], ymin, ymax)
                box_zmax = tools.modclip(posz[i]+radius[i], zmin, zmax)
                wrap_x = (box_xmin>posx[i]) or (box_xmax<posx[i])
                wrap_y = (box_ymin>posy[i]) or (box_ymax<posy[i])
                wrap_z = (box_zmin>posz[i]) or (box_zmax<posz[i])
                line += f';{box_xmin/pc:8.4f};{box_ymin/pc:8.4f};{box_zmin/pc:8.4f}'
                line += f';{box_xmax/pc:8.4f};{box_ymax/pc:8.4f};{box_zmax/pc:8.4f}'
                line += f';{wrap_x};{wrap_y};{wrap_z}'
                f.write(line+'\n')
        f.close()
        print(f'<< Object track {csvpath}')
        outfiles.append(csvpath)
    return outfiles


#===============================================================================
# ==== Scatter Plot Code  ======================================================
#===============================================================================
# TODO: This should live in its own scatter plot module!
# Such a module is actually present, but not yet ported from Py2,
# hence not in a working state

# Histogram parameters:
histo_res = 201

def VSBinData(leafdata, varkey):
    units, norm, cmap, label, ulabel, vartitle, weighvar = GetVarSettings(varkey)
    data = leafdata[varkey] / units
    varname = varkey.split('::')[0]
    res = int(norm.vmax-norm.vmin) if varname in intvar else histo_res
    partition = norm.inverse(np.linspace(0, 1, res+1))
    bins = np.searchsorted(partition, data.ravel())
    return bins, res

def axVarScatter(ax, leafdata, x_var, y_var,
                 weighting='vol', weights=None, normalize=True, wspan=1e6,
                 debug=False, show_cbar=True):
    # Get settings for the given axis variables
    x_units, x_norm, x_cmap, x_label, x_ulabel, x_vartitle, x_weighvar = GetVarSettings(x_var)
    y_units, y_norm, y_cmap, y_label, y_ulabel, y_vartitle, y_weighvar = GetVarSettings(y_var)
    w_units, w_norm, w_cmap, w_label, w_ulabel, w_vartitle, w_weighvar = GetVarSettings(weighting)
    # Bin data
    x_bins, x_res = VSBinData(leafdata, x_var)
    y_bins, y_res = VSBinData(leafdata, y_var)
    # Determine the weighting scheme for the 2d-heatmap
    w_weights = leafdata[weighting] / w_units
    if weights is None:
        weights = np.ones_like(w_weights)     
    hm_weights = (w_weights*weights).ravel()
    # Normalize heatmap weights
    if normalize:
        hm_weights /= hm_weights.sum()
        n_pixel = x_res*y_res
        assert n_pixel > 0
        hm_weights *= n_pixel
        wmax = .5*n_pixel
    else:
        wmax = np.sum(hm_weights)
    #
    if wmax <= 0.:
        wmax_old, wmax = wmax, 1.
        print(f'WARNING! [axVarScatter]: wmax={wmax_old}<=0. Reset to wmax={wmax}')
        print(f'         This might will yield an empty plot.')
    hm_norm = LogNorm(wmax/wspan, wmax, clip=True)
    # Compute weighted 2d-heatmap for the xy-binned data
    hm_loads = mplot.bincount2d(x_bins, y_bins, hm_weights, x_res+2, y_res+2)
    hm_data = np.clip(hm_loads[1:-1,1:-1], hm_norm.vmin, hm_norm.vmax)
    # Display the heatmap
    extent = [x_norm.vmin, x_norm.vmax, y_norm.vmin, y_norm.vmax]
    im = ax.imshow(hm_data.T, origin="lower", interpolation='none',
        extent=extent, aspect='auto', norm=hm_norm, cmap=w_cmap)
    # Format the axis ticks/labels to show them scaled to the correct norm
    # Note: While the bin-bounds are not necessarily spaced linearly,
    #       the bins itself are always supposed to be displayed equally sized.
    #       For all norms except Normalize, matplotlib has to be tricked to allow this.    
    # Format x-axis
    ax.set(xlim=extent[0:2], xlabel=f'{x_label} [{x_ulabel}]')
    xticks, xtickpos = mplot.get_ticks(x_norm)
    ax.set_xticks(xtickpos)
    ax.set_xticklabels(xticks)
    # Format y-axis
    ax.set(ylim=extent[2:4], ylabel=f'{y_label} [{y_ulabel}]')
    yticks, ytickpos = mplot.get_ticks(y_norm)
    ax.set_yticks(ytickpos)
    ax.set_yticklabels(yticks)
    # Prepare colorbar label
    if normalize:
        barlabel = f'Incidence ({w_vartitle})'
    else:
        barlabel = f'{w_vartitle} {w_label} [{w_ulabel}]'
    # Add colorbar
    if show_cbar:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, label=barlabel)
    #
    if debug:
        print('# Debug [axVarScatter]:')
        print(f'x-norm    : [{x_norm.vmin},{x_norm.vmax}]')
        print(f'y-norm    : [{y_norm.vmin},{y_norm.vmax}]')
        print(f'hm-norm   : [{hm_norm.vmin},{hm_norm.vmax}]')
        print(f'hm-val.   : [{np.nanmin(hm_data)},{np.nanmax(hm_data)}]')
        print(f'hm-weights: [{np.nanmin(hm_weights)},{np.nanmax(hm_weights)}]')
        print(f'w-weights : [{np.nanmin(w_weights)},{np.nanmax(w_weights)}]')
        print(f'weights   : [{np.nanmin(weights)},{np.nanmax(weights)}]')
        print()

    return im, cbar


#===============================================================================
# ==== Core Evolution Track Analysis ===========================================
#===============================================================================

def DecodeEvolution(evotrack, dkey):
    return np.array([data[dkey] for data in evotrack])

def ThresholdIndex(a, threshold):
    y = np.copy(a)
    y[-1] = threshold
    if a[0] > a[-1]:
        i = np.argmax(y <= threshold)
    elif a[0] < a[-1]:
        i = np.argmax(y >= threshold)
    if i == 0: i = 1
    ys = (threshold-y[i-1]) / (y[i]-y[i-1])
    i_fractional = (i-1.) + np.clip(ys,0.,1.)
    i_rounded = int(i_fractional+.5)
    return i_rounded

def ObjCollapseIndex(evotrack, debug=False):
    objres_10p = DecodeEvolution(evotrack, 'jr_10pm')
    objres_50p = DecodeEvolution(evotrack, 'jr_50pm')
    # Calculate the index for 10% of object mass below jeans res. 1
    k_10p_jr1 = ThresholdIndex(objres_10p, 1.)
    # Calculate the index for 50% of object mass below jeans res. 4
    k_50p_jr4 = ThresholdIndex(objres_50p, 4.)
    #
    k_collapse = min(k_10p_jr1,k_50p_jr4)
    #
    if debug:
        print('[ObjCollapseIndex]: Debug dump:')
        print(f'objres_10p: (Index {k_10p_jr1} for thresh. 1.)')
        print(objres_10p)
        print(f'objres_50p: (Index {k_50p_jr4} for thresh. 4.)')
        print(objres_50p)
    return k_collapse

def StaticRefinementBox(objtrack, evotrack, i_collapse, i_end=None):
    ''' Compute the static refinement box that would be needed 
        to contain the object between the given indicies,
        taking object size and object movement into account '''
    if i_end is None:
        i_end = len(objtrack)
    #
    pos, vel, time, radius = DecodeTrack(objtrack)
    #
    assert np.all(radius>0.)
    #
    box_xmin = np.min(pos[i_collapse:i_end,0] - radius[i_collapse:i_end])
    box_ymin = np.min(pos[i_collapse:i_end,1] - radius[i_collapse:i_end])
    box_zmin = np.min(pos[i_collapse:i_end,2] - radius[i_collapse:i_end])
    box_xmax = np.max(pos[i_collapse:i_end,0] + radius[i_collapse:i_end])
    box_ymax = np.max(pos[i_collapse:i_end,1] + radius[i_collapse:i_end])
    box_zmax = np.max(pos[i_collapse:i_end,2] + radius[i_collapse:i_end])
    box_extent = [[box_xmin,box_xmax],[box_ymin,box_ymax],[box_zmin,box_zmax]]
    return box_extent

def ObjectSummary(dirpath, objtracks, evotracks, criterion):
    simname = os.path.basename(dirpath)
    print(f'Object Summary: Simulation {simname}')
    # Prepare csv file
    figpath = os.path.join(mplot.fig_basepath, mplot.fig_outdir)
    csvname = f'objects-{simname}-{criterion}.csv' #TODO
    csvpath = os.path.join(figpath,csvname)
    f = open(csvpath,'w')
    # objtracks[k,i]: tracks of object position/velocity/radius
    # evotracks[k,i]: tracks of object evolutionary state
    # k: time series index
    # i: object index
    f.write(f'obj_index;obj_mass(Msol);t_collapse(Myr);x_min(cm);y_min(cm);z_min(cm);x_max(cm);y_max(cm);z_max(cm)')
    for i,_ in enumerate(objtracks[0]):
        objname = f'{criterion}({i:02})'
        # Read the record backwards, because it was recorded backwards in time.
        # The outut is then supposed to be in order of ascending time.
        objtrack = [objvectors[i] for objvectors in objtracks[::-1]]
        evotrack = [objstats[i] for objstats in evotracks[::-1]]
        assert len(objtrack) == len(evotrack)
        # Verify ascending time
        assert objtrack[0][2] < objtrack[1][2] 
        assert evotrack[0]['time'] < evotrack[1]['time'] 
        # Get statistics at begin of collapse
        k0 = ObjCollapseIndex(evotrack)
        pos0 = objtrack[k0][0]
        time0 = objtrack[k0][2]
        # Get object statistics at latest point
        objmass = evotrack[-1]['mass']
        # Get extent of static refinement box
        objbox = StaticRefinementBox(objtrack, evotrack, k0)
        ((xmin,xmax),(ymin,ymax),(zmin,zmax)) = objbox
        # Output data
        mass_str = f'{objmass/M_sol:7.2f} Msol'
        tc_str = f'{time0/Myr:5.2f} Myr'
        cf = lambda coord: f'{coord/pc:+7.3f}' # String formatting for coordinate numbers
        boxmin_str = f'({cf(xmin)},{cf(ymin)},{cf(zmin)}) pc'
        boxmax_str = f'({cf(xmax)},{cf(ymax)},{cf(zmax)}) pc'
        # Assemble output string from parts
        print(f'Object {objname}: {mass_str} {tc_str} | {boxmin_str} | {boxmax_str}')
        # Output csv
        f.write(f'\n{i:02};{objmass/M_sol};{time0/Myr:5.2f};{xmin};{ymin};{zmin};{xmax};{ymax};{zmax}')
    f.close()
    return csvpath


#===============================================================================
# ==== Core Evolution Track Visualization ======================================
#===============================================================================

def axPlotResEvolution(ax, evotrack, *args, **kwargs):
    time = DecodeEvolution(evotrack, 'time')
    # Draw resolution 50-percentile
    objres_50p = DecodeEvolution(evotrack, 'jr_50pm')
    plargs = dict(label='50-percentile', linestyle='-', color='blue')
    plargs.update(kwargs)
    pl50 = ax.plot(time/Myr, objres_50p, *args, **plargs)
    # Draw resolution 10-percentile
    objres_10p = DecodeEvolution(evotrack, 'jr_10pm')
    plargs = dict(label='10-percentile', linestyle='--', color='blue')
    plargs.update(kwargs)
    pl10 = ax.plot(time/Myr, objres_10p, *args, **plargs)
    # Draw resolution minimum
    objres_min = DecodeEvolution(evotrack, 'jr_min')
    plargs = dict(label='minimum', linestyle=':', color='blue')
    plargs.update(kwargs)
    plmin = ax.plot(time/Myr, objres_min, *args, **plargs)
    # Format axis
    ax.set(xlabel='Time [Myr]')
    ax.set(ylabel='Jeans Length [cells]', yscale='log', ylim=[1e-2,1e+3])
    # Draw helper lines
    xlim = ax.get_xlim()
    ax.plot(xlim, 2*[1.,], '-', color='dimgray')
    ax.plot(xlim, 2*[4.,], ':', color='dimgray')
    #
    ax.legend()
    return pl50, pl10, plmin
   
def axPlotMassEvolution(ax, evotrack, *args, **kwargs):
    time = DecodeEvolution(evotrack, 'time')
    objmass = DecodeEvolution(evotrack, 'mass')
    plargs = dict(color='blue')
    plargs.update(kwargs)
    pl = ax.plot(time[1:]/Myr, objmass[1:]/M_sol, *args, **plargs)
    ax.scatter(time[0]/Myr, objmass[0]/M_sol, marker='x', color='blue')
    # Format axis
    ax.set(xlabel='Time [Myr]')
    ax.set(ylabel='Object Mass [M$_\odot$]', yscale='log', ylim=[1e-2,1e+4])
    #
    return pl
    
def InspectObjectEvolution(dirpath, evotracks, criterion):
    simname = os.path.basename(dirpath)
    for i,_ in enumerate(evotracks[0]):
        evotrack = [objvectors[i] for objvectors in evotracks]
        fig, (ax0,ax1) = plt.subplots(1,2)
        axPlotResEvolution(ax0, evotrack)
        axPlotMassEvolution(ax1, evotrack)
        subpath = f'{simname}-{criterion}_{i}'
        fig.suptitle(f'Object Evolution\n{simname}\n{criterion}({i})')
        mplot.savefig(fig, 'evolution', subpath)


#===============================================================================
# ==== Core Inspection / Visualization =========================================
#===============================================================================

def ID(leafdata):
    filepath = leafdata.filename()
    simdir, filename = os.path.split(filepath)
    simname = os.path.basename(simdir)
    return simname, filename

def MarkBox(octree, center, extent, debug=False):
    ''' Mark the parts of the simulation box that are inside the extent '''
    # This version marks marks a cell if and only if its center
    # lies inside, or exactly at the lower boundary of, the given extent.
    # This is done in a somewhat inderect way to make it easier to
    # account for periodic boundary conditions, at which the extent
    # needs to be wrapped around the boundary.
    x_center, y_center, z_center = center
    x_extent, y_extent, z_extent = extent
    x_relmin, x_relmax = x_extent[0]-x_center, x_extent[1]-x_center
    y_relmin, y_relmax = y_extent[0]-y_center, y_extent[1]-y_center
    z_relmin, z_relmax = z_extent[0]-z_center, z_extent[1]-z_center
    x_rel, y_rel, z_rel = BG_RelCoords(octree, center)
    boxmask = np.ones_like(x_rel, dtype=bool)
    boxmask *= (x_relmin<=x_rel) * (x_rel<x_relmax)
    boxmask *= (y_relmin<=y_rel) * (y_rel<y_relmax)
    boxmask *= (z_relmin<=z_rel) * (z_rel<z_relmax)
    if debug:
        print('# Debug [MarkBox]:')
        print(f'center: [{x_center},{y_center},{z_center}]')
        print(f'x-extent: [{x_extent[0]},{x_extent[1]}]')
        print(f'y-extent: [{y_extent[0]},{y_extent[1]}]')
        print(f'z-extent: [{z_extent[0]},{z_extent[1]}]')
        print(f'Marked cells: [{np.sum(boxmask)}]')
        print()
    return boxmask

def RadiusExtent(center, radius):
    #x_extent = (center[0]-2.*radius, center[0]+2.*radius)
    #y_extent = (center[1]-2.*radius, center[1]+2.*radius)
    #z_extent = (center[2]-2.*radius, center[2]+2.*radius)
    x_extent = (center-2.*radius)[0], (center+2.*radius)[0]
    y_extent = (center-2.*radius)[1], (center+2.*radius)[1]
    z_extent = (center-2.*radius)[2], (center+2.*radius)[2]
    return x_extent, y_extent, z_extent

def axJeansDiagnostic(ax, leafdata, octree, boxmask):
    scplot = axVarScatter(ax, leafdata, 'rlevel', 'jeans_res',
                          weighting='mass', weights=boxmask, normalize=False)
    return scplot

def axJeansResolution(ax, leafdata, octree, boxmask):
    scplot = axVarScatter(ax, leafdata, 'dens::corelow', 'jeans_res',
                          weighting='mass', weights=boxmask, normalize=False)
    return scplot

def PlotResolutionAna(leafdata, octree, objvector):
    # Prepare figure (2 Panels)
    fig, (ax0,ax1) = plt.subplots(1,2)
    # Prepare data
    pos, vel, time, radius = objvector
    extent = RadiusExtent(pos, radius)
    boxmask = MarkBox(octree, pos, extent)
    #
    axJeansDiagnostic(ax0, leafdata, octree, boxmask)
    axJeansResolution(ax1, leafdata, octree, boxmask)
    return fig

def PlotColumns(leafdata, octree, objvector, plotvar=None):
    # Process parameters
    if plotvar is None:
        plotvar = 'dens::coldens'
    # Prepare figure: Panels yz and xz
    fig = plt.figure()
    gs = GridSpec(1, 21, top=0.98, bottom=0.02, left=0.02, right=0.98)
    ax0 = fig.add_subplot(gs[:,0:9]) # Slice (yz)
    ax1 = fig.add_subplot(gs[:,9:18], sharey=ax0 ) # Slice (xz)
    axa = [ax0, ax1]
    # Prepare column plot data
    objdata = leafdata[plotvar]
    objgrid = mflash.datagrid(objdata, octree, ng=col_dgrid_ng)
    # Prepare plot parameters
    pos, vel, time, radius = objvector
    x_extent, y_extent, z_extent = RadiusExtent(pos, radius)
    param = dict(x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
    # Plot both panels
    for ia,axis in enumerate([1,0]):
        param.update(dict(axis=axis))
        ax = axa[ia]
        map_cbar = bool(ia) if col_show_cbar else None
        objcol.axVarSurfdens(ax, objgrid, plotvar, show_cbar=map_cbar, **param)
        mplot.ax_format_ticks(ax)
        # Remove y-axis label of but the left plot:
        if ia: ax.yaxis.label.set_visible(False)
    # Remove y-tick numbers from right panel:
    plt.setp(ax1.get_yticklabels(), visible=False)
    # Put the panels directly next to each other:
    plt.subplots_adjust(wspace=.0)
    #
    return fig
    
def PlotColumn(leafdata, octree, objvector, axis=0, plotvar=None, title=None):
    # Process parameters
    if plotvar is None:
        plotvar = 'dens::coldens'
    fig, ax = plt.subplots()
    # Prepare column plot data
    objdata = leafdata[plotvar]
    objgrid = mflash.datagrid(objdata, octree, ng=col_dgrid_ng)
    # Prepare plot parameters
    pos, vel, time, radius = objvector
    x_extent, y_extent, z_extent = RadiusExtent(pos, radius)
    param = dict(x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
    # Plot panel
    param.update(dict(axis=axis,show_cbar=True))
    objcol.axVarSurfdens(ax, objgrid, plotvar, **param)
    mplot.ax_format_ticks(ax)
    ax.set_title(title)
    #
    return fig
    
def InspectObject(leafdata, octree, objvectors, criterion, simname):
    simtag, filename = ID(leafdata)
    cnt = filename[-4:]
    header = f'{simtag}_{cnt}'
    # Remove format specifier from column plotvar:
    col_varname = col_plotvar.split('::')[0]
    for i, objvec in enumerate(objvectors):
        # Make 2-panel column density plot
        subpath = f'{simtag}-{criterion}_{i}'
        if create_coldens_fig:
            fig = PlotColumns(leafdata, octree, objvec, plotvar=col_plotvar)
            fig.suptitle(f'Columns ({col_varname})\n{header}\n{criterion}({i})')
            mplot.savefig(fig, f'col{col_varname}-{cnt}', subpath)
        # Make clean single panel column density plot
        if create_coldens_fig_singleax:
            for axis in [0,1,2]:
                subpath = f'{criterion}/{simname}{i}-ax{axis}'
                axtitle = f'Object {simname}{i}'
                fig = PlotColumn(leafdata, octree, objvec, axis=axis, plotvar=col_plotvar, title=axtitle)
                mplot.savefig(fig, f'col{col_varname}-ax{axis}-{cnt}', subpath)
        # Make resolution analysis plot
        if create_perobj_res_fig:
            fig = PlotResolutionAna(leafdata, octree, objvec)
            fig.suptitle(f'Resolution\n{header}\n{criterion}({i})')
            mplot.savefig(fig, f'res-{cnt}', subpath)
    return


#===============================================================================
# ==== Core Evolution Data =====================================================
#===============================================================================

def WeightedPercentile(data, p, weights=None):
    if weights is None:
        return np.percentile(data, p)
    assert np.all(0.<=p)
    assert np.all(p<=100.)
    quantile = p*.01
    I = np.argsort(data)
    a,w = data[I], weights[I]
    cw = np.cumsum(w) / np.sum(w)
    percentile = np.interp(quantile,cw,a)
    return percentile

def MarkObjCriterion(leafdata, obj_key):
    dens = leafdata['dens']
    mask = (dens == dens)
    try:
        select_criteria = clfind.select_criteria_combos['obj_key']['criteria']
    except KeyError:
        print(f'WARNING! Criterion {obj_key} yielded no cell selection condition.')
        print(f'         Returning an all-ones mask.')
        return mask
    else:
        for criterion in select_criteria:
            data = leafdata[criterion]
            datamin, datamax = select_criteria[criterion]
            mask *= (data == np.clip(data, datamin, datamax))
        return mask

def ObjData(leafdata, octree, objvector, objwhere, debug=False):
    # Prepare data mask
    pos, vel, time, radius = objvector
    #
    data = dict()
    # Check if the number of given elements is zero.
    # This is a bit complicated, because objwhere could be either
    # a np.where argument or a bool array.
    objmask = np.zeros_like(leafdata['dens'], dtype=bool)
    objmask[objwhere] = True
    # If selection is empty, return usable object values:
    if not np.any(objmask):
        data['mass'] = 0.
        data['jr_min'] = np.inf
        data['jr_10pm'] = np.inf
        data['jr_50pm'] = np.inf
        data['r_90pm'] = 0.
        print(f'WARNING! Cell selection empty.')
        print(f'         Returning default fault values.')
    else:
        objd = lambda varkey: leafdata[varkey][objwhere]
        mass = objd('mass')
        jeansres = objd('jeans_res')
        data['mass'] = np.sum(mass)
        data['jr_min'] = np.min(jeansres)
        data['jr_10pm'] = WeightedPercentile(jeansres, 10., weights=mass)
        data['jr_50pm'] = WeightedPercentile(jeansres, 50., weights=mass)
        x_rel, y_rel, z_rel = BG_RelCoords(octree, pos)
        r = np.sqrt(x_rel**2+y_rel**2+z_rel**2)[objwhere]
        data['r_90pm'] = WeightedPercentile(r, 90., weights=mass)
    data['time'] = time
    if debug:
        print('[ObjData]: Object statistics:')
        print(data)
    return data

def InitEvoTracks(leafdata, octree, objvectors, criterion, allowempty=False):
    objects = clfind.get_clist(leafdata, octree, criterion)
    if not objects:
        if not allowempty:
            raise RuntimeError(f'InitObjectTracks: Criterion {criterion} yielded no objects!')
        else: 
            print(f'WARNING: InitObjectTracks: Criterion {criterion} yielded no objects!')
    objs_data = list()
    for i, objl in enumerate(objects):
        objwhere = Select(objl)
        objdata = ObjData(leafdata, octree, objvectors[i], objwhere)
        objs_data.append(objdata)
    return objs_data

def CoreEvoStat(leafdata, octree, objvector):
    pos, vel, time, radius = objvector
    extent = RadiusExtent(pos, radius)
    boxmask = MarkBox(octree, pos, extent)
    corestat = ObjData(leafdata, octree, objvector, boxmask)
    return corestat

def IterateEvoTracks(leafdata, octree, objvectors):
    objs_data = list()
    for i, objv in enumerate(objvectors):
        corestat = CoreEvoStat(leafdata, octree, objv)
        objs_data.append(corestat)
    return objs_data


#===============================================================================
# ==== Object Tracking Interface ===============================================
#===============================================================================

def PrintVectors(objvectors):
    for i, objvec in enumerate(objvectors):
        pos, vel, time, radius = objvec
        simtime_str = f'{time/Myr:06.3f} Myr'
        pos_str = f'({pos[0]/pc:+07.3f},{pos[1]/pc:+07.3f},{pos[2]/pc:+07.3f})pc'
        vel_str = f'({vel[0]/km:+07.3f},{vel[1]/km:+07.3f},{vel[2]/km:+07.3f})km/s'
        print(f'Object {i:02}: {simtime_str} {pos_str} {vel_str}')
    print()

def InitObjectTracks(leafdata, octree, criterion, allowempty=False):
    objects = clfind.get_clist(leafdata, octree, criterion)
    if not objects:
        if not allowempty:
            raise RuntimeError(f'InitObjectTracks: Criterion {criterion} yielded no objects!')
        else: 
            print(f'WARNING: InitObjectTracks: Criterion {criterion} yielded no objects!')
    simtime = leafdata['real scalars/time']
    routes = list()
    for i, objl in enumerate(objects):
        # Get object cell selector
        objwhere = Select(objl)
        objpos = ObjCenter(leafdata, octree, objwhere)
        objvel = ObjVel(leafdata, octree, objwhere)
        objradius = ObjRadius(leafdata, octree, objwhere, objpos)
        route = [objpos, objvel, simtime, objradius]
        routes.append(route)
    return routes

def IterateObjTracks(hdfdata, octree, objvectors_old):
    objvectors_new = list()
    for i, objv_old in enumerate(objvectors_old):
        pos_old, vel_old, t_old, r_old = objv_old
        pos_new, vel_new, t_new, r_new = Track(hdfdata, octree, pos_old, vel_old, t_old, r_old)
        objv_new = pos_new, vel_new, t_new, r_new
        objvectors_new.append(objv_new)
    return objvectors_new

def ObjectTrackingLoop(dirpath, plotfiles, criterion, simname, inspect=True):
    # Initiate Tracking at first plotfile
    pf_start = plotfiles[0]
    leafdata, octree = LoadPlotfile(dirpath, pf_start, verbose=True)
    objvectors = InitObjectTracks(leafdata, octree, criterion)
    objtracks = [objvectors,]
    evovectors = InitEvoTracks(leafdata, octree, objvectors, criterion)
    evotracks = [evovectors,]
    # Give initial output
    PrintVectors(objvectors)
    # Continue Tracking in order of plotfiles
    for n, pf_new in enumerate(plotfiles[1:]):
        leafdata, octree = LoadPlotfile(dirpath, pf_new, verbose=True)
        objvectors_new = IterateObjTracks(leafdata, octree, objtracks[-1])
        PrintVectors(objvectors_new)
        if inspect:
            InspectObject(leafdata, octree, objvectors_new, criterion, simname)
        objtracks.append(objvectors_new)
        evovectors_new = IterateEvoTracks(leafdata, octree, objvectors_new)
        evotracks.append(evovectors_new)
    return objtracks, evotracks


#===============================================================================
# ==== FILE SELECTION ==========================================================
#===============================================================================

def PF_Simtime(dirpath, filename):
    plotfile = os.path.join(dirpath, filename)
    with mflash.plotfile(plotfile, memorize=False) as hdfdata:
        simtime = hdfdata['real scalars/time']
    return simtime

def ReverseTrackFiles(dirpath, initfile, dirfiles):
    t_init = PF_Simtime(dirpath, initfile)
    t_plotfiles = np.array([PF_Simtime(dirpath, pf) for pf in dirfiles])
    It = np.argsort(-1.*t_plotfiles) # Sort in order of descending time
    T = t_plotfiles[It]
    dt = T[:-1]-T[1:]
    dt_typical = np.median(dt)
    # Filter out plotfiles that are close to or later than the init file
    useplotfile  = (T < (t_init-.1*dt_typical))
    # Filter out plotfiles earlier than tmin-margin
    useplotfile *= (T > (plotfile_tmin-.5*dt_typical))
    # Filter out plotfiles that are too close to their next file
    # useplotfile[1:] *= (dt > .5*dt_typical)
    # 
    dirfile_sortargs = It[useplotfile]
    dirfiles_sorted = [dirfiles[i] for i in dirfile_sortargs]
    return dirfiles_sorted


#===============================================================================
# ==== MAIN ====================================================================
#===============================================================================

def Main(dirpath, initfile, dirfiles, core_key, simname):
    trackfiles = ReverseTrackFiles(dirpath, initfile, dirfiles)
    trackfiles.insert(0, initfile)
    print
    objtracks, evotracks = ObjectTrackingLoop(dirpath, trackfiles, core_key, simname)
    RecordTracks(dirpath, trackfiles, objtracks, core_key)
    InspectTracks(dirpath, objtracks, core_key)
    InspectObjectEvolution(dirpath, evotracks, core_key)
    ObjectSummary(dirpath, objtracks, evotracks, core_key)


#===============================================================================
# ==== LOBBY ===================================================================
#===============================================================================

if __name__ == '__main__':
    simroot = '/home/mweis/scratch/CF-ab/'
    initfiles = [
        #'CF-B0.01a/CF97W6R_N1B0000X25_hdf5_chk_0027',
        #'CF-B0.01b/CF97W6R_N1B0000X_IFC_hdf5_chk_0039',
        #'CF-B1.25a/CF97W6R_N1B0125X22_hdf5_chk_0025',
        #'CF-B1.25b/CF97W6R_N1B0125_IFCX_hdf5_chk_0029',
        'CF-B2.50a/CF97W6R_N1B0250X30_hdf5_chk_0031',
        #'CF-B2.50b/CF97W6R_N1B0250X_IFC_hdf5_chk_0041',
        'CF-B5.00a/CF97W6R_N1B0500X41_hdf5_chk_0042',
        #'CF-B5.00b/CF97W6R_N1B0500_IFCX_hdf5_chk_0032',
    ]
    tools.CheckFiles([os.path.join(simroot,pf) for pf in initfiles])
    
    #simname = ['Aa', 'Ab', 'Ba', 'Bb', 'Ca', 'Cb', 'Da', 'Db',]
    simnames = ['Ca', 'Da']

    dirdict = tools.scanpath(simroot, plotfile_pattern)

    print('>> Checking plotfile libraries:')
    for initfile in initfiles:
        initfp = os.path.join(simroot,initfile)
        simdir, initfn = os.path.split(initfp)
        entry_ok = simdir in dirdict
        print(f'[{entry_ok}] {simdir}')
        if not entry_ok: raise IOError(f'Plotfile library not found: {simdir}')

    for simname,initfile in zip(simnames,initfiles):
        initfp = os.path.join(simroot,initfile)
        simdir, initfn = os.path.split(initfp)
        dirfiles = dirdict[simdir]
        Main(simdir, initfn, dirfiles, core_key, simname)
        
    print('FINISHED!')
