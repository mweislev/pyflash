# -*- coding: utf-8 -*-
import os, sys
import numpy as np
import matplotlib as mpl
import matplotlib.pyplot as plt
import matplotlib.colors as cl

import libmf.micflash as mflash
import libmf.micplot as mplot
import libmf.tools as tools

import cl_fofg_142 as clfof
import cz_tools as cztools
import cl_find_182 as clfind

from mpl_toolkits.axes_grid1 import make_axes_locatable
from matplotlib.gridspec import GridSpec

from var_settings_v3 import var_settings, intvar, GetVarSettings

plt.switch_backend('agg')

__author__ = "Michael Weis"
__version__ = "0.0.1.0"


#===============================================================================
# ==== CONSTANTS ===============================================================
#===============================================================================

from constants import *
version_str = 'clcolgal006c'

#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

# ==== FEATURE SETTINGS ========================================================
show_center_of_mass = True
show_gpot_minima = False
gpot_min_nneigh = 2
dgrid_ng = 0
center_format = dict(marker='x', c='lime', label='CoG.', alpha=.5)
minima_format = dict(marker='x', c='salmon', label='Min.', alpha=.5)
figtitle_force = False

# ==== DATA SETTINGS ===========================================================
proj_vars = ['dens::coldens',]
core_key = 'COCex8p30:dendroleaf'
var_ch = mflash.var_ch5
sphere_relradius = 1.5

# ==== FIGURE SETTINGS =========================================================
mplot.fig_target = 'mnras_col_narrow'
mplot.fig_outdir = 'objgal/%s-%s'%(version_str,mplot.fig_target)
bbox = dict(boxstyle="round", fc="lightgrey")
show_map_cbar = True
grid_interpolation = None
imshow_interpolation = None
res_std = [None,None,None]

# ==== CONTOUR SETTINGS ========================================================
contours = { 
#    'abund_co': dict(thresh=[1e-4,], ls=[':',], col=['cornflowerblue',], lw=1.),
}

# ==== TECH SETTINGS ===========================================================
verbose = 2


#===============================================================================
# ==== PLOT HELPER =============================================================
#===============================================================================

fs = mplot.get_fig_preset()

def axFakeLabel(ax, text, *args, **kwargs):
    ''' Add a specific label to an ax legend without plotting anything '''
    kwargs.update(dict(label=text))
    ax.plot([np.nan, np.nan], [np.nan, np.nan], *args, **kwargs)

def GetClBox(leafdata, octree, selector, clcenter):
    # Set center of mass as object center
    center = clcenter['clposx'], clcenter['clposy'], clcenter['clposz']
    # Get domain cell center coordinates relative to object center,
    # where the domain is centered around the gravitational minimum above
    domain_xrel, domain_yrel, domain_zrel = cztools.BG_RelCoords_D(leafdata, octree, center)
    domain_r = np.sqrt(domain_xrel**2 +domain_yrel**2 +domain_zrel**2)
    # Get radius of sphere around center encompassing the clump/core
    cl_r = domain_r[selector]
    clradius = np.max(cl_r)
    sphere_radius = sphere_relradius *clradius
    ## Create slice plots
    x_extent = center[0]-1.05*sphere_radius, center[0]+1.05*sphere_radius
    y_extent = center[1]-1.05*sphere_radius, center[1]+1.05*sphere_radius
    z_extent = center[2]-1.05*sphere_radius, center[2]+1.05*sphere_radius
    return center, x_extent, y_extent, z_extent
    
def GetClGrid(leafdata, octree, selector, foreign_selectors):
    # Get datashape
    datashape = leafdata['dens'].shape
    #
    clump_fmask = np.zeros(datashape, dtype=float)
    clump_fmask[selector] = 1.
    clmask_grid = mflash.datagrid(clump_fmask, octree, ng=dgrid_ng, verbose=verbose-1)
    #
    foreign_fmask = np.zeros(datashape, dtype=float)
    if foreign_selectors is not None:
        for fsel in foreign_selectors:
            foreign_fmask[fsel] = 1.
        foreign_fmask[selector] = 0.
        foreign_fmgrid = mflash.datagrid(foreign_fmask, octree, ng=dgrid_ng, verbose=verbose-1)
    else:
        foreign_fmgrid = None
    
    return clmask_grid, foreign_fmgrid

    
#===============================================================================
# ==== SLICE HELPER/BITS =======================================================
#===============================================================================

ax_xy_order = {0:[1,2], 1:[0,2], 2:[0,1]}

def axFormat(ax, axis, offset, vartitle, axulabel, show_axlabel=True, show_title=True):
   if show_axlabel:
      axustr = ' [%s]'%axulabel
      ax_x, ax_y = ax_xy_order[axis]
      ax.set(xlabel=['x','y','z'][ax_x]+axustr, ylabel=['x','y','z'][ax_y]+axustr)
   if show_title:
      slice_title = vartitle
      if offset:
         slice_title += ' ('
         slice_title +=  ['x','y','z'][axis]
         slice_title += '= %.3g %s'%(offset, axulabel)
         slice_title += ')'
      ax.set_title(slice_title)
   return ax   


#===============================================================================
# ==== BOX PROJECTION MARKER ===================================================
#===============================================================================

def axMarkCenter(ax, hdfdata, octree, obji_stat, axis=0, *args, **kwargs):
    ax_x, ax_y = {0:[1,2], 1:[0,2], 2:[0,1]}[axis]
    kX = ['clposx','clposy','clposz'][ax_x]
    kY = ['clposx','clposy','clposz'][ax_y]
    X = obji_stat[kX]/pc
    Y = obji_stat[kY]/pc
    pl = ax.scatter(X,Y, *args, **kwargs)
    return pl

def axMarkMinima(ax, hdfdata, octree, obji_mincoords, axis=0, *args, **kwargs):
    ax_x, ax_y = {0:[1,2], 1:[0,2], 2:[0,1]}[axis]
    Xcgs_ = np.array([obji_min[ax_x] for obji_min in obji_mincoords])
    Ycgs_ = np.array([obji_min[ax_y] for obji_min in obji_mincoords])
    X, Y = Xcgs_/pc, Ycgs_/pc
    pl = ax.scatter(X,Y, *args, **kwargs)
    return pl

def ObjMinima(hdfdata, octree, varkey, obj_list, min_nneigh=4):
    obj_minlists = list()
    if not obj_list:
        return obj_minlists
        
    potential = hdfdata[varkey] # TODO: directions

    baseshape = potential.shape
    obj_masks = [clfind.GetMask(baseshape, objl) for objl in obj_list]
    allobj_mask = np.logical_or.reduce(obj_masks)
    
    allobj_minmask = clfof.GetMinima(potential, octree, allobj_mask, nneigh=min_nneigh)
    
    for i, objmask in enumerate(obj_masks):
        obj_minmask = objmask*allobj_minmask
        nmin = np.sum(obj_minmask)
        print(f'Object {i+1}: Found {nmin} minima.')
        obj_where = np.where(obj_minmask)
        obj_minlist = clfind.lzip(*obj_where)
        obj_minlists.append(obj_minlist)
    
    return obj_minlists

def ObjCellCoords(octree, objstat, obj_cells):
    assert len(objstat) == len(obj_cells)
    #
    coord_list = list()
    #
    if not obj_cells:
        return coord_list
    # Get domain layout data
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    xsize, ysize, zsize = xmax -xmin, ymax -ymin, zmax -zmin
    coords = octree.coords()
    #
    objcell_selectors = list(map(clfind.select, obj_cells))
    for i, selector in enumerate(objcell_selectors):
        data = objstat[i]
        # Get minima cell raw positions
        posx_raw = coords[:,0,:,:,:][selector]
        posy_raw = coords[:,1,:,:,:][selector]
        posz_raw = coords[:,2,:,:,:][selector]
        # Rotate the coordinate system window to center the objects center of mass
        minposx = tools.modclip(posx_raw, data['clposx']-.5*xsize, data['clposx']+.5*xsize)
        minposy = tools.modclip(posy_raw, data['clposy']-.5*ysize, data['clposy']+.5*ysize)
        minposz = tools.modclip(posz_raw, data['clposz']-.5*zsize, data['clposz']+.5*zsize)
        objcoords = list(zip(minposx,minposy,minposz))
        coord_list.append(objcoords)
    return coord_list

#===============================================================================
# ==== BOX PROJECTION IMAGER ===================================================
#===============================================================================

def axVarSurfdens(ax, dgrid, var, axis=2,
            x_extent=None, y_extent=None, z_extent=None, norm=None, avgoffset=0.0,
            show_cbar=True, show_axlabel=True, show_title=False, leafdata=None, objmask=None,
            axunit=None, axulabel=None, res=None, alpha=1.0, extreturn=False):

    if dgrid is None:
        return None, None
    if res is None:
        res = res_std
    
    units, varnorm, hist_cmap, varlabel, ulabel, vartitle, weighvar = GetVarSettings(var)
    barlabel = '%s [%s]'%(varlabel,ulabel)
    
    if norm is None:
        norm = varnorm
        
    print(f'#DEBUG# norm vmin,vmax: {norm.vmin}, {norm.vmax}')
    print(f'varkey: {var}, weighvar: {weighvar}')
    #
    if weighvar is None or weighvar == 'vol':
        im_data = dgrid.integrate_axis(axis,
            interpolation=grid_interpolation,
            x_extent=x_extent, y_extent=y_extent, z_extent=z_extent,
            xres=res[0], yres=res[1], zres=res[2]) / units
    elif leafdata is not None:
        # If the weight var is extensive (i.e. it increases with volume),
        # we need to remove the volume, since the integration sample points
        # are assumed to be uniformly spaced, which already gives intrinsic weighting by volume.
        weight_data = leafdata[weighvar]
        vol_data = leafdata['vol']
        sample_weights = weight_data / vol_data

        if objmask is None:
            objmask = np.ones_like(vol_data, dtype=bool)
            print('Warning: axVarSurfdens: No object mask provided')
        sample_weights *= objmask

        im_data = (avgoffset + dgrid.average_axis(axis,
            interpolation=grid_interpolation, weight_data=sample_weights,
            x_extent=x_extent, y_extent=y_extent, z_extent=z_extent,
            xres=res[0], yres=res[1], zres=res[2])) / units
    else:
        raise RuntimeError('Weighted column averaging required, but leafdata argument NOT provided.')
    
        
    print(f'#DEBUG# im_data min,max: {np.min(im_data)}, {np.max(im_data)}')

    # Read simulation domain boundary coordinates
    extent_cgs = dgrid.fgrid.extent(axis)
    ax_x, ax_y = ax_xy_order[axis]
    if (x_extent, y_extent, z_extent)[ax_x] is not None:
        #extent_cgs[0] = np.array((x_extent, y_extent, z_extent))[ax_x]
        extent_cgs[0] = np.array((x_extent, y_extent, z_extent)[ax_x])
    if (x_extent, y_extent, z_extent)[ax_y] is not None:
        #extent_cgs[1] = np.array((x_extent, y_extent, z_extent))[ax_y]
        extent_cgs[1] = np.array((x_extent, y_extent, z_extent)[ax_y])
    #
    #
    if axunit is None or axulabel is None:
        axunit, axulabel = pc, 'pc'
    extent = extent_cgs / axunit
    #
    print(f'axVarSurfdens extent: {extent[0]} | {extent[1]}')
    #
    im = ax.imshow(im_data.T, origin="lower", cmap=hist_cmap, norm=norm,
    interpolation=imshow_interpolation, extent=extent.flatten(), alpha=alpha)
    ax.set(xlim=extent[0], ylim=extent[1])
    #
    if show_cbar==True:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="4%", pad=0.1)
        cbar = plt.colorbar(im, cax=cax, label=barlabel)
    elif show_cbar==False:
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("left", size="4%", pad=0.1)
        cbar = None
        cax.set_axis_off()
    elif show_cbar==None:
        cbar = None
    else:
        cax = show_cbar
        cbar = plt.colorbar(im, cax=cax, label=barlabel)
    #
    axFormat(ax, axis, None, vartitle, axulabel, show_axlabel, show_title)
    #
    if extreturn:
        return im, cbar, extent, axunit
    else:
        return im, cbar
    

#===============================================================================
# ==== OBJECT COLUMN PLOT ======================================================
#===============================================================================

def ObjColPlot(leafdata, col_grid, col_var, selector, clcenter,
        clmins=None, foreign_selectors=None, descr=None):
    octree = col_grid.octree    
    
    #XXX# gpot = leafdata['gpot']
    #XXX# datashape = gpot.shape

    # Get bounding box with margins around object:
    center, x_extent, y_extent, z_extent = GetClBox(leafdata, octree, selector, clcenter)

    # Get float grids marking the object and foreign objects
    #XXX# clmask_grid, foreign_fmgrid = GetClGrid(leafdata, octree, selector, foreign_selectors)

    # Prepare Figure
    fig = plt.figure()
    gs = GridSpec(1, 21, top=0.98, bottom=0.02, left=0.02, right=0.98)
    ax0 = fig.add_subplot(gs[:,0:9]) # Slice (yz)
    ax1 = fig.add_subplot(gs[:,9:18], sharey=ax0 ) # Slice (xz)
    axa = [ax0, ax1]
    param = dict(x_extent=x_extent, y_extent=y_extent, z_extent=z_extent)
    
    for ia,axis in enumerate([1,0]):
        param.update(dict(axis=axis))
        ax = axa[ia]
        map_cbar = bool(ia) if show_map_cbar else None
        # Draw object coldens            
        axVarSurfdens(ax, col_grid, col_var, show_cbar=map_cbar, **param)
        # Draw projected object contour             
        #XXX# axVarContour(ax, clmask_grid, levels=[.5,], colors=['k',], **param) # XXX BAUSTELLE XXX
        #        
        mplot.ax_format_ticks(ax)
        #ax.text(0.89, 0.07, ax_label[ax_x], ha='center', va='center', transform=ax.transAxes, bbox=None)
        #if not bool(ia):
        #    ax.text(0.12, 0.94, ax_label[ax_y], ha='center', va='center', transform=ax.transAxes, bbox=None)
        
        # Remove y-axis label of but the left plot:
        if ia:
            ax.yaxis.label.set_visible(False)
            
        if clmins and show_gpot_minima:
            axMarkMinima(ax, leafdata, octree, clmins, axis=axis, **minima_format)
        
        if show_center_of_mass:
            ax_x, ax_y = ax_xy_order[axis]
            ax.scatter([center[ax_x]/pc,], [center[ax_y]/pc], **center_format)
            
    plt.setp(ax1.get_yticklabels(), visible=False)
    plt.subplots_adjust(wspace=.0)

    if descr is not None:
        ax1.text(0.00, 0.98, descr, ha='center', va='center', transform=ax1.transAxes, bbox=bbox)

    #
    return fig
    
    
#===============================================================================
# ==== MAIN ====================================================================
#===============================================================================

def Main(dirpath, filename, criterion, varkey,
        cl_range=None, subpath=None, descr_prefix=None):
    if subpath is None:
        slicevar_basename = varkey.split('::')[0]
        subpath = '%s-%s'%(criterion,slicevar_basename)
    if descr_prefix is None:
        descr_prefix = ''

    # Load datafile regarding clump/core selection
    leafdata, octree = cztools.LoadPlotfile(dirpath, filename)
    # Get clump/core selectors
    objects = clfind.get_clist(leafdata, octree, criterion)
    clselectors = list(map(clfind.select, objects))
    clcenters = [cztools.ObjCenter(leafdata, octree, ow) for ow in clselectors]

    # Create datagrid for a heightmap of slice var
    vardata = leafdata[varkey]
    #vargrid = mflash.datagrid(vardata, octree, ng=dgrid_ng, verbose=verbose-1)

    # Create a contour data grid for later contour generation
    cont_grids = {}
    for cont_var in contours:
        cont_data = leafdata[cont_var]
        cont_grids[cont_var] = mflash.datagrid(cont_data, octree, ng=dgrid_ng, verbose=verbose-1)
        
    # Compute object gpot minima
    if show_gpot_minima:
        obj_minlists = ObjMinima(leafdata, octree, 'gpot', objects,
                min_nneigh=gpot_min_nneigh)
        obj_mincoords = ObjCellCoords(octree, clcenters, obj_minlists)
        
        objects_nmins = [len(minlist) for minlist in obj_minlists]
        print('#DEBUG# Object Minima:')
        print(objects_nmins)
    
    # Loop over all clumps/cores if not ordered otherwise
    if cl_range is None: 
        cl_range = range(len(clselectors))
        
    for i in cl_range:
        selector = clselectors[i]
        clcenter = clcenters[i]
        if not fs['figtitle_enable']:
            descr = descr_prefix +'%i'%i
            
        clmins = obj_mincoords[i] if show_gpot_minima else None
        
        objectl = objects[i]
        objmask = clfind.GetMask(vardata.shape, objectl)
        objdata = vardata*objmask
        objgrid = mflash.datagrid(objdata, octree, ng=dgrid_ng, verbose=verbose-1)
        fig = ObjColPlot(leafdata, objgrid, varkey, selector,
                clcenter, clmins=clmins, foreign_selectors=clselectors, descr=descr)

        if fs['figtitle_enable'] or figtitle_force:
            title_str  = 'Data File:      %s'%(filename)
            title_str += '\nStructure ' +criterion +'[%i](%i)'%(i,len(objects))
            title_str += ' (%i minima)'%len(clmins)
            fig.suptitle(title_str)
            

        figname_scraps = [filename,i]
        if not criterion in subpath:
            figname_scraps.insert(0, criterion)
        figname = mplot.compose_name(*figname_scraps)
        
        mplot.savefig(fig, figname, subpath)
        sys.stdout.flush()

    leafdata.close()
    return None


#===============================================================================
# ==== LOBBY ===================================================================
#===============================================================================

if __name__ == '__main__':

    basedir = '/home/mweis/scratch/CF-ab/'
    #basedir = '/home/mweis/scratch/CF-Lars/'

    plotfiles_tx = [
        #'CF-B0.01a/CF97W6R_N1B0000_hdf5_plt_cnt_',
        #'CF-B0.01b/CF97W6R_N1B0000_IFChdf5_plt_cnt_',
        #'CF-B1.25a/CF97W6R_N1B0125_hdf5_plt_cnt_',
        #'CF-B1.25b/CF97W6R_N1B0125_IFChdf5_plt_cnt_',
        'CF-B2.50a/CF97W6R_N1B0250_hdf5_plt_cnt_',
        #'CF-B2.50b/CF97W6R_N1B0250_IFC_hdf5_plt_cnt_',
        'CF-B5.00a/CF97W6R_N1B0500_hdf5_plt_cnt_',
        #'CF-B5.00b/CF97W6R_N1B0500_IFC_hdf5_plt_cnt_',
    ]
    
    plotfiles_cnt = lambda cnt: [f'{pf}{cnt:04}' for pf in plotfiles_tx]

    plotfiles_t20 = [
        'CF-B0.01a/CF97W6R_N1B0000X25_hdf5_chk_0027',
        'CF-B0.01b/CF97W6R_N1B0000X_IFC_hdf5_chk_0039',
        'CF-B1.25a/CF97W6R_N1B0125X22_hdf5_chk_0025',
        'CF-B1.25b/CF97W6R_N1B0125_IFCX_hdf5_chk_0029',
        'CF-B2.50a/CF97W6R_N1B0250X30_hdf5_chk_0031',
        'CF-B2.50b/CF97W6R_N1B0250X_IFC_hdf5_chk_0041',
        'CF-B5.00a/CF97W6R_N1B0500X41_hdf5_chk_0042',
        'CF-B5.00b/CF97W6R_N1B0500_IFCX_hdf5_chk_0032',
    ]


    descr_prefixes = ['Core Aa', 'Core Ab', 'Core Ba', 'Core Bb',
        'Core Ca', 'Core Cb', 'Core Da', 'Core Db']

    plotfiles = plotfiles_t20

    pfpaths = [os.path.join(basedir,pf) for pf in plotfiles]

    for i,plotfile in enumerate(pfpaths):
       if not os.path.isfile(plotfile):
           print(f'Plotfile not found. Skipping: {plotfile}')
           continue
       dirpath, filename = os.path.split(plotfile)
       simname = os.path.basename(dirpath)
       descr_prefix = descr_prefixes[i]
       for slice_var in proj_vars:
           slicevar_basename = slice_var.split('::')[0]
           corekey_name = core_key.replace(':','_')
           subpath = f'{corekey_name}-{slicevar_basename}/{simname}'
           Main(dirpath, filename, core_key, slice_var,
                   subpath=subpath, descr_prefix=descr_prefix)
               
    print('FINISHED!')
