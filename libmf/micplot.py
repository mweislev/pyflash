# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt
import sys
import fnmatch
import os
import time
import pickle
from ast import literal_eval
import matplotlib as mpl
import matplotlib.cm as cm
import matplotlib.colors as cl
from mpl_toolkits.axes_grid1 import make_axes_locatable
from .tools import makedir

__author__ = "Michael Weis"
__version__ = '1.0.0.0'

#===============================================================================
# ==== CONSTANTS ===============================================================
#===============================================================================

homepath = os.path.expanduser("~")
wall = str(int(time.time()))

ax_xy_order = {0:[1,2], 1:[0,2], 2:[0,1]}


#===============================================================================
# ==== GENERAL FIGURE TOOLS ====================================================
#===============================================================================

# ==== SETTINGS ================================================================
pickle_fileext = '.pkl'
fig_basepath = os.path.join(homepath, 'plots')
fig_outdir = wall
fig_target = 'display'

# ==== PRESETS =================================================================

mnras_ticks = {
    'direction': 'in',
    'labelsize': 12,
    'bottom': True,
    'left': True,
    'top': True,
    'right': True,
    'labelbottom': True,
    'labelleft': True,
    'labeltop': False,
    'labelright': False,
}

mnras_ticks_small = {}
mnras_ticks_small.update(mnras_ticks)
mnras_ticks_small.update({'labelsize': 10,})

mnras_matrix_ticks = {
    'direction': 'in',
    'labelsize': 12,
    'bottom': True,
    'left': True,
    'top': True,
    'right': True,
}

fig_presets = {
    'enum': {
        'figsize': (38.40, 21.60),
        'figdpi': 150,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': True,
        'extension': '.png',
        'bbox_inches': None,
        'enum_marker': True,
        }, 
    'display': {
        'figsize': (25.60, 14.40),
        'figdpi': 150,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': None,
        'enum_marker': False,
        }, 
    'highres': {
        'figsize': (51.20, 28.80),
        'figdpi': 200,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': None,
        'enum_marker': False,
        }, 
    'thesis': {
        'figsize': (8.00, 6.00),
        'figdpi': 300,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': False,
        'counter_enable': True,
        'extension': '.pdf',
        'bbox_inches': None,
        'enum_marker': False,
        }, 
    'presentation': {
        'figsize': (12.00, 9.00),
        'figdpi': 300,
        'fontsize': 14,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': False,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis': {
        'figsize': (5.70, 4.30),
        'figdpi': 300,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis_half': {
        'figsize': (3.0, 4.30),
        'figdpi': 300,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis_fine': {
        'figsize': (8.55, 6.45),
        'figdpi': 300,
        'fontsize': 14,
        'legend_fontsize': 14,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis_half_fine': {
        'figsize': (4.5, 5.2),
        'figdpi': 300,
        'fontsize': 14,
        'legend_fontsize': 14,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis_hfl': {
        'figsize': (4.5, 5.2),
        'figdpi': 300,
        'fontsize': 14,
        'legend_fontsize': 14,
        'legend_enable': True,#False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis_large_fine': {
        'figsize': (8.55, 9.00),
        'figdpi': 300,
        'fontsize': 14,
        'legend_fontsize': 14,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mthesis_large_fine2': {
        'figsize': (8.55, 9.00),
        'figdpi': 300,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'mnras_page': {
        'figsize': (14.00, 10.50),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': None,
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_page_legend': {
        'figsize': (14.00, 10.50),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': None,
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_col': {
        'figsize': (6.60, 6.00),
        'figdpi': 600,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_col_legend': {
        'figsize': (6.60, 6.00),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 10,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_col_slegend': {
        'figsize': (6.60, 6.00),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 8,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_2col_legend': {
        'figsize': (13.20, 6.00),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 10,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_page': {
        'figsize': (2.*504./72.27, .72*2.*682./72.27),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'mnras_col_narrow': {
        'figsize': (6.60, 4.40),
        'figdpi': 700,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks_small,
        }, 
    'mnras_col_narrow_lowres': {
        'figsize': (6.60, 4.40),
        'figdpi': 100,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks_small,
        }, 
    'mnras_col_narrow_jpg': {
        'figsize': (6.60, 4.40),
        'figdpi': 100,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.jpg',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks_small,
        }, 
    'mnras_matrix': {
        'figsize': (.75*2.*504./72.27, .75*.72*2.*682./72.27),
        'figdpi': 600,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_matrix_ticks,
        }, 

    'screen_matrix': {
        'figsize': (19.2, 10.8),
        'figdpi': 200,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_matrix_ticks,
        }, 
    'present_col': {
        'figsize': (8.00, 6.00),
        'figdpi': 600,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'impress': {
        'figsize': (18.00, 11.00),
        'figdpi': 200,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'impress_14': {
        'figsize': (14.00, 11.00),
        'figdpi': 200,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'impress_highres': {
        'figsize': (18.00, 11.00),
        'figdpi': 200,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'impress_col': {
        'figsize': (10.00, 11.00),
        'figdpi': 200,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        }, 
    'screen_matrix_hires': {
        'figsize': (38.4, 21.6),
        'figdpi': 400,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_matrix_ticks,
        }, 
    'screen_matrix_uhires': {
        'figsize': (76.8, 43.2),
        'figdpi': 200,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': True,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_matrix_ticks,
        }, 
    'mnras_2col_legend_slim': {
        #'figsize': (13.20, 6.00),
        'figsize': (13.20, 4.60),
        'figdpi': 600,
        #'fontsize': 12,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        },
    'mnras_2col_compact': {
        #'figsize': (13.20, 6.00),
        'figsize': (14.00, 6.40),
        'figdpi': 600,
        'fontsize': 8,
        'legend_fontsize': 8,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',#'.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        },
    'mnras_2col_online': {
        #'figsize': (13.20, 6.00),
        'figsize': (28.00, 12.80),
        'figdpi': 200,
        'fontsize': 12,
        'legend_fontsize': 12,
        'legend_enable': True,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',#'.pdf',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        },

    'phdthesis_fullwidth': {
        'figsize': (5.70, 4.30),
        'figdpi': 300,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    'phdthesis_halfwidth': {
        'figsize': (3.0, 4.30),
        'figdpi': 300,
        'fontsize': 10,
        'legend_fontsize': 10,
        'legend_enable': False,
        'figtitle_enable': False,
        'subpath_enable': True,
        'counter_enable': False,
        'extension': '.png',
        'bbox_inches': 'tight',
        'enum_marker': False,
        'tick_params': mnras_ticks,
        }, 
    }

def get_fig_preset(target=None):
    if target is None:
        target=fig_target
    return fig_presets[target]
    
def set_fig_preset(target='mnras_col_legend'):
    global fig_target
    fig_target = target
    fs = fig_presets[fig_target]
    plt.rcParams.update({'font.size': fs['fontsize']})
    return fs

# ==== AXIS TOOLS ==============================================================
def ax_title(ax, title):
    # Load figure settings
    fs = fig_presets[fig_target]
    if isinstance(ax, list):
        #for axi in ax:
        #    plot_title(axi, title)
        return ax_title(ax[0], title)
    # Set title if figure titles are enabled:
    elif fs['figtitle_enable']:
        ax.set_title(title)
        return True
    else:
        return False

def ax_format_ticks(ax, axis='both'):
    # Load figure settings
    fs = fig_presets[fig_target]
    #
    kwargs = fs.get('tick_params', dict())
    ax.tick_params(axis=axis, which='both', **kwargs)
    return kwargs
    
def ax_legend(ax, *args, **kwargs):
    fs = fig_presets[fig_target]
    if not fs['legend_enable']:
        return None
    prop = dict(size=fs['legend_fontsize'])
    if not 'prop' in kwargs:
        kwargs['prop'] = prop
    else:
        kwargs['prop'].update(prop)
    lg = ax.legend(*args, **kwargs)
    return lg    

def axFakeLabel(ax, text, *args, **kwargs):
    ''' Add a specific label to an ax legend without plotting anything '''
    kwargs.update(dict(label=text))
    ax.plot([np.nan, np.nan], [np.nan, np.nan], *args, **kwargs)


#===============================================================================
# ==== MISC HELPER =============================================================
#===============================================================================

def get_cmap(name):
    try:
        cmap = plt.get_cmap(name)
    except ValueError:
        print(f'Warning: Could not load colormap {name}. Using viridis.')
        cmap = plt.get_cmap('viridis')
    return cmap

#===============================================================================
# ==== SAVE FIGURE =============================================================
#===============================================================================

def compose_name(*scraps):
    figname = ''
    for part in scraps:
        if part is not None:
            if figname:
                figname += '-'
            figname += str(part)
    return figname

def slugify(value):
    import re
    """
    Normalizes string, converts to lowercase, removes non-alpha characters,
    and converts spaces to hyphens.
    """
    #value = normalize('NFKD', value).encode('ascii', 'ignore')
    value = re.sub('[^\w\s-]', '', value).strip()
    value = re.sub('[-\s]+', '-', value)
    return value
    
def get_savefig_target(name, subpath=''):
    fs = fig_presets[fig_target]
    figpath = os.path.join(fig_basepath, fig_outdir)
    # Get directory to save file to
    if fs['subpath_enable']:
        filedir = os.path.join(figpath, subpath)
    else:
        filedir = figpath
    # Add counter prefix to filename
    if fs['counter_enable']:
        if not subpath in savefig.counter:
            savefig.counter[subpath] = 0
        savefig.counter[subpath] += 1
        cntstr = '%i_-_'%savefig.counter[subpath]
    else:
        cntstr = ''
    # Assemble filename
    if fs['subpath_enable']:
        filename = cntstr +name
    else:
        subpathname = slugify(subpath)
        filename = subpathname +'_-_' +cntstr +name           
    #
    return filedir, filename, fs['extension']
    
def get_savefig_filepath(name, subpath='', sanitize=False):
    # Get unused filepath in guaranteed directory
    filedir, filename, fileext = get_savefig_target(name, subpath=subpath)
    # Sanitize
    if sanitize:
        filepath = sanitize_save_target(filedir, filename, fileext)
    else:
        filepath = os.path.join(filedir, filename+fileext)
    return filepath

def sanitize_save_target(filedir, filename, fileext, autorename=True):
    # Assemble path to file
    filepath = os.path.join(filedir, filename+fileext)
    if autorename:
        # Rename if a file with the same name already exists:
        i = 1
        while os.path.exists(filepath):
            i += 1
            filename_new = filename +'_V%i'%i
            filepath = os.path.join(filedir, filename_new+fileext)
    # Make target directory if non-existent
    makedir(os.path.dirname(filepath))
    return filepath
    
def find_newest_target(filedir, filename, fileext):
    filepath_exist = None
    # Assemble path to file
    filepath = os.path.join(filedir, filename+fileext)
    # Rename while a file with the same name already exists:
    i = 1
    while os.path.exists(filepath):
        filepath_exist = filepath
        i += 1
        filename_new = filename +'_V%i'%i
        filepath = os.path.join(filedir, filename_new+fileext)
    # 
    return filepath_exist
    
def saveimg(img, name, subpath='', raw_gamma=2.2, **overrides):
    assert isinstance(raw_gamma, float)
    from imageio import imwrite
    # Get unused filepath in guaranteed directory
    filedir, filename, fileext = get_savefig_target(name, subpath=subpath)
    fileext = '.png'
    filepath = sanitize_save_target(filedir, filename, fileext)
    # Write image raw data
    data = img.get_array()
    try:
        cmap = img.get_cmap()
        norm = img.norm
        imwrite(filepath, cmap(norm(data)), **overrides)
    except:
        print(f'#D# raw_gamma: {raw_gamma}')
        data_raw = data**(1./raw_gamma)
        imwrite(filepath, data_raw, **overrides)
    return filepath
    

def savefig(fig, name, subpath='', autoclose=True, autopickle=False, autorename=True, **overrides):
    # Load figure settings
    fs = fig_presets[fig_target]
    # Get unused filepath in guaranteed directory
    filedir, filename, fileext = get_savefig_target(name, subpath=subpath)
    filepath = sanitize_save_target(filedir, filename, fileext, autorename=autorename)
    # Set figure parameters
    # figsize is ignored by savefig, but set as a parameter anyways,
    # because the user can then use the override to control the figure size.
    savefig_param = dict(dpi=fs['figdpi'], bbox_inches=fs['bbox_inches'],
        figsize=fs['figsize'])
    savefig_param.update(overrides)
    # Pop the figure size from the dictionary, which is always present,
    # as per above construction.
    # This is necessary because savefig now throws an error if given a figsize parameter
    fig_width, fig_height = savefig_param.pop('figsize')
    # Set figure size, like specified in the parameter dict.
    fig.set_size_inches(fig_width, fig_height, True)
    #print fig_width, fig_height
    #print savefig_param
    # Write figure
    print(f'-> Exporting {filepath}')
    fig.savefig(filepath, **savefig_param)
    # Store figure as pickle for later usage (e.g. combining figures)
    if autopickle:
        storefig(fig, name, subpath=subpath, autoclose=False)
    # Autoclose figure
    if autoclose:
        plt.close(fig)
    return filepath

savefig.counter = dict()

def storefig(fig, name, subpath='', autoclose=False):
    # Get unused filepath in guaranteed directory
    filedir, filename, fig_fileext = get_savefig_target(name, subpath=subpath)
    filepath = sanitize_save_target(filedir, filename, pickle_fileext)
    # Pickle figure
    print(f'-> Storing {filepath}')
    with open(filepath, 'w') as f:
        pickle.dump(fig, f)
    # Autoclose figure
    if autoclose:
        plt.close(fig)
    #
    return filepath
    
def restorefig(name, subpath=''):
    filedir, filename, fig_fileext = get_savefig_target(name, subpath=subpath)
    filepath = find_newest_target(filedir, filename, pickle_fileext)
    filepath_expected = os.path.join(filedir, filename+pickle_fileext)
    if filepath is None:
        raise ValueError('Not Found: Pickled Figure %s'%filepath_expected)
    else:
        fig = pickle.load(filepath)
        #TODO: ADD CHECK: IS RETRIEVED OBJECT ACTUALLY A FIGURE?
    return fig


#===============================================================================
# ==== TICK POSITIONING HELPER =================================================
#===============================================================================

def is_linear(norm):
    v0, v1 = norm.vmin, norm.vmax
    lnorm = cl.Normalize(v0,v1)
    ref = np.random.rand(5)
    tst = lnorm.inverse(norm(ref))
    return np.all(np.isclose(ref,tst))

def filter_ticks(tryticks, norm, nmin):
    """
    get_ticks subfunction:
    Filter ticks to display only those having a sufficient distance.
    """
    vmin, vmax = norm.inverse(0), norm.inverse(1)
    a = np.array(tryticks)
    rangeticks = np.sort(a[(a>=vmin)*(a<=vmax)])
    relpos = norm(rangeticks)
    delta = relpos[1:]-relpos[:-1]
    deltamin = 1./(2.*(nmin-1))
    if len(rangeticks) == 0:
        return []
    ticks = [rangeticks[0], ]
    dsum = 0.
    for i, d in enumerate(delta):
        dsum += d
        if dsum>deltamin:
            ticks.append(rangeticks[i+1])
            dsum = 0.
    return list(set(ticks))

def get_iregular_ticks(norm, nmin=4):
    vmin, vmax = norm.vmin, norm.vmax
    assert vmax > vmin
    if vmin>0:
        floororder = np.floor(np.log10(vmin))
        orderticks = 10**np.arange(floororder, np.ceil(np.log10(vmax)))
    else:
        orderticks = np.array([])
    ticks = filter_ticks(orderticks, norm, nmin)
    if len(ticks)<nmin: # stage 1: Add boundary ticks
        tryticks = [vmin,] + list(orderticks) + [vmax,]
        ticks = filter_ticks(tryticks, norm, nmin)
    if len(ticks)<nmin: # stage 2: Add ticks at 20%/50% of span
        tryticks += list(2.*orderticks)
        tryticks += list(5.*orderticks)
        ticks = filter_ticks(tryticks, norm, nmin)
    if len(ticks)<nmin: # stage 3: Add 2nd digit ticks
        order = np.int64(np.rint(np.log10(vmax-vmin)))
        tryticks += list(np.round(norm.inverse(np.linspace(0, 1, nmin+1)), -order+3))
        ticks = filter_ticks(tryticks, norm, nmin)
    tickpos = cl.Normalize(vmin, vmax).inverse(norm(ticks))
    ticklabel = ['%.4g'%tick for tick in ticks]
    return ticklabel, tickpos

def get_lin_ticks(norm, nmin):
    nmax = 3*nmin
    vmin, vmax = norm.vmin, norm.vmax
    assert vmax > vmin
    step = 1.
    ticks = np.arange(np.ceil(vmin),vmax,step)
    while len(ticks) < nmin:
        step /= 2.
        ticks = np.arange(np.ceil(vmin),vmax,step)
    while len(ticks) > nmax:
        ticks = ticks[::2]
    tickpos = cl.Normalize(vmin, vmax).inverse(norm(ticks))
    ticklabel = [f'{tick}' for tick in ticks]
    return ticklabel, tickpos

def get_ticks(norm, nmin=4):
    if is_linear(norm):
        return get_lin_ticks(norm, nmin)
    else:
        return get_iregular_ticks(norm, nmin)

    
#===============================================================================
# ==== Linear Interpolation Helper =============================================
#===============================================================================
def lin_interpolator(X_base, Y_base):
    I = np.argsort(X_base)
    X = np.array(X_base)[I]
    Y = np.array(Y_base)[I]
    Xmin, Xmax = X[[0,-1]]
    dX = X[1:]-X[:-1]
    def interpolate(X_in):
        X_in = np.array(X_in)
        Ifloor = np.clip(np.searchsorted(X, X_in)-1, 0, len(dX)-1)
        Ifrac = (X_in-X[Ifloor])/dX[Ifloor]
        return ((1.-Ifrac).T*(Y[Ifloor]).T + (Ifrac).T*(Y[Ifloor+1]).T).T
    return interpolate
    

#===============================================================================
# ==== 2D-Histogram-Helper =====================================================
#===============================================================================

def bincount2d(Ix, Iy, weights, minlenx=None, minleny=None):
    lenx = max(np.max(Ix)+1, minlenx)
    leny = max(np.max(Iy)+1, minleny)
    Iflat = Ix*leny + Iy
    n = len(Ix)//100000+1
    countsflat = np.bincount(Iflat[0::n], weights=weights[0::n], minlength=lenx*leny)
    for i in range(1,n):
        countsflat += np.bincount(Iflat[i::n], weights=weights[i::n], minlength=lenx*leny)
    return countsflat.reshape(lenx, leny)

class digiscatter(object):
    def __init__(self, X, Y, x_norm=None, y_norm=None, x_res=400, y_res=400,
                 x_unit=1., y_unit=1.):
        # Select a x-norm if not given 
        x_data = X/x_unit
        if x_norm is None:
            x_min, x_max = np.percentile(x_data, 3), np.percentile(x_data, 97)
            self.x_norm = cl.LogNorm(x_min, x_max) if (x_min*x_max>0.) else cl.Normalize(x_min, x_max)
        else:
            self.x_norm = x_norm
        # Select a y-norm if not given 
        y_data = Y/y_unit
        if y_norm is None:
            y_min, y_max = np.percentile(y_data, 3), np.percentile(y_data, 97)
            self.y_norm = cl.LogNorm(y_min, y_max) if (y_min*y_max>0.) else cl.Normalize(x_min, x_max)
        else:
            self.y_norm = y_norm
        #
        self.x_res=x_res
        self.y_res=y_res
        #
        self.x_partition = x_norm.inverse(np.linspace(0, 1, x_res+1))
        self.y_partition = y_norm.inverse(np.linspace(0, 1, y_res+1))
        #
        self.x_bins = np.searchsorted(self.x_partition, x_data.flatten())
        self.y_bins = np.searchsorted(self.y_partition, y_data.flatten())
        #
        self.extent = [self.x_norm.inverse(0), self.x_norm.inverse(1),
                       self.y_norm.inverse(0), self.y_norm.inverse(1)]
        
    def heatmap(self, weights=None):
        lenx = self.x_res+2
        leny = self.y_res+2
        xy_histo = bincount2d(self.x_bins, self.y_bins, weights, lenx, leny)
        return xy_histo[1:-1,1:-1]
        
    def ax_scatter(self, ax, weight_data=None, cbar=True, **kwargs):
        if weight_data is None:
            weight_data = np.ones_like(self.x_bins, dtype=np.float64)
        weights = weight_data/weight_data.sum()*self.x_res*self.y_res
        xy_data = self.heatmap(weights)
        #
        vmax = self.x_res*self.y_res*.5
        vmin = vmax*1e-7
        np.clip(xy_data, vmin, vmax, xy_data)        
        im = ax.imshow(xy_data.T, origin="lower", interpolation='none',
            norm=cl.LogNorm(vmin=vmin, vmax=vmax),
            extent=self.extent, aspect='auto', **kwargs)
        ax.set(xlim=self.extent[0:2], ylim=self.extent[2:4])
        #
        xticks, xtickpos = get_ticks(self.x_norm)
        ax.set_xticks(xtickpos)
        ax.set_xticklabels(xticks)
        yticks, ytickpos = get_ticks(self.y_norm)
        ax.set_yticks(ytickpos)
        ax.set_yticklabels(yticks)
        #
        if cbar:
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="4%", pad=0.1)
            cbar = plt.colorbar(im, cax=cax)
            return im, cbar
        else:
            return im
            
    def ax_colscatter(self, ax, C, c_norm=None, c_unit=1., Cw_data=None, Yw_data=None, **kwargs):
        # Select a x-norm if not given 
        c_data = C/c_unit
        if c_norm is None:
            c_min, c_max = np.percentile(c_data, 3), np.percentile(c_data, 97)
            self.x_norm = cl.LogNorm(c_min, c_max) if (c_min*c_max>0.) else cl.Normalize(c_min, c_max)
        else:
            self.c_norm = c_norm
        #
        vmax = self.x_res*self.y_res*.5
        vmin = vmax*1e-7
        #
        if Cw_data is None:
            cw_data = np.ones_like(self.x_bins, dtype=np.float64)
        c_weights = cw_data/cw_data.sum()*self.x_res*self.y_res
        c_weight_histo = self.heatmap(c_weights)
        loads =  c_data.flatten() * c_weights
        load_histo = self.heatmap(loads)
        xyC_data = load_histo/(c_weight_histo+1e-20*vmax)
        #
        if Yw_data is None:
            Yw_data = np.ones_like(self.x_bins, dtype=np.float64)
        weights = Yw_data/Yw_data.sum()*self.x_res*self.y_res
        xyY_data = self.heatmap(weights)
        #
        Y_norm = cl.LogNorm(vmin, vmax, clip=True)
        Y = Y_norm(np.clip(xyY_data, vmin, vmax))
        C = c_norm(np.clip(xyC_data, c_norm.inverse(0), c_norm.inverse(1)))
        RGB_map = RGB(1.-Y, -1.8+1.2*np.pi*C)
        #
        im = ax.imshow(RGB_map, origin="lower", interpolation='none',
            extent=self.extent, aspect='auto')
        ax.set(xlim=self.extent[0:2], ylim=self.extent[2:4])
        #
        divider = make_axes_locatable(ax)
        c1ax = divider.append_axes("right", size="4%", pad=0.1)
        cb1 = mpl.colorbar.ColorbarBase(c1ax, cmap='Y_map', norm=Y_norm,
            orientation='vertical')
        #
        c2ax = divider.append_axes("bottom", size="4%", pad=0.6)
        cb2 = mpl.colorbar.ColorbarBase(c2ax, cmap='C_map', norm=c_norm,
            orientation='horizontal')
        #
        xticks, xtickpos = get_ticks(self.x_norm)
        ax.set_xticks(xtickpos)
        ax.set_xticklabels(xticks)
        yticks, ytickpos = get_ticks(self.y_norm)
        ax.set_yticks(ytickpos)
        ax.set_yticklabels(yticks)
        #
        return im, (cb1, cb2)
    

#===============================================================================
# ==== Function Fitting ========================================================
#===============================================================================

from scipy.optimize import leastsq

def fit_lrelsq(p0, datax, datay, func, erry=None, verbose=False):
    if erry is None:
        yweights = np.ones_like(datay)
    else:
        yweights = 1./erry**2
        yweights /= np.sum(yweights)
        yweights *= len(yweights)
        if not np.all(np.isfinite(yweights)):
            print('!E [fit_lrelsq]: Non-finite weights!')
            print(f'erry{erry.shape}:')
            print(erry)
            raise ValueError('[fit_lrelsq]: Non-finite weights!')
        
    def errfunc(p, x, y):
        residual = (y-func(x,p))
        assert np.all(np.isfinite(residual))
        badness = yweights * np.fabs(residual)
        return badness
    
    pfit, pcov, infodict, errmsg, success = \
        leastsq( errfunc, p0, args=(datax, datay), full_output=1)
    if verbose:
        print('## FS:', success)
        print('## PCOV:', pcov)
    if (len(datay) > len(p0)) and pcov is not None:
        s_sq = (errfunc(pfit, datax, datay)**2).sum()/(len(datay)-len(p0))
        pcov = pcov * s_sq # Multiply hessian with variance of residuals -> covariance matrix
    else:
        pcov = np.inf
    error = [] 
    for i, _ in enumerate(pfit):
        try:
          error.append( np.absolute(pcov[i][i])**0.5)
        except:
          error.append( 0.00 )            
    return pfit, np.array(error)
    
def polynomial(x, p):
    ''' Polynomial of degree n=len(p)-1  '''
    result = 0.
    for i, ai in enumerate(p):
        result += ai*x**i
    return result

def LinFit(X, Y, EY=None):
    finite = np.isfinite(X)*np.isfinite(Y)
    n_samples = np.sum(finite)
    x_fit, y_fit = X[finite], Y[finite]
    ey_fit = None if EY is None else EY[finite] 
    if n_samples > 2:
        slope0 = (y_fit[-1]-y_fit[0])/(x_fit[-1]-x_fit[0])
        offset0 = y_fit[0] -slope0*x_fit[0]
        pstart = [offset0,slope0]
        pfit, perr = fit_lrelsq(pstart, x_fit, y_fit, polynomial, erry=ey_fit)
    else:
        pfit = [np.nan, np.nan]
        perr = np.array([np.inf, np.inf])
    return pfit, perr

def powerlaw(x, p):
    log_y =  polynomial(np.log10(x), p)
    return 10**log_y

def PowerFit(X, Y, EY=None):
    ElY = EY/Y/np.log(10.) if EY is not None else None
    return LinFit(np.log10(X), np.log10(Y), ElY)

def WeightedAvg(values, errors):
    weights = 1./errors**2
    avg = np.average(values, weights=weights)
    deltas = values-avg
    n = len(deltas)
    err_extern = np.sqrt(1./(n-1.)*np.sum(weights*deltas**2)/np.sum(weights))
    err_intern = 1./np.sqrt(np.sum(weights))
    err = max(err_extern,err_intern)
    return avg, err    


#===============================================================================
# ==== temperature color calculator ============================================
#===============================================================================

def _PlanckE(wl, temp):
    # Spectral energy density in terms of wavelength
    h = 6.626070e-34
    c = 3e+8
    kb = 1.380649e-23
    u_wl = 8.*np.pi*h*c/wl**5 / (np.exp(h*c/(wl*kb*temp))-1.)
    return np.log10(u_wl)

def _ConeFundamentals():
    # For tables see http://cvrl.ucl.ac.uk/cones.htm
    # We use the 2-deg fundamentals based on the Stiles & Burch 10-deg CMFs
    conefile = './libmf/ss2_10e_1.csv'
    fields = ['wavelength', 'logL', 'logM', 'logS']
    data = np.genfromtxt(conefile, names=fields, delimiter=',')
    return data

cfd = _ConeFundamentals()
def TempColor(temp, gamma=1.0):
    ta = np.array(temp)[...,None]
    wl = cfd['wavelength'] * 1e-9
    lL, lM, lS = cfd['logL'], cfd['logM'], cfd['logS']
    lPl = _PlanckE(wl, ta)
    L_response = np.sum(10**(lL+lPl), axis=-1)
    M_response = np.sum(10**(lM+lPl), axis=-1)
    S_response = np.sum(10**(lS+lPl), axis=-1)
    RGB = (np.array([L_response, M_response, S_response])**gamma).T
    RGB /= np.maximum(1e-10,np.nanmax(RGB, axis=-1))[...,None]
    return RGB
    

#===============================================================================
# ==== color value manipulation helper =========================================
#===============================================================================
def rescale(X, Ymin=0.05, Ymax=0.95, Xmin=0., Xmax=1.):
    np.clip(X, Xmin, Xmax, X)
    Xn = (X -Xmin) / (Xmax -Xmin)
    Y = Ymin +(Ymax-Ymin)*Xn
    return Y


# ==== color model base vectors ===============================================
# Luminance model: Percieved luminance Y:
# Y = KR*R +KG*G +KB*B and KR +KG +KB = 1
# K = KR, KG, KB
K_color = dict()
K_color['NTSC'] = np.array((.3, .59, .11))
K_color['PAL'] = np.array((.299, .587, .114))
K_color['HDTV'] = np.array((.2126, .7152, .0722))
K_color['RGB'] = np.array((1./3., 1./3., 1./3.))
K_std_model = 'HDTV'


# ==== YPbPr color space converter =============================================
def RGB_to_YPbPr(R, G, B, model=K_std_model):
    # Read coefficients from color model
    KR, foo, KB = K_color[model]
    KG = 1. -KR -KB
    # Convert
    Y = KR*R +KG*G +KB*B
    Pr = .5*(R-Y)/(1.-KR)
    Pb = .5*(B-Y)/(1.-KB)
    return Y, Pb, Pr

def YPbPr_to_RGB(Y, Pb, Pr, model=K_std_model):
    # Read coefficients from color model
    KR, foo, KB = K_color[model]
    KG = 1. -KR -KB
    # Convert
    R = Y +2.*Pr*(1.-KR)
    B = Y +2.*Pb*(1.-KB)
    G = (Y -KR*R -KB*B) / KG
    return R, G, B


# ==== HSY color space converter ===============================================
def HCY_to_RGB(H, C, Y, model=K_std_model):
    # Read coefficients from color model
    KR, foo, KB = K_color[model]
    KG = 1. -KR -KB
    # Reconstruct Hue information
    a = C * np.cos(H)
    b = C * np.sin(H)
    #
    x = np.sqrt(1./3.)
    R = Y +(KG+KB)*a +x*(KB-KG)*b
    G = Y -KR*a      +x*(KR+2.*KB)*b
    B = Y -KR*a      -x*(KR+2.*KG)*b
    return R, G, B

def HY_to_Cmax(H, Y, model):
    RGBx = np.array(HCY_to_RGB(H, 1., Y, model=model))
    dRGBx = RGBx -Y
    dx_neg = np.min(dRGBx, axis=0)
    dx_pos = np.max(dRGBx, axis=0)
    with np.errstate(divide='ignore', invalid='ignore'):
        Cmax_neg =   (-Y) / dx_neg
        Cmax_pos = (1.-Y) / dx_pos 
        Cmax = np.fmin(Cmax_neg, Cmax_pos)
    return Cmax

def HSY_to_RGB(H, S, Y, model=K_std_model):
    C = S * HY_to_Cmax(H, Y, model)
    R, G, B = HCY_to_RGB(H, C, Y, model=model)
    return R, G, B

def RGB_to_HSY(R, G, B, model=K_std_model):
    # Read coefficients from color model
    KR, foo, KB = K_color[model]
    KG = 1. -KR -KB
    # Calculate Luma
    Y = KR*R +KG*G +KB*B
    # Calculate Hue
    a = R -.5*G -.5*B
    b = np.sqrt(.75)*(G-B)
    H = np.arctan2(b, a)
    # Calculate Saturation
    C = np.sqrt(a**2 +b**2)
    with np.errstate(divide='ignore', invalid='ignore'):
        S = np.nan_to_num(C / HY_to_Cmax(H, Y, model))
    return H, S, Y


# ==== COLORMAP HELPER =========================================================
def register_cmap(**kwargs):
    name = kwargs.pop('name')
    segmentdata = kwargs.pop('data')
    lsc = cl.LinearSegmentedColormap(name, segmentdata)
    #plt.register_cmap(cmap=lsc, **kwargs)
    mpl.colormaps.register(cmap=lsc, **kwargs)
    # TODO: WHAT NOW?

# ==== COLORMAP FADER ==========================================================
def transform_cmap(cmap, H=None, S=None, Y=None, name=None):
    X = np.linspace(0., 1., 1025)
    R, G, B, A = cmap(X).T
    H_X, S_X, Y_X = RGB_to_HSY(R, G, B)

    if H is None:
        H = H_X
    if S is None:
        S = S_X
    if Y is None:
        Y = Y_X
    else:
        Y_diff = Y_X -Y
        S = np.clip(.5*S-Y_diff, 0., 1.)
    

    Rx, Gx, Bx = HSY_to_RGB(H, S, Y)
    cdict = {'red':  [(x,Rx[i],Rx[i]) for i,x in enumerate(X)],
            'green': [(x,Gx[i],Gx[i]) for i,x in enumerate(X)],
            'blue':  [(x,Bx[i],Bx[i]) for i,x in enumerate(X)],}

    cmap_name = str(hash(str(cdict))) if name is None else name
    register_cmap(name=cmap_name, data=cdict)
    return get_cmap(cmap_name)


def build_faded_cmap(cmap_str, sat=None, Ymin=.01, Ymax=.85):
    cmap = get_cmap(cmap_str)

    X = np.linspace(0., 1., 1025)
    R, G, B, A = cmap(X).T
    H, S, Y = RGB_to_HSY(R, G, B)

    S_new = S if sat is None else sat
    Y_new = rescale(X, Ymin=Ymin, Ymax=Ymax)
    Rx, Gx, Bx = HSY_to_RGB(H, S_new, Y_new)

    cdict = {'red':  [(x,Rx[i],Rx[i]) for i,x in enumerate(X)],
            'green': [(x,Gx[i],Gx[i]) for i,x in enumerate(X)],
            'blue':  [(x,Bx[i],Bx[i]) for i,x in enumerate(X)],}
    register_cmap(name='f_'+cmap_str, data=cdict)

    Y_new_r = rescale(1.-X, Ymin=Ymin, Ymax=Ymax)
    Rxr, Gxr, Bxr = HSY_to_RGB(H, S_new, Y_new_r)
    cdict = {'red':  [(x,Rxr[i],Rxr[i]) for i,x in enumerate(X)],
            'green': [(x,Gxr[i],Gxr[i]) for i,x in enumerate(X)],
            'blue':  [(x,Bxr[i],Bxr[i]) for i,x in enumerate(X)],}
    register_cmap(name='fr_'+cmap_str, data=cdict)


# ==== HELIX-RGB-HELPER ========================================================
cR, cG, cB = K_color[K_std_model]
# Define corresponding orthonormal (R,G,B) vectors describing a plane of
# constant luminance, i.e. Y(R,G,B) = 0:
v1  = np.array((-cR, -cG, (cR**2+cG**2)*1./cB))
v1 /= np.linalg.norm(v1)
v2  = np.array((cG, -cR, 0.))
v2 /= np.linalg.norm(v2)
Rv = np.vstack((v1,v2))
def RGB(Y, phi, hue=1., gamma=1.):
    n = np.array((np.cos(phi), np.sin(phi)))
    amp = hue*(Y**gamma)*(1.-Y**gamma)
    RGB = Y**gamma +amp*np.dot(n.T, Rv).T
    return RGB.T

X = np.linspace(0., 1., 1025)

Y_RGB = lambda Y: RGB(1.-Y, .5+0.*Y, hue=0.)
R, G, B = Y_RGB(X).T
cdict = {'red':  [(x,R[i],R[i]) for i,x in enumerate(X)],
        'green': [(x,G[i],G[i]) for i,x in enumerate(X)],
        'blue':  [(x,B[i],B[i]) for i,x in enumerate(X)],}
register_cmap(name='Y_map', data=cdict)

C_RGB = lambda C: RGB(.5+0.*C, -1.8+1.2*np.pi*C)
R, G, B = C_RGB(X).T
cdict = {'red':  [(x,R[i],R[i]) for i,x in enumerate(X)],
        'green': [(x,G[i],G[i]) for i,x in enumerate(X)],
        'blue':  [(x,B[i],B[i]) for i,x in enumerate(X)],}
register_cmap(name='C_map', data=cdict)


# ==== Function: helix colormap constructor ====================================
def build_helixmap(name, start=.5, rotations=-1.5, sat=1.0, gamma=1.0,
                   Ymin=0., Ymax=1., model=K_std_model):

    cR, cG, cB = K_color[model]
    
    # Define corresponding orthonormal (R,G,B) vectors describing a plane of
    # constant luminance, i.e. Y(R,G,B) = 0:
    v1  = np.array((-cR, -cG, (cR**2+cG**2)*1./cB))
    v1 /= np.linalg.norm(v1)
    v2  = np.array((cG, -cR, 0.))
    v2 /= np.linalg.norm(v2)
    Rv = np.vstack((v1,v2))
    
    def helixmap(x):
        # Calculate Luma
        Y = rescale(x**gamma, Ymin=Ymin, Ymax=Ymax)
        # Calculate Hue vector
        H = 2.*np.pi*(start*1./3.+rotations*x)
        n = np.array((np.cos(H), np.sin(H)))
        Hn = np.dot(n.T, Rv).T
        # Calculate RGB
        C = sat*Y*(1.-Y)
        RGB = Y +C*Hn
        return RGB
        
    X = np.linspace(0., 1., 1025)
    
    R,G,B = helixmap(X)
    cdict = {'red':  [(x,R[i],R[i]) for i,x in enumerate(X)],
            'green': [(x,G[i],G[i]) for i,x in enumerate(X)],
            'blue':  [(x,B[i],B[i]) for i,x in enumerate(X)],}
    register_cmap(name=name, data=cdict)

    R,G,B = helixmap(1.-X)
    cdict = {'red':  [(x,R[i],R[i]) for i,x in enumerate(X)],
            'green': [(x,G[i],G[i]) for i,x in enumerate(X)],
            'blue':  [(x,B[i],B[i]) for i,x in enumerate(X)],}
    register_cmap(name=name+'_r', data=cdict)
        

# ==== Function: saturated helix colormap constructor ==========================
def build_shelixmap(name, start=2.5, rotations=-1.5, sat=.75, gamma=1.0,
                    Ymin=0.05, Ymax=0.95, model=K_std_model):
        
    def helixmap(x):
        # Calculate Luma
        Y = rescale(x**gamma, Ymin=Ymin, Ymax=Ymax)
        # Calculate Hue
        H = 2.*np.pi*(start*1./3.+rotations*x)
        #
        RGB = np.array(HSY_to_RGB(H, sat, Y, model=model))
        return RGB

    X = np.linspace(0., 1., 65537)
        
    R,G,B = helixmap(X)
    cdict = {'red':  [(x,R[i],R[i]) for i,x in enumerate(X)],
            'green': [(x,G[i],G[i]) for i,x in enumerate(X)],
            'blue':  [(x,B[i],B[i]) for i,x in enumerate(X)],}
    register_cmap(name=name, data=cdict)

    R,G,B = helixmap(1.-X)
    cdict = {'red':  [(x,R[i],R[i]) for i,x in enumerate(X)],
            'green': [(x,G[i],G[i]) for i,x in enumerate(X)],
            'blue':  [(x,B[i],B[i]) for i,x in enumerate(X)],}
    register_cmap(name=name+'_r', data=cdict)
    

# ==== Function: saturated helix colormap constructor ==========================
def build_loghelixmap(name, norm, rotscale=1., rotoffset=0., **kwords):
    logspan = np.log10(norm.vmax/norm.vmin)
    kwords['rotations'] = logspan/(3.*rotscale)
    kwords['start'] = rotoffset +(np.log10(norm.vmin)/rotscale)%3.
    build_shelixmap(name, **kwords)

# ==== Build some additional colormaps =========================================
### build colormap variants with linear fade in Y'
from matplotlib._cm import datad
from matplotlib._cm_listed import cmaps as cmaps_listed
cmap_list =  sorted(list(datad.keys())+list(cmaps_listed.keys()))
for cmap_str in cmap_list:
    build_faded_cmap(cmap_str)
### helix colormaps (those are cubehelix variations)
build_helixmap('mrcubehelix', rotations=-1.)
build_helixmap('denshelix', start=.35, rotations=5./3.)
build_helixmap('prismhelix', start=.7, rotations=-5./3.)
build_helixmap('bluehelix', start=0., rotations=0.)
build_helixmap('redhelix', start=1., rotations=0.)
build_helixmap('orange_x', start=.7, rotations=.25)
build_helixmap('greenhelix', start=2., rotations=0.)
build_helixmap('glowhelix', start=.35, rotations=1.0, sat=.75)
build_helixmap('skinhelix', start=2.1, rotations=1.0, sat=1.0)
build_helixmap('jethelix', start=2., rotations=1.32)
### digihelix colormaps (similar to cubehelix-style maps, but maximized saturation)
build_shelixmap('cubeshelix')
build_shelixmap('digidens', start=2.5, rotations=+1.5/3.)
build_shelixmap('digilight', start=1., rotations=-1.5/3.)
build_shelixmap('digicube', start=1., rotations=-4./3., sat=1.)
build_shelixmap('digimale', start=2.25, rotations=5./6., sat=1.)
build_shelixmap('digifemale', start=1.25, rotations=5./6., sat=1.)
build_shelixmap('digimic', start=.5, rotations=-1., sat=1.)
build_shelixmap('digired', start=.16, rotations=-.32)
build_shelixmap('digigreen', start=1.16, rotations=-.32)
build_shelixmap('digiblue', start=2.16, rotations=-.32)
build_shelixmap('digijet', start=2., rotations=1.32)

# ==== Load parula colormap, as found in matlab ===============================
from matplotlib.colors import LinearSegmentedColormap

cm_data = [[0.2081, 0.1663, 0.5292], [0.2116238095, 0.1897809524, 0.5776761905], 
 [0.212252381, 0.2137714286, 0.6269714286], [0.2081, 0.2386, 0.6770857143], 
 [0.1959047619, 0.2644571429, 0.7279], [0.1707285714, 0.2919380952, 
  0.779247619], [0.1252714286, 0.3242428571, 0.8302714286], 
 [0.0591333333, 0.3598333333, 0.8683333333], [0.0116952381, 0.3875095238, 
  0.8819571429], [0.0059571429, 0.4086142857, 0.8828428571], 
 [0.0165142857, 0.4266, 0.8786333333], [0.032852381, 0.4430428571, 
  0.8719571429], [0.0498142857, 0.4585714286, 0.8640571429], 
 [0.0629333333, 0.4736904762, 0.8554380952], [0.0722666667, 0.4886666667, 
  0.8467], [0.0779428571, 0.5039857143, 0.8383714286], 
 [0.079347619, 0.5200238095, 0.8311809524], [0.0749428571, 0.5375428571, 
  0.8262714286], [0.0640571429, 0.5569857143, 0.8239571429], 
 [0.0487714286, 0.5772238095, 0.8228285714], [0.0343428571, 0.5965809524, 
  0.819852381], [0.0265, 0.6137, 0.8135], [0.0238904762, 0.6286619048, 
  0.8037619048], [0.0230904762, 0.6417857143, 0.7912666667], 
 [0.0227714286, 0.6534857143, 0.7767571429], [0.0266619048, 0.6641952381, 
  0.7607190476], [0.0383714286, 0.6742714286, 0.743552381], 
 [0.0589714286, 0.6837571429, 0.7253857143], 
 [0.0843, 0.6928333333, 0.7061666667], [0.1132952381, 0.7015, 0.6858571429], 
 [0.1452714286, 0.7097571429, 0.6646285714], [0.1801333333, 0.7176571429, 
  0.6424333333], [0.2178285714, 0.7250428571, 0.6192619048], 
 [0.2586428571, 0.7317142857, 0.5954285714], [0.3021714286, 0.7376047619, 
  0.5711857143], [0.3481666667, 0.7424333333, 0.5472666667], 
 [0.3952571429, 0.7459, 0.5244428571], [0.4420095238, 0.7480809524, 
  0.5033142857], [0.4871238095, 0.7490619048, 0.4839761905], 
 [0.5300285714, 0.7491142857, 0.4661142857], [0.5708571429, 0.7485190476, 
  0.4493904762], [0.609852381, 0.7473142857, 0.4336857143], 
 [0.6473, 0.7456, 0.4188], [0.6834190476, 0.7434761905, 0.4044333333], 
 [0.7184095238, 0.7411333333, 0.3904761905], 
 [0.7524857143, 0.7384, 0.3768142857], [0.7858428571, 0.7355666667, 
  0.3632714286], [0.8185047619, 0.7327333333, 0.3497904762], 
 [0.8506571429, 0.7299, 0.3360285714], [0.8824333333, 0.7274333333, 0.3217], 
 [0.9139333333, 0.7257857143, 0.3062761905], [0.9449571429, 0.7261142857, 
  0.2886428571], [0.9738952381, 0.7313952381, 0.266647619], 
 [0.9937714286, 0.7454571429, 0.240347619], [0.9990428571, 0.7653142857, 
  0.2164142857], [0.9955333333, 0.7860571429, 0.196652381], 
 [0.988, 0.8066, 0.1793666667], [0.9788571429, 0.8271428571, 0.1633142857], 
 [0.9697, 0.8481380952, 0.147452381], [0.9625857143, 0.8705142857, 0.1309], 
 [0.9588714286, 0.8949, 0.1132428571], [0.9598238095, 0.9218333333, 
  0.0948380952], [0.9661, 0.9514428571, 0.0755333333], 
 [0.9763, 0.9831, 0.0538]]
parula_map = LinearSegmentedColormap.from_list('parula', cm_data)
#plt.register_cmap(cmap=parula_map)
#mpl.colormaps.register(cmap=parula_map)
# TODO: WHAT NOW?
