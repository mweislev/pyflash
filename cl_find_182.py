#!/usr/bin/env python
# -*- coding: utf-8 -*-
import os,sys
import numpy as np
from collections import OrderedDict

import libmf.micflash as mflash
import libmf.tools as tools

import cz_tools as cz
import cl_fofg_142 as fof
import cl_fellwalk_090 as fellwalk
import cl_gaussfit as gaussfrag
import cl_dendrofind as ddg
import cz_astrodendro as adr
import cz_findfragment as czff

from py27hash.hash import hash27

__author__ = "Michael Weis"
__version__ = "1.8.1.0"


#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

#listonly = True

# ==== CONSTANTS ===============================================================
from constants import *

# ==== BASE SETTINGS ===========================================================
var_ch = mflash.var_ch5 # XXX TRIPLE-CHECK ! XXX
plotfile_pattern = '*plt_cnt_???5'#'*hdf5_*'#'*chk_*'#'*plt_cnt_*'#'*hdf5_*'#
clist_basename = 'cl_l1810_'#'cl_l870_'
skip_patterns = ['_part_', '.pkl', clist_basename, '-rd_', '-td_']

datadir = f'/home/mweis/scratch/external/clist'

# ==== CELL SELECTION CRITERIA =================================================
cell_criteria = OrderedDict()

cell_criteria['COCx']   = {'abund_co':[1e-4, np.inf], }
cell_criteria['COCex8'] = {'abund_co':[1e-4, np.inf], 'cdto':[8., np.inf],}

cell_criteria['COCxNL99']   = {'abund_co':[7e-5, np.inf], }
cell_criteria['COCex8NL99'] = {'abund_co':[7e-5, np.inf], 'cdto':[8., np.inf],}

cell_criteria['COCex'] = {'abund_co':[1e-4, np.inf], 'cdto':[7., np.inf],}
cell_criteria['COCex5'] = {'abund_co':[1e-4, np.inf], 'cdto':[5., np.inf],}
cell_criteria['COCex6'] = {'abund_co':[1e-4, np.inf], 'cdto':[6., np.inf],}
cell_criteria['COCxco']   = {'abund_co':[1.3975e-4, np.inf], }
cell_criteria['COCxp'] = {'abund_co':[1e-4, np.inf], 'p_sum':[-np.inf, 0.],}
cell_criteria['COCxphd'] = {'abund_co':[1e-4, np.inf], 'p_hdyn':[-np.inf, 0.],}
cell_criteria['CCE19x'] = {'abund_co':[1e-4, np.inf], 'dens':[1e-19, np.inf], }
cell_criteria['CC2E19x'] = {'abund_co':[1e-4, np.inf], 'dens':[2e-19, np.inf], }
cell_criteria['CC5E19x'] = {'abund_co':[1e-4, np.inf], 'dens':[5e-19, np.inf], }
cell_criteria['test1']   = {'abund_co':[1e-4, np.inf], }
cell_criteria['HIIregb']   = {'ihp':[.6, np.inf], }
cell_criteria['CCE20x'] = {'abund_co':[1e-4, np.inf], 'dens':[1e-20, np.inf], }
cell_criteria['COCxgg'] = {'abund_co':[1e-4, np.inf], 'absgrad_gpot':[5e-8, np.inf],}
cell_criteria['COCxggs2'] = {'abund_co':[1e-4, np.inf], 'aggps':[5e-8, np.inf],}
cell_criteria['COCxggs8l'] = {'abund_co':[1e-4, np.inf], 'gcritl':[1., np.inf],}
cell_criteria['COCxggs8q'] = {'abund_co':[1e-4, np.inf], 'gcritsq':[1., np.inf],}
cell_criteria['COCxggt2'] = {'abund_co':[1e-4, np.inf], 'absgrad_gpot':[5e-8, np.inf],}
cell_criteria['all'] = {'dens':[0., np.inf],}

cell_criteria['Coletta25_clump'] = {'numdens':[1e+3,np.inf], 'temp':[0.,200.]} # Max temp increased from 60 to include core cells
cell_criteria['Coletta25_core'] = {'numdens':[1e+6,np.inf], 'temp':[0.,200.]}

cell_criteria['COCxggt2'] = {'abund_co':[1e-4, np.inf], 'absgrad_gpot':[5e-8, np.inf],}

cell_criteria['Cex8'] = {'cdto':[8., np.inf],}
cell_criteria['Cex12E19'] = {'cdto':[12., np.inf],'dens':[1e-19, np.inf]}

#cell_criteria['COCex8-reg1'] = {'abund_co':[1e-4, np.inf], 'cdto':[8., np.inf], 'reg1':[.5, np.inf]}
#cell_criteria['COCex8-reg2'] = {'abund_co':[1e-4, np.inf], 'cdto':[8., np.inf], 'reg2':[.5, np.inf]}

# ==== ALGORITHM SETTINGS ======================================================
select_criteria_combos = OrderedDict()

select_criteria_combos['CCE20g_fw2']  = {'var':'gpot', 'threshold':None, 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'criteria':cell_criteria['CCE20x']}
select_criteria_combos['COCg_fw2']  = {'var':'gpot', 'threshold':None, 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'criteria':cell_criteria['COCx']}
select_criteria_combos['COCg0_fw2']  = {'var':'gpot', 'threshold':0., 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'criteria':cell_criteria['COCx']}
select_criteria_combos['COMax_fw2']  = {'var':'gpot', 'threshold':None, 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'criteria':{'abund_co':[1.375e-4, np.inf], }}
select_criteria_combos['COCg0_fw2s']  = {'var':'gpot', 'threshold':0., 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'preselect':'COCx'}
select_criteria_combos['COCg0_fw2fs']  = {'var':'gpot', 'threshold':0., 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'preselect':'COCx'}
select_criteria_combos['COCxgg_fw2fs']  = {'var':'gpot', 'threshold':None, 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'preselect':'COCxgg'}
select_criteria_combos['COCxgg_stitch'] = {'base_criterion':'COCxgg', 'outer_criterion':'COCx', }
select_criteria_combos['COCxgg2_stitch'] = {'base_criterion':'COCxgg', 'outer_criterion':'COCx', }

select_criteria_combos['COCx'] = {'criteria':cell_criteria['COCx'], 'MinPts':100}
select_criteria_combos['COCex8'] = {'criteria':cell_criteria['COCex8'], 'MinPts':100}
select_criteria_combos['COCxs'] = {'criteria':cell_criteria['COCx'], 'MinPts':5}
select_criteria_combos['COCex8s'] = {'criteria':cell_criteria['COCex8'], 'MinPts':5}
select_criteria_combos['COCxgg'] = {'criteria':cell_criteria['COCxgg'], 'MinPts':100}
select_criteria_combos['COC_iso'] = {'criteria':cell_criteria['COCx'],
    'var':'gpot', 'direction':'up', 'dperc':1.0, 'MinPts':30}

#select_criteria_combos['COCex8p30:isocut'] = {'criteria':cell_criteria['all'],
#    'var':'gpot', 'direction':'up', 'dperc':0.1}
#select_criteria_combos['COCex8p30:dendroleaf'] = {'criteria':cell_criteria['all'],
#    'var':'dens_log', 'min_value':-20., 'min_delta':1., 'min_npix':30}
select_criteria_combos['dendroleaf'] = {'var':'dens_log', 'min_value':-20., 'min_delta':1., 'min_npix':30}
select_criteria_combos['dendronode'] = {'var':'dens_log', 'min_value':-20., 'min_delta':1., 'min_npix':30}

select_criteria_combos['adrleaf'] = {'var':'dens_log', 'min_value':-20., 'min_delta':1., 'min_npix':30}
select_criteria_combos['adrnode'] = {'var':'dens_log', 'min_value':-20., 'min_delta':1., 'min_npix':30}

select_criteria_combos['adrnode_h'] = {'var':'dens_log', 'min_value':-20.5, 'min_delta':.5, 'min_npix':30}

select_criteria_combos['czfrag'] = {'hierarchical':False, 'checkpeak':False} # if needed, follow up with czfilterhc to get both filtered and unfiltered

select_criteria_combos['COCxp30'] = {'criteria':cell_criteria['COCx'], 'MinPts':30}
select_criteria_combos['COCex8p30'] = {'criteria':cell_criteria['COCex8'], 'MinPts':30}
select_criteria_combos['COCex12E19p30'] = {'criteria':cell_criteria['Cex12E19'], 'MinPts':30}

select_criteria_combos['COCxp30NL99'] = {'criteria':cell_criteria['COCxNL99'], 'MinPts':30}
select_criteria_combos['COCex8p30NL99'] = {'criteria':cell_criteria['COCex8NL99'], 'MinPts':30}

select_criteria_combos['Cg0a_fw2']  = {'var':'gpot', 'threshold':0., 'direction':'down',
    'MaxJump':4, 'MinDipStd':0., 'criteria':cell_criteria['all']}
select_criteria_combos['Cex8p30'] = {'criteria':cell_criteria['Cex8'], 'MinPts':30}

    
#select_criteria_combos['COCex8p30-reg1'] = {'criteria':cell_criteria['COCex8-reg1'], 'MinPts':30}
#select_criteria_combos['COCex8p30-reg2'] = {'criteria':cell_criteria['COCex8-reg2'], 'MinPts':30}

select_criteria_combos['dens_fw2']  = {'var':'dens', 'threshold':1e-19, 'direction':'up',
    'MaxJump':4, 'MinDipStd':0., 'criteria':cell_criteria['all']}

select_criteria_combos['AG_clumps'] = {'criteria':cell_criteria['Coletta25_clump'], 'MinPts':30}
select_criteria_combos['AG_dendrocores'] = {'var':'numdens_log', 'min_value':6., 'min_delta':1., 'min_npix':30}
select_criteria_combos['AG_gausscores_gsfrag'] = {'maxvar':'numdens', 'threshold':1.e+6, 'max_vol':.1*pc**3, 'max_aspect':20, 'max_elong':8,
                                                  'minfraction_fit':1e-5}
select_criteria_combos['AG_gravcores_gsfrag'] = {'maxvar':'gpot', 'threshold':0., 'invertdir':True, 'max_vol':.1*pc**3, 'max_aspect':20, 'max_elong':8,
                                                 'minfraction_fit':0., 'criteria':cell_criteria['Coletta25_core']}
# For Almagal-cores use AG_clumps:AG_dendroleaf

require_gridvars = ['COCxgg', 'COCxgg_stitch', 'COCxgg2_stitch', 'COCxggs2',
        'COCxggs8l', 'COCxggs8q', 'COCxggt2', 'COCxgg_fw2fs']

#===============================================================================
# ==== HELPER ==================================================================    
#===============================================================================

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

def mask_cells(blockdata, select_criteria):
    mask = np.ones(blockdata.dshape(), dtype=bool)
    for criterion in select_criteria:
        data = blockdata[criterion]
        datamin, datamax = select_criteria[criterion]
        mask *= (data == np.clip(data, datamin, datamax))
    return mask
    
def GetMask(baseshape, clset):
    cl_where = select(clset)
    mask = np.zeros(baseshape, dtype=bool)
    mask[cl_where] = True
    return mask

def MasksToClists(mlist):
    print(f'Converting masks to clists...', end=' ')
    sys.stdout.flush()
    clists = [list(zip(*np.where(m))) for m in mlist]
    print('OK')
    sys.stdout.flush()
    return clists


#===============================================================================
# ==== CRITERIA INTERFACE ======================================================
#===============================================================================

def GetCriterionLimits(criterion):
    #XXX THIS FUNCTION SHOULD NOT EXIST! REFACTOR CRITERION FORMAT!
    critdict = select_criteria_combos[criterion]
    critlim = dict()
    if '_fw' in criterion:
        if 'preselect' in critdict:
            critkey = critdict['preselect']
            critlim.update(GetCriterionLimits(critkey))
        if 'criteria' in critdict:
            critlim.update(critdict['criteria'])
    else:
        critlim.update(critdict)
    return critlim
    
    
#===============================================================================
# ==== ADDITIONAL GRIDVARS =====================================================
#===============================================================================

def load_gridvars(leafblocks, octree):
    import mf_fields as clfields
    
    octree = mflash.pm3dgrid(leafblocks)
    fields = clfields.get_fields(leafblocks, octree)
    dx_gpot, dy_gpot, dz_gpot = fields['g_gpot']
    
    absgrad_gpot = np.sqrt(dx_gpot**2 +dy_gpot**2 +dz_gpot**2)
    leafblocks.cache('absgrad_gpot', absgrad_gpot)
    
    # XXX MW 01.03.2022: Hack: Stitch hole in criterion
    dens = leafblocks['dens']
#    gmask = absgrad_gpot > 5e-8
#    if np.any(gmask):
#        densthr = np.percentile(dens[gmask], 50)
#        stitchmask = dens > densthr
#        aggps = np.copy(absgrad_gpot)
#        aggps[stitchmask] = np.min(absgrad_gpot[gmask])
#    leafblocks.cache('aggps', absgrad_gpot)

    # XXX MW 03.03.2022: Hack didn't work. Try continuous relaxation strategy
    from matplotlib.colors import LogNorm, Normalize
    gcrit_a = np.clip(Normalize(0., 5e-8)(absgrad_gpot), 0., np.inf)
    #gcrit_b = np.clip(LogNorm(1e-18, 1e-16)(dens), 0., np.inf)
    gcrit_b = np.clip(LogNorm(1e-19, 1e-16)(dens), 0., np.inf)
    gcritsq = np.sqrt(gcrit_a**2 +gcrit_b**2)
    gcritl = gcrit_a +gcrit_b
    leafblocks.cache('gcritl', gcritl)
    leafblocks.cache('gcritsq', gcritsq)
    return None


#===============================================================================
# ==== LOAD/SAVE DATA MANAGEMENT ===============================================
#===============================================================================

def split_criterion(criterion): # TODO: Use to make subcriteria stackable
    critlist = criterion.split(':') 
    headcrit = ':'.join(critlist[:-1])
    tailcrit = critlist[-1]
    return headcrit, tailcrit

def get_criterion_fingerprint(criterion):
    crit_dict = select_criteria_combos.get(criterion, dict())
    crit_repr = str(crit_dict)
    crit_hex = format(hash27(crit_repr), 'x')
    crit_fp = criterion +'_' +crit_hex
    return crit_fp
    
def get_clist_path(hdfdata, criterion, debug=False):
    crit_fp  = get_criterion_fingerprint(criterion)
    hdf_fp  = tools.get_plotfile_fingerprint(hdfdata)
    prefix = clist_basename +crit_fp +'_' +hdf_fp +'_'
    filepath = tools.get_datafile_path(hdfdata, prefix, outdir=datadir)
    if debug>0:
        print(f'CLIST PATH: {filepath}')
        sys.stdout.flush()
    return filepath
    
def determine_clist(hdfdata, octree, criterion):
    ''' Compile a cluster list according to the selected criterion '''
    print(f'Criterion: {criterion}')
    headcrits, tailcrit = split_criterion(criterion)
    clinfo = list()

    if not tailcrit:
        raise RuntimeError(f'cl_find: Empty tail criterion: {criterion}')

    select_criteria = select_criteria_combos.get(tailcrit, dict()) # TODO: Untangle criterion settings (one entry per : -stage)
    print('#D############################################/')
    print(f'headcrits, tailcrit: {headcrits},{tailcrit}')
    print('select_criteria')
    print(select_criteria)
    print('/#D#')

    if tailcrit in require_gridvars:
        load_gridvars(hdfdata, octree)

    if 'domain' == tailcrit:
        if not headcrits:
            allmask = np.ones(hdfdata.dshape(), dtype=bool)
            clist = MasksToClists([allmask,])
        else:
            raise RuntimeError(f'Unsuitable criterion: {criterion}')

    elif 'regions' == tailcrit:
        print('-> Retrieving regions')
        simdir = os.path.dirname(hdfdata.filename())
        regtable = cz.RegionTable(simdir)
        stepregs = cz.StepRegions(hdfdata, regtable)
        regmasks = [cz.RegionMask(hdfdata, octree, regdict) for regdict in stepregs]
        if not headcrits:
            clist = MasksToClists(regmasks)
        else:
            # Retrieve cluster masks from head criterion
            l1masks = get_cmasks(hdfdata, octree, headcrits)
            # Filter out all clusters that do not interset with any zoom region
            m_allregs = np.logical_or.reduce(regmasks)
            cmasks = [m for m in l1masks if np.any(m*m_allregs)]
            clist = MasksToClists(cmasks)

    elif 'sinkregs' == tailcrit:
        print('-> Retrieving sink regions')
        from cz_inspect_sinks import SinkRegs
        sd = cz.SinkData(hdfdata)
        sr = SinkRegs(hdfdata, sd)
        regmasks = [cz.RegionMask(hdfdata, octree, regdict) for regdict in sr]
        if not headcrits:
            clist = MasksToClists(regmasks)
        else:
            # Retrieve cluster masks from head criterion
            l1masks = get_cmasks(hdfdata, octree, headcrits)
            # Filter out all clusters that do not interset with any zoom region
            m_allregs = np.logical_or.reduce(regmasks)
            cmasks = [m for m in l1masks if np.any(m*m_allregs)]
            clist = MasksToClists(cmasks)

    elif '_gsfrag' in tailcrit:
        pre_mask = np.ones(hdfdata.dshape(), dtype=bool)
        if 'preselect' in select_criteria:
            pre_criterion = select_criteria['preselect']
            pre_cmasks = get_cmasks(hdfdata, octree, pre_criterion)
            pre_mask *= np.logical_or.reduce(pre_cmasks)
        if 'criteria' in select_criteria:
            criteria = select_criteria['criteria']
            critmask = mask_cells(hdfdata, criteria)
            pre_mask *= critmask
        print(f'-> Using Gaussian fragment finder on {np.sum(pre_mask)}/{pre_mask.size} cells')
        fragmasks, clinfo = gaussfrag.FitFragmentsGauss(hdfdata, octree, mask=pre_mask, **select_criteria)
        clist = MasksToClists(fragmasks)
    elif '_fw2fs' in tailcrit:
        print('-> Using Fellwalker algorithm on fused N-o-N preseletion')
        pre_criterion = select_criteria['preselect']
        pre_clist = get_clist(hdfdata, octree, pre_criterion)
        clist = fellwalk.fellwalk_clist_fused(pre_clist, hdfdata, octree, **select_criteria)        
    elif '_fw2s' in tailcrit:
        print('-> Using Fellwalker algorithm on N-o-N preseletion')
        pre_criterion = select_criteria['preselect']
        pre_clist = get_clist(hdfdata, octree, pre_criterion)
        clist = fellwalk.fellwalk_clist(pre_clist, hdfdata, octree, **select_criteria)
    elif '_fw' in tailcrit:
        print('-> Using Fellwalker algorithm')
        clist = fellwalk.fellwalk(hdfdata, octree, **select_criteria)
    elif '_adv' in tailcrit:
        print('Using 1-connected neighbors of neighbors sampling algorithm')
        clist = fof.findcluster_adv(hdfdata, octree, **select_criteria)
    elif '_stitch' in tailcrit:
        print('-> Using stitched neighbors of neighbors sampling algorithm')
        base_criterion_key = select_criteria['base_criterion']
        base_criteria = select_criteria_combos[base_criterion_key]
        outer_criterion_key = select_criteria['outer_criterion']
        outer_criteria = select_criteria_combos[outer_criterion_key]
        pre_clist = fof.findcluster_direct(hdfdata, octree, base_criteria)
        clist = fof.StitchCluster(hdfdata, octree, pre_clist, outer_criteria)
    elif '_iso' in tailcrit:
        print('-> Using isosurface maximization algorithm')
        clist = fof.findcluster_isos(hdfdata, octree, **select_criteria)
        print(f'#DEBUG# len clist: {len(clist)}')

    elif 'isocut' in tailcrit:
        print(f'-> Using isosurface cutout algorithm: Retrieve {headcrits} objects')
        l1clist = get_clist(hdfdata, octree, headcrits)
        print(f'-> Using isosurface cutout algorithm: Process {headcrits} objects')
        clist   = fof.padcluster_isos(hdfdata, octree, l1clist, **select_criteria)
    elif 'dendroleaf' in tailcrit:
        print(f'-> Using dendrogram leaf algorithm: Retrieve {headcrits} objects')
        l1clist = get_clist(hdfdata, octree, headcrits)
        print(f'-> Using dendrogram leaf algorithm: Process {headcrits} objects')
        clist   = ddg.isolate_leafs(hdfdata, octree, l1clist, **select_criteria)
    elif 'dendronode' in tailcrit:
        print(f'-> Using dendrogram node algorithm: Retrieve {headcrits} objects')
        l1clist = get_clist(hdfdata, octree, headcrits)
        print(f'-> Using dendrogram node algorithm: Process {headcrits} objects')
        clist   = ddg.get_nodes(hdfdata, octree, l1clist, **select_criteria)
    elif 'adrleaf' in tailcrit:
        print(f'-> Using AMR dendrogram leaf algorithm: Retrieve {headcrits} objects')
        l1clist = get_clist(hdfdata, octree, headcrits)
        print(f'-> Using AMR dendrogram leaf algorithm: Process {headcrits} objects')
        clist   = adr.isolate_leafs(hdfdata, octree, l1clist, **select_criteria)
    elif 'adrnode' in tailcrit:
        print(f'-> Using AMR dendrogram node algorithm: Retrieve {headcrits} objects')
        l1clist = get_clist(hdfdata, octree, headcrits)
        print(f'-> Using AMR dendrogram node algorithm: Process {headcrits} objects')
        clist   = adr.get_nodes(hdfdata, octree, l1clist, **select_criteria)
    elif 'czfrag' in tailcrit:
        print(f'-> Using czfrag filter algorithm: Retrieve {headcrits} objects')
        l1cmasks = get_cmasks(hdfdata, octree, headcrits)
        print(f'-> Using czfrag filter algorithm: Process {headcrits} objects')
        kwargs = dict(sinkdata=cz.SinkData(hdfdata))
        kwargs.update(select_criteria)
        fragmasks = czff.FilterFragments(hdfdata, octree, l1cmasks, **kwargs)
        clist = MasksToClists(fragmasks)
    elif 'czfilterhc' in tailcrit:
        print(f'-> Using hierarchical filter algorithm: Retrieve {headcrits} objects')
        l1fragmasks = get_cmasks(hdfdata, octree, headcrits)
        print(f'-> Using hierarchical filter algorithm: Process {headcrits} objects')
        fragmasks_f = czff.FilterHierarchical(l1fragmasks)
        clist = MasksToClists(fragmasks_f)
    else:
        print('-> Using neighbors of neighbors sampling algorithm')
        clist = fof.findcluster_direct(hdfdata, octree, **select_criteria)
        print(f'#DEBUG# len clist: {len(clist)}')

    #clist_compressed = [np.array(list(clset), dtype=np.int32) for clset in clist]
        
    if not clinfo: clinfo = len(clist)*[None,]
    return clist, clinfo

def prepare_clist(hdfdata, octree, criterion):
    ''' Compile and save a cluster list according to criterion if not existing yet '''
    filepath = get_clist_path(hdfdata, criterion)
    # Check if the list is already in place
    if os.path.exists(filepath) and not 'test' in criterion:
        print(f'-> Cluster list already exists: {filepath}')
        sys.stdout.flush()
    else:
        D = determine_clist(hdfdata, octree, criterion)
        try:
            tools.save_data(D, filepath, force=False)
        except KeyError:
            raise
        except:
            print('!! Warning: Could not save cluster data !!')
    return filepath

def get_clist(hdfdata, octree, criterion, getinfo=False, save=True):
    ''' Obtain a cluster list according to criterion; compile if necessary '''
    filepath = get_clist_path(hdfdata, criterion)
    #
    force = False    
    if os.path.exists(filepath):
        print(f'-> Loading cluster list ({filepath}:{criterion})')
        sys.stdout.flush()
        try:
            D = tools.load_data(filepath)
        except:
            print('!! Warning: Could not load cluster list !!')
            force = True
        else:
            if getinfo:
                return D
            else:
                clist, clinfo = D
                return clist
    D = determine_clist(hdfdata, octree, criterion)
    if save:
        print(f'-> Saving cluster list ({criterion})')
        sys.stdout.flush()
        try:
            tools.save_data(D, filepath, force=force)
        except KeyError:
            raise
        except:
            print('!! Warning: Could not save cluster data !!')
    if getinfo:
        return D
    else:
        clist, clinfo = D
        return clist

def get_cmasks(hdfdata, octree, criterion, **kwargs):
    ''' 
    Obtain a cluster list in leaf mask format according to criterion; compile if necessary.
    If in doubt, this is probably what you can use to interface with other code parts.
    '''
    clist = get_clist(hdfdata, octree, criterion, **kwargs)
    baseshape = hdfdata.dshape()
    cmasks = [GetMask(baseshape,clset) for clset in clist]
    return cmasks


#===============================================================================
# ==== DATA ACQUISITION LOOP ===================================================
#===============================================================================

def make_clists(dirpath, dirfiles, criterion):
    for filename in dirfiles:
        plotfile = os.path.join(dirpath, filename)
        for pattern in skip_patterns:
            if pattern in filename:
                print(f'WW Skipping {plotfile} ')
                break
        else:
            print(f'>> Processing {plotfile}')
            sys.stdout.flush()
            try:
                # Open plotfile data
                hdfdata = mflash.plotfile(plotfile, memorize=False)
                hdfdata.learn(mflash.var_mhd)
                hdfdata.learn(mflash.var_grid)
                hdfdata.learn(var_ch)
            except (AttributeError,FileNotFoundError,IOError) as e:
                print(f'EE PLOTFILE NOT READABLE: {plotfile}')
                continue
            # Get octree from plotfile
            octree = mflash.pm3dgrid(hdfdata)
            # Prepare cluster list if not yet done
            prepare_clist(hdfdata, octree, criterion)
            # Close plotfile
            hdfdata.close()
    return 0

def list_resultfiles(dirpath, dirfiles, criterion):
    pass


#===============================================================================
# ==== MAIN ====================================================================
#===============================================================================

def main(simpath, criterion, stride, offset):
    dirdict = tools.scanpath(simpath, plotfile_pattern)
    for dirpath in dirdict:
        print(f'-> Walking {dirpath}')
        sys.stdout.flush()
        dirfiles = dirdict[dirpath]
        make_clists(dirpath, dirfiles[offset::stride], criterion)
    return 0


#===============================================================================
# ==== LOBBY ===================================================================
#===============================================================================
        
if __name__ == '__main__':
    criterion = sys.argv[1]
    try:
        stride = int(sys.argv[2])
        offset = int(sys.argv[3])
        dirs = sys.argv[4:]
    except ValueError:
        stride = 1
        offset = 0
        dirs = sys.argv[2:]
    for simpath in dirs:
        main(simpath, criterion, stride, offset)
    print('FINISHED!')
    
