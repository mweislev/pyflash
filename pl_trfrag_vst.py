import os, sys
import numpy as np

import matplotlib.pyplot as plt
import libmf.micplot as mplot

import cz_tracermatch as cztrmatch
import cz_trfragstat as czfragstat
import cz_trfrag_catalog as czfragcat
from cz_trfrag_catalog import GraphNode
from cz_tracerfrag import EdgeType, NodeType
from libmf.tools import fprint

from functools import reduce
from matplotlib.collections import LineCollection
from constants import *

from collections import defaultdict

#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

# ==== I/O SETTINGS ============================================================
fragvst_basename = 'cz_trobjgraph_051'
datadir = f'/home/mweis/scratch/external/clist'

# ==== TEST SETTINGS ===========================================================
test_imin, test_imax = 0, None
test_criterion = 'COCxp30:adrnode'
sim8b_dir1 = '/home/mweis/scratch/thesisdata/corezoom-work-wbox/b.Aa1-Aa4-2zone/out'
sim8b_dir2 = '/home/mweis/scratch/thesisdata/corezoom-scratch-wbox/b.Aa1-Aa4-2zone/out'
test_simdirs = (sim8b_dir1,sim8b_dir2)
xvar, yvar = 'time', 'mass'

# ==== PLOT SETTINGS ===========================================================
fs = mplot.set_fig_preset('display')
mplot.fig_outdir = f'thesis/objgraph/{fragvst_basename}'

# ==== DATA SELECTION ==========================================================
t_min_Myr = 19.15
t_max_Myr = 19.45


#===============================================================================
# ==== OBJVAR SETTINGS =========================================================
#===============================================================================

var_settings = {
'DEFAULT': dict(
    unit=1., ulim=None, logscale=True, ulabel=r'(cgs units)',),
'simtime': dict(label='Time',
    varlabel=r'$t$',
    unit=Myr, ulim=[t_min_Myr,t_max_Myr], logscale=False, ulabel=r'Myr',),
'nstep': dict(label='Simulation timesteps',
    varlabel=r'$N_\mathrm{step}$',
    unit=1., ulim=[0.,1e+6], logscale=False, ulabel=r'count',),
'clmass': dict(label='Object mass',
    varlabel=r'$M_\mathrm{obj}$',
    unit=M_sol, ulim=[1e-2,1e+3], logscale=True, ulabel=r'$M_{\odot}$',),
'clvol': dict(label='Object volume',
    varlabel=r'$V_\mathrm{obj}$',
    unit=pc**3, ulim=[3.16e-7,1.], logscale=True, ulabel=r'pc$^3$',),
'alpha_vir': dict(label='Virial Ratio',
    varlabel=r'$\alpha_\mathrm{vir}$',
    unit=1., ulim=[1e-2,1e+2], logscale=True, ulabel=None,),
'clIrad': dict(label='Object radius',
    varlabel=r'$R_{I}$',
    unit=pc, ulim=[3.16e-3,1.], logscale=True, ulabel=r'pc',),
'accrateS_bulk': dict(label='Accretion rate (bulk motion corrected)',
    varlabel=r'$\dot{M}_\mathrm{bulk}$',
    unit=M_sol/kyr, ulim=[1e-5,1e+2], logscale=True, ulabel=r'$M_{\odot}\,\mathrm{kyr}^{-1}$',),
}

#===============================================================================
# ==== Var Formatting Interface ================================================
#===============================================================================

def GetVarSettings(key):
    vs = dict(label=key, varlabel=key)
    keyname = key.removeprefix('sinkmap_').removeprefix('excl_')
    vs_preset = var_settings.get(keyname, var_settings['DEFAULT'])
    vs.update(vs_preset)
    return vs

def GetVarLabel(key):
    vs = GetVarSettings(key)
    varlabel = vs['varlabel']
    if vs['logscale'] == 'abs':
        label = r'|%s|'%varlabel
    elif vs['logscale'] == 'neg':
        label = r'$-$(%s)'%varlabel
    else:
        label = r'%s'%varlabel
    return label

def GetVarAxisLabel(key):
    vs = GetVarSettings(key)
    ulabel = vs['ulabel']
    varlabel = GetVarLabel(key)
    if ulabel is not None:
      label = r'%s [%s]'%(varlabel, ulabel)
    else:
      label = r'%s'%varlabel
    return label

def GetVarAxisScale(key):
    vs = GetVarSettings(key)
    if vs['logscale'] in [True, 'abs', 'neg', 'pos']:
        axscale = 'log'
    else:
        axscale = 'linear'
    return axscale

def GetVarAxisLim(key):
    vs = GetVarSettings(key)
    vlim = vs.get('ulim', None)
    if vlim is None:
        axlim = [None, None]
    else:
        axlim = vlim
    return axlim   
    
def GetAxSettings(key, axi):
    vs = GetVarSettings(key)
    assert axi in range(3)
    a = ['x', 'y', 'z'][axi]
    ax_settings = {
        a+'label': GetVarAxisLabel(key),
        a+'scale': GetVarAxisScale(key),
        a+'lim':   GetVarAxisLim(key),
    }
    return ax_settings

def GetVarUnit(key):
    vs = GetVarSettings(key)
    if vs['logscale'] == 'neg':
        vs_unit = -1.*vs['unit']
    else:
        vs_unit = vs['unit']
    return vs_unit
    
#def HasAbsScaling(key):
#    vs = GetVarSettings(key)
#    abs_scaling = vs['logscale'] in ['abs',]
#    return abs_scaling

#def RescaleVarAxisLim(key, unit_new):
#    umin, umax = GetVarAxisLim(key)
#    unit = GetVarUnit(key)
#    cgsmin, cgsmax = umin*unit, umax*unit
#    rescaled_lim = cgsmin/unit_new, cgsmax/unit_new
#    return rescaled_lim

def axFormatVars(ax, xvar, yvar):
    ax.set(**GetAxSettings(xvar, 0))
    ax.set(**GetAxSettings(yvar, 1))
    return ax


#===============================================================================
# ==== GRAPH PLOT ==============================================================
#===============================================================================

def ReadVar(nodes, fragdata_t, basedata_t, varname, exclusive=False):
    varname_core = varname.removeprefix('sinkmap_').removeprefix('excl_')
    unit = GetVarUnit(varname_core)
    # TODO: Add scaling (units) from cgs to whatever
    It, Io = zip(*[(n.plotfile_idx,n.object_idx) for n in nodes])
    # Note: Changed data format from cz_fragstat_010_* onwards
    #exc = int(exclusive)
    # Data format: fragdata_t[i_t][inclusive/exclusive][iobject]
    #if varname in fragdata_t[0][exc][0]:
    #    vardata = [fragdata_t[it][exc][io][varname] for it,io in zip(It,Io)]
    # Prepare to read data fields with missing sink data
    vns = varname.removeprefix('sinkmap_')
    # 
    try:
        if unit is None:
            vardata = [(fragdata_t[it].get(varname,fragdata_t[it][vns]))[io] for it,io in zip(It,Io)]
        else:
            vardata = [(fragdata_t[it].get(varname,fragdata_t[it][vns]))[io]/unit for it,io in zip(It,Io)]
    except KeyError:
        try:
            if unit is None:
                vardata = [basedata_t[it][varname] for it in It]
            else:
                vardata = [basedata_t[it][varname]/unit for it in It]
        except KeyError:
            raise KeyError(f'ReadVar: Varname not found: {varname}')
    return np.array(vardata)

def FormatNodes(nodes):
    fD = list()
    for node in nodes:
        fdict = dict()
        #fdict['marker'] = 'o' if node['isroot'] else '+'
        #if node['isleaf']: fdict['marker'] = 'x' 
        fD.append(fdict)
    return fD

def axPlotNodes(ax, nodes, fragdata_t, basedata_t, xvar, yvar):
    fprint('Plotting nodes...')
    xdata = ReadVar(nodes, fragdata_t, basedata_t, xvar)
    ydata = ReadVar(nodes, fragdata_t, basedata_t, yvar)
    nfmt = FormatNodes(nodes)
    for x,y,fmt in zip(xdata,ydata,nfmt):
        ax.scatter(xdata, ydata, **fmt)
    return ax

edgekind_formats = {
    EdgeType.CONTINUED_MAIN: {'alpha':1.0, 'color':'blue', 'linewidth':2.0, 'label':'continuation'},
    EdgeType.CONTINUED_AMBIGUOUS: {'alpha':1.0, 'color':'lightblue', 'linewidth':1.4, 'label':'ambig. cont.'},
    EdgeType.FRAGMENTATION: {'alpha':1.0, 'color':'red', 'linewidth':2.0, 'label':'fragmentation'},
    #EdgeType.RESEGMENTATION: {'alpha':1.0, 'color':'salmon', 'linewidth':1.0, 'label':'resegmentation'},
    #EdgeType.PARENT: {'alpha':1.0, 'color':'yellow', 'linewidth':1.0, 'label':'parent'},
}

def axPlotEdges(ax, G, edges, fragdata_t, basedata_t, xvar, yvar):
    d_edges = EdgesByKind(edges)
    collections = dict()
    for kind, k_edges in d_edges.items():
        fprint(f'Plotting {len(k_edges)} {kind.name} edges... ')
        if len(k_edges) == 0: continue
        style = edgekind_formats[kind]
        nodesL, nodesR = zip(*[(node,succ) for node,succ,data in k_edges])
        xdataL = ReadVar(nodesL, fragdata_t, basedata_t, xvar)
        xdataR = ReadVar(nodesR, fragdata_t, basedata_t, xvar)
        ydataL = ReadVar(nodesL, fragdata_t, basedata_t, yvar)
        ydataR = ReadVar(nodesR, fragdata_t, basedata_t, yvar)
        #
        isleafL = [G.nodes[n]['isleaf'] for n in nodesL]
        isleafR = [G.nodes[n]['isleaf'] for n in nodesR]
        leafedgeLR = [bL*bR for bL,bR in zip(isleafL,isleafR)]
        leafedgeR = [bR for bL,bR in zip(isleafL,isleafR)]
        #isrootL = [G.nodes[n]['isroot'] for n in nodesL]
        #isrootR = [G.nodes[n]['isroot'] for n in nodesR]
        #rootedge = [bL*bR for bL,bR in zip(isrootL,isrootR)]
        #ils = [['dotted','solid','dashed','dashdot'][le+2*re] for le,re in zip(leafedge,rootedge)]
        #style['linestyles'] = ils
        #style['linewidth'] = [.7+re*2.+le*1. for re,le in zip(rootedge,leafedge)]
        if kind==EdgeType.CONTINUED_MAIN:
            style['alpha'] = [.5*(1.+int(b)) for b in leafedgeLR]
        if kind==EdgeType.CONTINUED_AMBIGUOUS:
            continue
        if kind==EdgeType.FRAGMENTATION:
            style['alpha'] = [int(b) for b in leafedgeLR]
        #
        print(f'xdataL range:', np.min(xdataL), np.max(xdataL))
        print(f'xdataR range:', np.min(xdataR), np.max(xdataR))
        print(f'ydataL range:', np.min(ydataL), np.max(ydataL))
        print(f'ydataR range:', np.min(ydataR), np.max(ydataR))
        #
        Lcs = np.column_stack([xdataL,ydataL])
        #print(f'xdataL.shape: {xdataL.shape}')
        #print(f'Lcs.shape: {Lcs.shape}')
        #print(f'len(alpha): {len(style['alpha'])}')
        Rcs = np.column_stack([xdataR,ydataR])
        segments = np.stack([Lcs,Rcs], axis=1)
        lc = LineCollection(segments, **style)
        ax.add_collection(lc)
        collections[kind] = lc
    fprint(f'...OK')
    return collections


#===============================================================================
# ==== GRAPH STAT FILTERING ====================================================
#===============================================================================

def FilterNodeStat(nodes, fragdata_t, basedata_t, var, vmin=None, vmax=None):
    values = ReadVar(nodes, fragdata_t, basedata_t, var)
    if vmin is not None:
        nodes_f = [n for n,val in nodes if (vmin<=val)]
    if vmax is not None:
        nodes_f = [n for n,val in nodes_f if (val<=vmax)]
    return nodes_f

def GroupByStat(G, nodes, fragdata_t, basedata_t, var, partition, minlen=1):
    values = ReadVar(nodes, fragdata_t, basedata_t, var)
    print(f'#D {var} VALUES:')
    print(set(values))
    nbins = len(partition)+1
    nodebins = np.searchsorted(partition, values)
    gdict = dict()
    for i in range(nbins):
        bin_nodes = [n for n,nb in zip(nodes,nodebins) if nb==i]
        if len(bin_nodes)>=minlen: gdict[i] = bin_nodes
    return gdict


#===============================================================================
# ==== GRAPH FILTERING =========================================================
#===============================================================================

def EdgesByKind(edges):
    edgekinds = list(edgekind_formats.keys())
    d_edges = dict()
    for kind in edgekinds:
        d_edges[kind] = [edge for edge in edges if edge[2]['kind']==kind]
    return d_edges

def FilterNodeProp(G, nodes, prop, value=None, vrange=None):
    if value is not None:
        nodes_f = [n for n in nodes if G.nodes[n][prop] == value]
    elif vrange is not None:
        vmin, vmax = vrange
        nodes_fa = [n for n in nodes if G.nodes[n][prop] <= vmax]
        nodes_f = [n for n in nodes_fa if vmin <= G.nodes[n][prop]]
    return nodes_f

def MergeAmbiguous(G, nodegroups):
    # TODO
    # ATM, THIS IS ALREADY DONE BY ASSIGNING THE PERSISTENT ID
    # EVEN ON AMBIGUOUS CONTINUATIONS IN cz_trfrag_catalog
    return nodegroups

def ComplementContinuations(objgraph, nodes, debug=True):
    ids = {objgraph.nodes[n]['persistentID'] for n in nodes}
    allnodes = objgraph.nodes
    nodes_comp = [n for id in ids for n in FilterNodeProp(objgraph, allnodes, 'persistentID', value=id)]
    assert set(nodes).issubset(nodes_comp)
    if debug:
        n_nodes = len(set(nodes))
        n_comp = len(set(nodes_comp))
        print(f'Complemented {n_comp-n_nodes} continuations to {n_nodes} nodes.')
    return nodes_comp

def GroupByContinuation(objgraph, nodes, debug=True, minlen=10):
    gdict = defaultdict(list)
    for n in nodes:
        id = objgraph.nodes[n]['persistentID']
        gdict[id].append(n)
    # Sort into standard dict, sorting out short continuation paths
    contdict = {id:nl for id,nl in gdict.items() if len(nl)>=minlen}
    #
    if debug:
        print(f'Continuation lengths (per ID):')
        for id,nl in contdict.items():
            print(f'ID: {id} | LEN: {len(nl)}')
    return dict(contdict)

def FilterEdges(edges, prop, value):
    edges_f = [edge for edge in edges if edge[2][prop]==value]
    return edges_f

def NodeEdges(objgraph, node, kind=None):
    ne = objgraph.edges(node, data=True)
    if kind is None:
        return ne
    else:
        return FilterEdges(ne, 'kind', kind)

def ContainedNodes(objgraph, node, addself=False, debug=True, childtraverse=True, maxdepth=50):
    etype_child = EdgeType.PARENT # XXX FIXED IN GRAPH, SWAP OUT FOR CHILD AFTER COMPUTING NEW GRAPH
    descedges = NodeEdges(objgraph, node, kind=EdgeType.DESCENDANT)
    childedges = NodeEdges(objgraph, node, kind=etype_child)
    if childtraverse:
        subnodes = list()
        for depth in range(maxdepth):
            if not childedges: break
            childnodes = [edge[1] for edge in childedges]
            childedges = [edge for ce in childnodes for edge in NodeEdges(objgraph, ce, kind=etype_child)]
            subnodes.extend(childnodes)
        else:
            raise ValueError(f'ContainedNodes: Maximum recursion depth ({maxdepth}) breached!')
    else:
        subnodes = [edge[1] for edge in descedges]+[edge[1] for edge in childedges]
    if debug:
        print(f'#D Node: {node}, {len(subnodes)} Subnodes')
    if addself:
        subnodes.append(node)
    return subnodes


#===============================================================================
# ==== TEMPLATE GRAPH PLOTTING =================================================
#===============================================================================

def axPlotNodeOE(ax, objgraph, nodes, fragdata_t, basedata_t, xvar, yvar):
    outedges = [edge for n in nodes for edge in objgraph.out_edges(n, data=True)]
    axPlotEdges(ax, objgraph, outedges, fragdata_t, basedata_t, xvar, yvar)
    xvar_core = xvar.removeprefix('sinkmap_').removeprefix('excl_')
    yvar_core = yvar.removeprefix('sinkmap_').removeprefix('excl_')
    axFormatVars(ax, xvar_core, yvar_core)
    return

def FigNodeOE(objgraph, nodes, fragdata_t, basedata_t, xvar, yvar):
    fig, ax = plt.subplots()
    axPlotNodeOE(ax, objgraph, nodes, fragdata_t, basedata_t, xvar, yvar)
    mplot.ax_format_ticks(ax)
    mplot.ax_legend(ax)
    return fig, ax


def PlotLeafsByRegion(objgraph, fragdata_t, basedata_t, xvar, yvar, nreg=20):
    '''
    Do one plot per region, each group consisting of
    all the continuations of all leafs crossing the region.
    '''
    GN = objgraph.nodes
    leafnodes = FilterNodeProp(objgraph, GN, 'isleaf', value=True)
    # Comment this in (and change GroupByStat acc) if nodes that are part-time leafs ought to be
    # grouped by the region location even outside the leaf state:
    #leafcontnodes = ComplementContinuations(objgraph, leafnodes)
    statkey = 'containing_regflags'
    statparts = [(1<<ireg)-.5 for ireg in range(nreg)]
    reggroups = GroupByStat(objgraph, leafnodes, fragdata_t, basedata_t, statkey, statparts, minlen=1)
    for ireg,leafgroup in reggroups.items(): #enumerate(rngroups):
        # Complement continuations again in case some leaf continuation path was cut into multiple region groups.
        # Note that this is expected to cause those path to be displayed in every applicable region plot.
        ## plnodes = ComplementContinuations(objgraph, leafgroup) # XXX
        plnodes = leafgroup
        # Create group figure
        fig, ax = FigNodeOE(objgraph, plnodes, fragdata_t, basedata_t, xvar, yvar)
        axtitle = f'Region {ireg}' if ireg>0 else 'Outside Regions'
        mplot.ax_title(ax, axtitle)
        # Save group graph
        figname = f'by_region_{ireg}'
        subpath = f'{xvar}-{yvar}'
        mplot.savefig(fig, figname, subpath)

def PlotByRoot(objgraph, fragdata_t, basedata_t, xvar, yvar):
    '''
    Do one plot per node group, each group consisting of
    the subnodes of all continuations of a selected rootnode.
    '''
    GN = objgraph.nodes
    rootnodes = FilterNodeProp(objgraph, GN, 'isroot', value=True)
    rootcontnodes = ComplementContinuations(objgraph, rootnodes)
    rngroups = GroupByContinuation(objgraph, rootcontnodes, minlen=10)
    for id,group in rngroups.items(): #enumerate(rngroups):
        # Collect all (sub-)nodes relevant to the group
        plnodes = list()
        for rootnode in group:
            subnodes = ContainedNodes(objgraph, rootnode, addself=True)
            plnodes.extend(subnodes)
        # Create group figure
        fig, ax = FigNodeOE(objgraph, plnodes, fragdata_t, basedata_t, xvar, yvar)
        mplot.ax_title(ax, f'Root Cloud {id}')
        # Save group graph
        figname = f'by_root_{id}'
        subpath = f'{xvar}-{yvar}'
        mplot.savefig(fig, figname, subpath)

def PlotWithPropVal(objgraph, fragdata_t, basedata_t, xvar, yvar, prop, propval):
    GN = objgraph.nodes
    propnodes = FilterNodeProp(objgraph, GN, prop, value=propval)
    plnodes = ComplementContinuations(objgraph, propnodes)
    fig, ax = FigNodeOE(objgraph, plnodes, fragdata_t, basedata_t, xvar, yvar)
    # Save figure
    figname = f'prop-{prop}_{propval}'
    subpath = f'{xvar}-{yvar}'
    return mplot.savefig(fig, figname, subpath)    

def PlotEntireGraph(objgraph, fragdata_t, basedata_t, xvar, yvar):
    GN = objgraph.nodes
    fig, ax = FigNodeOE(objgraph, GN, fragdata_t, basedata_t, xvar, yvar)
    # Save figure
    figname = f'all'
    subpath = f'{xvar}-{yvar}'
    return mplot.savefig(fig, figname, subpath)    


#===============================================================================
# ==== SPECIFIC NON-TEMPLATE PLOTS =============================================
#===============================================================================

# DENDRO ROOT: CORE MASS FRACTION (SUM AND PER-CORE)

#===============================================================================
# ==== TEST ====================================================================
#===============================================================================

def Test():
    ## Open and object node and edge labels from all used plotfiles
    fprint('--- > Collecting plotfiles ---')
    plotfiles = cztrmatch.CollectPlotfiles(test_simdirs)
    plotfiles_sorted = cztrmatch.OrderPlotfiles(plotfiles)
    plotfiles_test = plotfiles_sorted[test_imin:test_imax]
    fprint('--- > Reading fragment data ---')
    fragdata_t, basedata_t = czfragstat.get_fdc(plotfiles_test, test_criterion)
    dkeys = list(fragdata_t[0].keys())
    fprint('--- > Reading object graph ---')
    objgraph = czfragcat.get_objgraph(plotfiles_test, test_criterion)
    fprint('--- > Plotting fragment graphs ---')
    Dt = objgraph, fragdata_t, basedata_t
    #PlotWithPropVal(*Dt, 'sinkmap_clmass', 'sinkmap_alpha_vir', 'isleaf', True)
    #PlotWithPropVal(*Dt, 'sinkmap_clIrad', 'sinkmap_alpha_vir', 'isleaf', True)
    #PlotWithPropVal(*Dt, 'sinkmap_clmass', 'sinkmap_clIrad', 'isleaf', True)
    # PlotWithPropVal(*Dt, 'simtime', 'sinkmap_clmass', 'isleaf', True)
    #xvars = ['clmass','clIrad','simtime','accrateS_bulk']
    #yvars = ['alpha_vir','Etot_self','clmass','clIrad','accrateS_bulk','cldisp','cldispvr','cldispvp']
    #varcombos = list()
    #for xvar in xvars:
    #    for yvar in yvars:
    #        if xvar==yvar: continue
    #        if set([xvar,yvar]) in varcombos: continue
    #        PlotWithPropVal(*Dt, xvar, yvar, 'isleaf', True)
    #        varcombos.append(set([xvar,yvar]))
    #for yvar in ('clps_mass', 'clps_volume', 'cldisp', 'cldispvr', 'cldispvp', 'dds_r_sphere', 'dds_r_sheet', 'dds_r_filament'):
    #    for prefix in ('', 'excl__'):
    #        pyvar = prefix+yvar
    #        PlotEntireGraph(*Dt, pyvar)
    #        PlotByRoot(*Dt, pyvar)
    #        PlotWithPropVal(*Dt, pyvar, 'isroot', True)
    #        PlotWithPropVal(*Dt, pyvar, 'isleaf', True)

    PlotWithPropVal(*Dt, 'simtime', 'sinkmap_clmass', 'isleaf', True)
    for yvar in ['sinkmap_accrateS_bulk', 'sinkmap_accrateS_euler']:
        PlotLeafsByRegion(*Dt, 'sinkmap_clmass', yvar)
        PlotWithPropVal(*Dt, 'simtime', yvar, 'isleaf', True)


#===============================================================================
# ==== LOBBY ===================================================================
#===============================================================================

if __name__ == '__main__':
    if len(sys.argv)<2:
        Test()
    else:
        pass # TODO XXX

    print('FINISHED!')
