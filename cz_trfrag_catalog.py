import os, sys
import numpy as np
import networkx as nx
from dataclasses import dataclass

import libmf.micflash as mflash
import libmf.tools as tools
import libmf.micplot as mplot
from libmf.tools import fprint
import cz_tools as cz
import cl_find_181 as clfind
import cz_tracerown as cztrown
import cz_tracermatch as cztrmatch
import cz_trfraginterpret as cztrfi
from cz_tracerfrag import EdgeType, NodeType

#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

# ==== I/O SETTINGS ============================================================
objgraph_basename = 'cz_objgraph_004_'
datadir = f'/home/mweis/scratch/external/clist'

# ==== TEST SETTINGS ===========================================================
test_imin, test_imax = 0, None
test_criterion = 'COCxp30:adrnode'#'COCxp30:adrnode:czfrag'
sim8b_dir1 = '/home/mweis/scratch/thesisdata/corezoom-work-wbox/b.Aa1-Aa4-2zone/out'
sim8b_dir2 = '/home/mweis/scratch/thesisdata/corezoom-scratch-wbox/b.Aa1-Aa4-2zone/out'
test_simdirs = (sim8b_dir1,sim8b_dir2)


#===============================================================================
# ==== OBJECT GRAPH STRUCTURE ==================================================
#===============================================================================

@dataclass(frozen=True)
class GraphNode:
    plotfile_idx: int
    object_idx: int

edgetypes = list(et for et in EdgeType if (et.value<999))
edgetypes_ignore = (EdgeType.NOISE, EdgeType.UNASSIGNED)

def ObjectCounts(edgelabels_t):
    n_t = len(edgelabels_t)+1
    n_obj_t = np.full(n_t, -1, dtype=int)
    elshapes_t = [el.shape for el in edgelabels_t]
    print(f'#D elshapes_t: {elshapes_t}')
    n_obj_tL, n_obj_tR = np.array(list(zip(*elshapes_t)))
    assert np.all(n_obj_tL[1:]==n_obj_tR[:-1])
    n_obj_t[:-1] = n_obj_tL
    n_obj_t[-1] = n_obj_tR[-1]
    assert np.all(n_obj_t>=0)
    return n_obj_t

def CreateGraphNodes(G, n_obj_t):
    print('Adding object nodes...')
    sys.stdout.flush()
    for t_idx, nobj in enumerate(n_obj_t):
        Iobj = np.arange(nobj)
        Iplt = np.full_like(Iobj, t_idx)
        GN = (GraphNode(*idx) for idx in zip(Iplt,Iobj))
        G.add_nodes_from(GN)
    return G

def InsertGraphNodeInfo(G, Moo_own_t):
    print('Inserting object node info...')
    sys.stdout.flush()
    for t_idx, Moo_own in enumerate(Moo_own_t):
        isleaf = cztrown.LeafObjMask(Moo_own)
        isroot = cztrown.RootObjMask(Moo_own)
        depth = cztrown.ObjDepth(Moo_own)
        nobj = len(isleaf)
        for obj_idx in range(nobj):
            gn = GraphNode(t_idx,obj_idx)
            G.nodes[gn]['isleaf'] = isleaf[obj_idx]
            G.nodes[gn]['isroot'] = isroot[obj_idx]
            G.nodes[gn]['depth'] = depth[obj_idx]
    return G

def CreateGraphEdgesEvol(G, edgelabels_t):
    print('Adding evolution edges...')
    sys.stdout.flush()
    for t_idx, edgelabels in enumerate(edgelabels_t):
        for et in edgetypes:
            if et in edgetypes_ignore: continue
            AB = np.where(edgelabels==et)
            GE = [(GraphNode(t_idx,a),GraphNode(t_idx+1,b),{'kind':et}) for a,b in zip(*AB)]
            G.add_edges_from(GE)
    return G

def CreateGraphEdgesDendroDirect(G, Moo_own_t):
    print('Adding dendrogram edges (direct)...')
    sys.stdout.flush()
    for t_idx, Moo_own in enumerate(Moo_own_t):
        IK = np.where(Moo_own)
        GEc = [(GraphNode(t_idx,i),GraphNode(t_idx,k),{'kind':EdgeType.CHILD}) for i,k in zip(*IK)]
        G.add_edges_from(GEc)
        GEp = [(GraphNode(t_idx,k),GraphNode(t_idx,i),{'kind':EdgeType.PARENT}) for i,k in zip(*IK)]
        G.add_edges_from(GEp)
    return G

def CreateGraphEdgesDendroIndirect(G, Moo_contain_t, Moo_own_t=None):
    print('Adding dendrogram edges (indirect)...')
    sys.stdout.flush()
    for t_idx, Moo_cont in enumerate(Moo_contain_t):
        if Moo_own_t is not None:
            Moo_own = Moo_own_t[t_idx]
            Moo_contain_indirect = Moo_cont*(~Moo_own)
            IK = np.where(Moo_contain_indirect)
        else:
            IK = np.where(Moo_cont)
        GEd = [(GraphNode(t_idx,i),GraphNode(t_idx,k),{'kind':EdgeType.DESCENDANT}) for i,k in zip(*IK)]
        G.add_edges_from(GEd)
        GEa = [(GraphNode(t_idx,k),GraphNode(t_idx,i),{'kind':EdgeType.ANCESTOR}) for i,k in zip(*IK)]
        G.add_edges_from(GEa)
    return G


def BuildObjectGraph(edgelabels_t, Moo_contain_t=None, Moo_own_t=None):
    print('-> Building object graph...')
    sys.stdout.flush()
    G = nx.DiGraph()
    n_obj_t = ObjectCounts(edgelabels_t)
    CreateGraphNodes(G, n_obj_t)
    CreateGraphEdgesEvol(G, edgelabels_t)
    if Moo_contain_t is not None:
        CreateGraphEdgesDendroIndirect(G, Moo_contain_t, Moo_own_t)
    if Moo_own_t is not None:
        CreateGraphEdgesDendroDirect(G, Moo_own_t)
        InsertGraphNodeInfo(G, Moo_own_t)
    InsertTraversalinfo(G, n_obj_t)
    return G



#===============================================================================
# ==== OBJECT GRAPH TRAVERSAL ==================================================
#===============================================================================

continuation_types = (EdgeType.CONTINUED_MAIN, EdgeType.CONTINUED_AMBIGUOUS)

def TraverseByKind(G, node, kind): ### XXX NOT WORKING XXX
    successors = G.successors(node)
    succ_edges = (G.edges[node,succ] for succ in successors)
    traversal_edges = (edge for edge in succ_edges if edge['kind']==kind)
    nodes_next = (n1 for n0,n1 in traversal_edges)
    return nodes_next

def ReverseByKind(G, node, kind): ### XXX NOT WORKING XXX
    predecessors = G.predecessors(node)
    pred_edges = (G.edges[pred,node] for pred in predecessors)
    reversal_edges = (edge for edge in pred_edges if edge['kind']==kind)
    nodes_prev = (n0 for n0,n1 in reversal_edges)
    return nodes_prev

def ContinuationNext(G, node):
    continuations = [succ
        for n0, succ, data in G.out_edges(node, data=True)
        if data['kind'] in continuation_types
        ]
    if continuations:
        assert len(continuations) == 1
        node_next = continuations[0]
        return node_next
    return None
    
def InsertTraversalinfo(G, n_obj_t, maxdepth=10000, debug=True):
    print('-> Inserting traversal info...')
    sys.stdout.flush()
    nextPersistentID = 0
    for t_idx, nobj in enumerate(n_obj_t):
        if debug:
            print(f'#D t_idx: {t_idx}')
            sys.stdout.flush()
        nodes = (GraphNode(t_idx, obj_idx) for obj_idx in range(nobj))
        for node in nodes:
            if 'persistentID' in G.nodes[node]:
                persistentID = G.nodes[node]['persistentID']
            else:
                node0 = node
                persistentID = nextPersistentID
                G.nodes[node]['persistentID'] = persistentID
                G.nodes[node]['isfirst'] = True
                G.nodes[node]['islast'] = True
                G.nodes[node]['contstep'] = 0
                G.add_edge(node,node0, kind=EdgeType.CONT_FIRST)
                nextPersistentID = nextPersistentID + 1
                if debug: print(f'ID:{persistentID}: {node}', end='')
                for recursions in range(maxdepth):
                    node_next = ContinuationNext(G, node)
                    if debug: print(f'->{node_next}', end='')
                    if node_next is None: break
                    G.nodes[node]['islast'] = False
                    node = node_next
                    G.nodes[node]['persistentID'] = persistentID
                    G.nodes[node]['isfirst'] = False
                    G.nodes[node]['islast'] = True
                    G.nodes[node]['contstep'] = recursions+1
                    G.add_edge(node,node0, kind=EdgeType.CONT_FIRST)
                else:
                    raise RuntimeError('InsertTraversalinfo: Exceeded max. recursion depth')
                if debug: print()
    return G



#===============================================================================
# ==== DATA COLLECTORS =========================================================
#===============================================================================

def CollectDendroRelations(filelist, criterion):
    print('-> Reading object relation matrices...')
    sys.stdout.flush()
    Moo_contain_t = list()
    Moo_own_t = list()
    for pf in filelist:
        leafdata, octree = cz.LoadPlotfile(pf)
        objmasks = clfind.get_cmasks(leafdata, octree, criterion)
        Moo_contain = cztrown.ObjObjContainmentMatrix(objmasks)
        Moo_own = cztrown.ObjObjOwnershipMatrix(Moo_contain)
        del(octree)
        leafdata.close()
        Moo_contain_t.append(Moo_contain)
        Moo_own_t.append(Moo_own)
        print(pf)
        sys.stdout.flush()
    return Moo_contain_t, Moo_own_t


#===============================================================================
# ==== RESULT PROCESSING (MAIN) ================================================
#===============================================================================

def determine_objgraph(plotfiles_sorted, criterion):
    # Open and object node and edge labels from all used plotfiles
    #matrices_ = cztrmatch.get_matchmatrices_collection(plotfiles_sorted, criterion)
    print(f'-> Loading edge data ({criterion}({len(plotfiles_sorted)}))...')
    edc = cztrfi.get_edgedata_collection(plotfiles_sorted, criterion)
    # Unpack edgedata
    print(f'-> Unpacking edge data ({criterion}({len(plotfiles_sorted)}))...')
    labels_t, RTCxx_et_t = zip(*edc)
    edgelabels_t, nodel_Aout_t, nodel_Bin_t, edgel_Aout_t, edgel_Bin_t = zip(*labels_t)
    # Build object graph
    Moo_contain_t, Moo_own_t = CollectDendroRelations(plotfiles_sorted, criterion)
    objgraph = BuildObjectGraph(edgelabels_t, Moo_contain_t=Moo_contain_t, Moo_own_t=Moo_own_t)
    #
    return objgraph


#===============================================================================
# ==== RESULT I/O ==============================================================
#===============================================================================

def objgraph_path(filelist_sorted, criterion, debug=True):
    pf0 = filelist_sorted[0]
    simdir = os.path.dirname(pf0)
    filelist_fp = tools.get_strlist_fingerprint(filelist_sorted)
    crit_fp = clfind.get_criterion_fingerprint(criterion)
    prefix = objgraph_basename +crit_fp +'_' +filelist_fp
    filepath = tools.get_collection_path(simdir, prefix, outdir=datadir)
    if debug>0:
        print(f'OBJGRAPH PATH: {filepath}')
        sys.stdout.flush()
    return filepath

def prepare_objgraph(filelist_sorted, criterion):
    ''' Compile and save a objgraph to criterion if not existing yet '''
    filepath = objgraph_path(filelist_sorted, criterion)
    # Check if the list is already in place
    if os.path.exists(filepath) and not 'test' in criterion:
        print(f'-> Object graph already exists: {filepath}')
        sys.stdout.flush()
    else:
        print(f'-> PREPARING OBJECT GRAPH: {criterion}')
        print(f'-> : {filelist_sorted[0]} ... {filelist_sorted[-1]}')
        sys.stdout.flush()
        objgraph = determine_objgraph(filelist_sorted, criterion)
        tools.save_data(objgraph, filepath, force=False)
    return filepath    

def get_objgraph(filelist_sorted, criterion, save=True):
    ''' Obtain object graph according to criterion and filelist; compile if necessary '''
    filepath = objgraph_path(filelist_sorted, criterion)
    #
    force = False    
    if os.path.exists(filepath):
        print(f'-> Loading object graph ({criterion})')
        sys.stdout.flush()
        try:
            objgraph = tools.load_data(filepath)
        except:
            print('!! Warning: Could not load object graph!!')
            force = True
        else:
            return objgraph
    objgraph = determine_objgraph(filelist_sorted, criterion)
    if save:
        print(f'-> Saving object graph ({criterion})')
        sys.stdout.flush()
        try:
            tools.save_data(objgraph, filepath, force=force)
        except:
            print('!! Warning: Could not save object graph !!')
    return objgraph


#===============================================================================
# ==== TEST ====================================================================
#===============================================================================

def Test():
    # Open and object node and edge labels from all used plotfiles
    plotfiles = cztrmatch.CollectPlotfiles(test_simdirs)
    plotfiles_sorted = cztrmatch.OrderPlotfiles(plotfiles)
    plotfiles_test = plotfiles_sorted[test_imin:test_imax]
    prepare_objgraph(plotfiles_test, test_criterion)


#===============================================================================
# ==== LOBBY ===================================================================
#===============================================================================

if __name__ == '__main__':
    if len(sys.argv)<2:
        Test()
    else:
        criterion = sys.argv[1]
        dirs = sys.argv[2:]
        if len(dirs)>1:
            print('CAUTION: Please make sure all given directories belong to the same simulation!')
        plotfiles = cztrmatch.CollectPlotfiles(dirs)
        plotfiles_sorted = cztrmatch.OrderPlotfiles(plotfiles)
        prepare_objgraph(plotfiles_sorted, criterion)
    print('FINISHED!')
