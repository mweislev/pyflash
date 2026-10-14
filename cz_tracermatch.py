import os, sys
import fnmatch
import numpy as np

import libmf.micflash as mflash
import libmf.tools as tools
from libmf.tools import fprint

import cz_tools as cz
import constants as cu

import cl_find_182 as clfind
import cz_tracerown as cztrown

__author__ = "Michael Weis"


#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

# ==== BASE SETTINGS ===========================================================
particle_idvar = 'hash'

# ==== INPUT FILE SETTINGS =====================================================
plotfile_include_patterns = ['*hdf5_plt*_????',] #['*hdf5_chk_*', '*hdf5_plt*']
plotfile_exclude_patterns = ['*_forced_*', '*.npz', '*-rd_hdf5*', '*.pkl', '*.pre']

# ==== I/O SETTINGS ============================================================
mmatrix_basename = 'cz_matchmatrix_003_'
trobject_basename = 'cz_trobject_004_'

datadir = f'/home/mweis/scratch/external/clist'


#===============================================================================
# ==== HELPER ==================================================================    
#===============================================================================

def FindPlotfiles(simdir, exclude=None, include=None, debug=True):
    #
    if exclude is None:
        exclude = plotfile_exclude_patterns
    if include is None:
        include = plotfile_include_patterns
    if debug:
        print('Exclude patterns:')
        print(exclude)
        print('Include patterns:')
        print(include)
    plotfiles_include = list()
    plotfiles_exclude = list()
    for dirpath, dirnames, dirfiles in os.walk(simdir, topdown=True):
        dirnames[:] = [d for d in dirnames if d != '.venv']
        print(f'{dirpath}')
        if '/.' in dirpath: continue
        if 'test' in dirpath: continue
        if 'cldata' in dirpath: continue
        if 'particle_files' in dirpath: continue
        for pattern in include:
            files = fnmatch.filter(dirfiles, pattern)
            filepathes = [os.path.join(dirpath,fn) for fn in files]
            plotfiles_include.extend(filepathes)
        for pattern in exclude:
            files = fnmatch.filter(dirfiles, pattern)
            filepathes = [os.path.join(dirpath,fn) for fn in files]
            plotfiles_exclude.extend(filepathes)
    plotfiles = set(plotfiles_include) - set(plotfiles_exclude)
    return list(plotfiles)

def CollectPlotfiles(dirpaths, **kwargs):
    plotfiles = list()
    for idir,dp in enumerate(dirpaths):
        try:
            assert os.path.isdir(dp)
        except AssertionError:
            continue
        # Add plotfiles directly in directory
        fargs = dict(debug=(idir==0))
        fargs.update(kwargs)
        plotfiles.extend(FindPlotfiles(dp, **fargs))
        # Add plotfiles in subdirs # this is already done by walk in FindPlotfiles
        #subdirs = os.listdir(dp)
        #subpaths = [os.path.join(dp,sd) for sd in subdirs]
        #plotfiles.extend(CollectPlotfiles(subpaths))
    return plotfiles

def EvaluatePlotfile(plotfile):
    try:
        leafdata = mflash.plotfile(plotfile, memorize=False)
        mtime = os.path.getmtime(plotfile),
        ctime = os.path.getctime(plotfile),
        atime = os.path.getatime(plotfile),
        pfdata = dict(
            simtime = leafdata['real scalars/time'],
            nstep = leafdata['integer scalars/nstep'],
            ts = np.min([mtime, ctime, atime]),
        )
    except KeyboardInterrupt:
        raise
    except Exception:
        # If the file is not readable, the step number will be given as nan.
        # Note that as of numpy 1.4, sort and argsort will put nan values to the end.
        print(f'WARNING: Plotfile not readable: {plotfile}')
        pfdata = dict(nstep=999999999999, simtime=np.inf, ts=np.nan)
    return pfdata

def OrderPlotfiles(plotfiles, shingled=False, debug=True, prefer_chk=False):
    pfdata_ = [EvaluatePlotfile(pf) for pf in plotfiles]
    nstep_ = np.array([d['nstep'] for d in pfdata_])
    simtime_ = np.array([d['simtime'] for d in pfdata_])
    timestamp_ = np.array([d['ts'] for d in pfdata_])
    # Assume shingled data, i.e. restarts back to earlier checkpoints,
    # where old plotfiles were kept. This will only work if either the
    # file creation times or the file modify times are intact.
    # Note that this will not worked on synced files from scratch partitions,
    # since rsync resets the former, while touch resets the latter.
    if shingled:
        I = np.argsort(timestamp_)
        ipf = I[0]
        plotfiles_ordered = [plotfiles[ipf],]
        nstep_ordered = [nstep_[ipf],]
        simtime_ordered = [simtime_[ipf],]
        for ipf in I[1:]:
            ts = timestamp_[ipf]
            if ts != ts: break
            step = nstep_[ipf]
            il = np.searchsorted(nstep_ordered,step)
            pf = plotfiles[ipf]
            plotfiles_ordered = plotfiles_ordered[:il] + [pf,]
            nstep_ordered = nstep_ordered[:il] + [step,]
            simtime_ordered = simtime_ordered[:il] + [simtime_[ipf],]
    # Assume that there is only one timeline of plotfiles present.
    # If in doubt, the user needs to clean up first by hand.
    else:
        I = np.argsort(nstep_)
        nstep_prior = -1
        plotfiles_ordered = list()
        nstep_ordered = list()
        simtime_ordered = list()
        for ipf in I:
            ts = timestamp_[ipf]
            if ts != ts: continue
            step = nstep_[ipf]
            pf = plotfiles[ipf]
            simtime = simtime_[ipf]
            if step == nstep_prior:
                # If nstep already in list:
                if prefer_chk == ('_chk_' in pf):
                    # Preferred filetype: Replace listed file by newly found one
                    plotfiles_ordered[-1] = pf
                else:
                    continue
            else:
                plotfiles_ordered.append(pf)
                nstep_ordered.append(step)
                simtime_ordered.append(simtime)
                nstep_prior = step
    print('PLOTFILES:')
    for nstep, simtime, pf in zip(nstep_ordered, simtime_ordered, plotfiles_ordered):
        print(f'({nstep}: {simtime/cu.Myr:.6f}) {pf}')
    return plotfiles_ordered


#===============================================================================
# ==== PARTICLE MASKS <-> PARTICLE ID SETS =====================================
#===============================================================================

def MaskFromParticleIDSet(partdata, partset):
    '''
    Create a particle mask for a specific particle file,
    using particle IDs that are unique and persistent through time.
    '''
    haystack = partdata[particle_idvar]
    needles = np.fromiter(partset, dtype=haystack.dtype)
    partmask = np.isin(haystack, needles, assume_unique=True)
    return partmask

def ParticleIDSetFromMask(partdata, partmask):
    '''
    Create a unique, persistent set of particle IDs
    from a particle mask of a specific plotfile.
    The particle IDs are useful to re-identify the same
    particles over the course of multiple particle files.
    '''
    partids_masked = partdata[particle_idvar][partmask]
    n_masked = np.sum(partmask)
    partidset = set(partids_masked)
    n_nonunique = n_masked - len(partidset)
    if n_nonunique>0:
        print(f'WARNING: Detected {n_nonunique}/{n_masked} non-unique particle ID entries ({100.*n_nonunique/n_masked}%).')
    return partidset


#===============================================================================
# ==== CLUSTER COMPARATOR ======================================================
#===============================================================================

def MatchRatios(IDset1, IDset2):
    sharedIDs = IDset1.intersection(IDset2)
    anyIDs = IDset1.union(IDset2)
    nshare = len(sharedIDs)
    n1 = len(IDset1)
    n2 = len(IDset2)
    if n1==0 or n2==0:
        return 0.,0.,0.
    # Object 1 is a part of object 2 to this fraction:
    f_forward = nshare * (1./n1)
    # Object 2 is a part of object 1 to this fraction:
    f_reverse = nshare * (1./n2)
    # Both objects are the same to this fraction (Jaccard index):
    ntotal = len(anyIDs)
    f_jaccard = nshare * (1./ntotal)
    #
    return f_forward, f_reverse, f_jaccard

def IntersectionMatrix(ObjIDsets1, ObjIDsets2):
    l1, l2 = len(ObjIDsets1), len(ObjIDsets2)
    M_forward = np.full((l1,l2), fill_value=np.nan)
    M_reverse = np.full((l1,l2), fill_value=np.nan)
    M_jaccard = np.full((l1,l2), fill_value=np.nan)
    for i,s1 in enumerate(ObjIDsets1):
        if len(s1)==0:
            fprint(f'Warning: ObjIDset1({i}/{len(ObjIDsets1)}) empty!')
        mrl = [MatchRatios(s1,s2) for s2 in ObjIDsets2]
        M_forward[i], M_reverse[i], M_jaccard[i] = zip(*mrl)
    for j,s2 in enumerate(ObjIDsets2):
        if len(s2)==0:
            fprint(f'Warning: ObjIDset2({j}/{len(ObjIDsets2)}) empty!')
    return M_forward, M_reverse, M_jaccard

def ParticleCounts(ObjIDsets):
    N_parts = np.array([len(s) for s in ObjIDsets])
    return N_parts


#===============================================================================
# ==== RESULT PROCESSING =======================================================
#===============================================================================

def TracerObjectSets(hdfdata, octree, criterion, trdata=None, objmasks=None):
    tools.fprint(f'Calculating tracer object set ({hdfdata.basename()})...')
    # Load node masks for Eulerian dendrogram objects
    if objmasks is None:
        objmasks = clfind.get_cmasks(hdfdata, octree, criterion)
    nobj = len(objmasks)
    # Determine the hierarchical structure of the dendrogram node masks
    Moo_contain = cztrown.ObjObjContainmentMatrix(objmasks)
    # Load tracer particle data
    if trdata is None:
        partfile = cz.PartfileName(hdfdata)
        trdata = cz.PartfileData(partfile)
    # Find out which tracer particles are contained in the space occupied by each dendrogram node
    tr_containsets = [ParticleIDSetFromMask(trdata,cztrown.ObjTracerContainmentMask(octree,om,trdata)) for om in objmasks]
    # Find out which tracer particles are owned by which dendrogram node, i.e. they are directly contained
    # by the dendro node and there is no contained subnode which in turn contains the tracer particles.
    tr_owner_obj = cztrown.TracerOwnershipObjLUT(octree, objmasks, Moo_contain, trdata)
    tr_ownersets = [ParticleIDSetFromMask(trdata,cztrown.ObjTracerOwnershipMask(tr_owner_obj,iobj)) for iobj in range(nobj)]
    # Check the determined data for consistency, i.e. ownership must imply containment
    for iobj in range(nobj):
        assert tr_ownersets[iobj].issubset(tr_containsets[iobj])
    # Return two lists of sets, with each list entry corresponding to one object in order of their masks,
    # each set containing the persistent IDs of tracer particles contained/owned by the corresponding object.
    return tr_containsets, tr_ownersets

def determine_matchmatrix(hdfdata1, octree1, hdfdata2, octree2, criterion, debug=True):
    tr_containsets1, tr_ownersets1 = get_trobjects(hdfdata1, octree1, criterion)
    tr_containsets2, tr_ownersets2 = get_trobjects(hdfdata2, octree2, criterion)
    tools.fprint('Calculating cc intersection matrices...')
    Mcc_f, Mcc_r, Mcc_j = IntersectionMatrix(tr_containsets1,tr_containsets2)
    tools.fprint('Calculating co intersection matrices...')
    Mco_f, Mco_r, Mco_j = IntersectionMatrix(tr_containsets1,tr_ownersets2)
    tools.fprint('Calculating oc intersection matrices...')
    Moc_f, Moc_r, Moc_j = IntersectionMatrix(tr_ownersets1,tr_containsets2)
    tools.fprint('Calculating oo intersection matrices...')
    Moo_f, Moo_r, Moo_j = IntersectionMatrix(tr_ownersets1,tr_ownersets2)
    matrices = [Mcc_f, Mcc_r, Mcc_j, Mco_f, Mco_r, Mco_j, Moc_f, Moc_r, Moc_j, Moo_f, Moo_r, Moo_j]
    tools.fprint('Calculating particle counts...')
    N1c = ParticleCounts(tr_containsets1)
    N2c = ParticleCounts(tr_containsets2)
    N1o = ParticleCounts(tr_ownersets1)
    N2o = ParticleCounts(tr_ownersets2)
    trcounts = [N1c,N2c,N1o,N2o]
    if debug>0:
        for im,matrix in enumerate(matrices):
            print(f'#{im} | min {np.min(matrix)} | med {np.median(matrix)} | avg {np.average(matrix)} | max {np.max(matrix)} | shape {matrix.shape}')
    return matrices, trcounts


#===============================================================================
# ==== TRACER OBJECT I/O =======================================================
#===============================================================================

def get_trobject_path(hdfdata, criterion, debug=True):
    hdf_fp = tools.get_plotfile_fingerprint(hdfdata)
    crit_fp = clfind.get_criterion_fingerprint(criterion)
    prefix = trobject_basename +crit_fp +'_' +hdf_fp
    filepath = tools.get_datafile_path(hdfdata, prefix, outdir=datadir)
    if debug>0:
        print(f'TROBJECT PATH: {filepath}')
        sys.stdout.flush()
    return filepath

def prepare_trobjects(hdfdata, octree, criterion):
    ''' Compile and save a tracer object set to criterion if not existing yet '''
    filepath = get_trobject_path(hdfdata, criterion)
    # Check if the list is already in place
    if os.path.exists(filepath) and not 'test' in criterion:
        print(f'-> Tracer object set already exists: {filepath}')
        sys.stdout.flush()
    else:
        trobjsets = TracerObjectSets(hdfdata, octree, criterion)
        tools.save_data(trobjsets, filepath, force=False)
    return filepath

def get_trobjects(hdfdata, octree, criterion, save=True):
    ''' Obtain a tracer object set according to criterion; compile if necessary '''
    filepath = get_trobject_path(hdfdata, criterion)
    #
    force = False    
    if os.path.exists(filepath):
        print(f'-> Loading tracer object set ({criterion})')
        sys.stdout.flush()
        try:
            trobjsets = tools.load_data(filepath)
        except:
            print('!! Warning: Could not load tracer object set !!')
            force = True
        else:
            return trobjsets
    trobjsets = TracerObjectSets(hdfdata, octree, criterion)
    if save:
        print(f'-> Saving tracer object set ({criterion})')
        sys.stdout.flush()
        try:
            tools.save_data(trobjsets, filepath, force=force)
        except:
            print('!! Warning: Could not save tracer object set !!')
    return trobjsets


#===============================================================================
# ==== MATCHMATRIX RESULT I/O ==================================================    
#===============================================================================

def get_matchmatrix_path(hdfdata1, hdfdata2, criterion, debug=False):
    hdf1_fp = tools.get_plotfile_fingerprint(hdfdata1)
    hdf2_fp = tools.get_plotfile_fingerprint(hdfdata2)
    crit_fp = clfind.get_criterion_fingerprint(criterion)
    prefix = mmatrix_basename +crit_fp +'_' +hdf1_fp +'_' +hdf2_fp +'_'
    filepath = tools.get_datafile_path(hdfdata1, prefix, outdir=datadir)
    if debug>0:
        print(f'MATCHMATRIX PATH: {filepath}')
        sys.stdout.flush()
    return filepath

def prepare_matchmatrix(hdfdata1, octree1, hdfdata2, octree2, criterion):
    ''' Compile and save a cluster match matrix to criterion if not existing yet '''
    filepath = get_matchmatrix_path(hdfdata1, hdfdata2, criterion)
    # Check if the list is already in place
    if os.path.exists(filepath) and not 'test' in criterion:
        print(f'-> Cluster match matrix already exists: {filepath}')
        sys.stdout.flush()
    else:
        mmatrix = determine_matchmatrix(hdfdata1, octree1, hdfdata2, octree2, criterion)
        tools.save_data(mmatrix, filepath, force=False)
    return filepath

def get_matchmatrix(hdfdata1, octree1, hdfdata2, octree2, criterion, save=True):
    ''' Obtain a cluster match matrix according to criterion; compile if necessary '''
    filepath = get_matchmatrix_path(hdfdata1, hdfdata2, criterion)
    #
    force = False    
    if os.path.exists(filepath):
        print(f'-> Loading cluster match matrix ({criterion})')
        sys.stdout.flush()
        try:
            mmatrix = tools.load_data(filepath)
        except:
            print('!! Warning: Could not load cluster match matrix !!')
            force = True
        else:
            return mmatrix
    mmatrix = determine_matchmatrix(hdfdata1, octree1, hdfdata2, octree2, criterion)
    if save:
        print(f'-> Saving cluster match matrix ({criterion})')
        sys.stdout.flush()
        try:
            tools.save_data(mmatrix, filepath, force=force)
        except:
            print('!! Warning: Could not save cluster match matrix !!')
    return mmatrix


#===============================================================================
# ==== RESULTLIST I/O ==========================================================    
#===============================================================================

def matchmatrices_collection_path(filelist_sorted, criterion, debug=True):
    pf0 = filelist_sorted[0]
    simdir = os.path.dirname(pf0)
    filelist_fp = tools.get_strlist_fingerprint(filelist_sorted)
    crit_fp = clfind.get_criterion_fingerprint(criterion)
    prefix = mmatrix_basename +'collection_' +crit_fp +'_' +filelist_fp
    filepath = tools.get_collection_path(simdir, prefix, outdir=datadir)
    if debug>0:
        print(f'MATCHMATRICES COLLECTION PATH: {filepath}')
        sys.stdout.flush()
    return filepath

def compile_matchmatrices_collection(filelist_sorted, criterion):
    matrices_collection = list()
    file_even = filelist_sorted[0]
    leafs_even, oct_even = cz.LoadPlotfile(file_even)
    for jfn, fn in enumerate(filelist_sorted[1:]):
        ifile = jfn+1 # note that we start with ifile=1 (odd)
        if (ifile%2): # current file odd, prior file even
            leafs_odd, oct_odd = cz.LoadPlotfile(fn)
            tools.fprint(f'--- {leafs_even.basename()} <> {leafs_odd.basename()} ---')
            mm = get_matchmatrix(leafs_even, oct_even, leafs_odd, oct_odd, criterion)
            leafs_even.close()
        else: # current file even, prior file odd
            leafs_even, oct_even = cz.LoadPlotfile(fn)
            tools.fprint(f'--- {leafs_odd.basename()} <> {leafs_even.basename()} ---')
            mm = get_matchmatrix(leafs_odd, oct_odd, leafs_even, oct_even, criterion)
            leafs_odd.close()
        assert mm is not None
        matrices_collection.append(mm)
        mm = None
    assert len(matrices_collection)==len(filelist_sorted[1:])
    return matrices_collection

def get_matchmatrices_collection(filelist_sorted, criterion, save=True):
    ''' Obtain all match matrices according to criterion and filelist; compile if necessary '''
    filepath = matchmatrices_collection_path(filelist_sorted, criterion)
    #
    force = False    
    if os.path.exists(filepath):
        print(f'-> Loading match matrices collection ({criterion})')
        sys.stdout.flush()
        try:
            matrices_collection = tools.load_data(filepath)
        except:
            print('!! Warning: Could not load match matrices collection!!')
            force = True
        else:
            return matrices_collection
    matrices_collection = compile_matchmatrices_collection(filelist_sorted, criterion)
    if save:
        print(f'-> Saving match matrices collection ({criterion})')
        sys.stdout.flush()
        try:
            tools.save_data(matrices_collection, filepath, force=force)
        except:
            print('!! Warning: Could not save match matrices collection !!')
    return matrices_collection


#===============================================================================
# ==== TEST ====================================================================
#===============================================================================

ppath = '/home/mweis/scratch/thesisdata/corezoom-work-wbox/b.Aa1-Aa4-2zone/out'
pf1 = os.path.join(ppath, 'CZ-b_hdf5_plt_cnt_0400')
pf2 = os.path.join(ppath, 'CZ-b_hdf5_plt_cnt_0401')
pd1 = cz.PartfileFromPlotfile(pf1)
pd2 = cz.PartfileFromPlotfile(pf2)
criterion = 'COCxp30:adrnode'#:czfrag'

def TestComparator():
    #np.set_printoptions(threshold=sys.maxsize)
    #
    leafs1, oct1 = cz.LoadPlotfile(pf1)
    objm1 = clfind.get_cmasks(leafs1, oct1, criterion)
    nobj1 = len(objm1)
    Moo_contain1 = cztrown.ObjObjContainmentMatrix(objm1)
    trdata1 = cz.PartfileData(pd1)
    tr_containsets1 = [ParticleIDSetFromMask(trdata1,cztrown.ObjTracerContainmentMask(oct1,om,trdata1)) for om in objm1]
    print(f'n_contain1: {[len(contset) for contset in tr_containsets1]}')
    tr_owner_obj1 = cztrown.TracerOwnershipObjLUT(oct1, objm1, Moo_contain1, trdata1)
    tr_ownersets1 = [ParticleIDSetFromMask(trdata1,cztrown.ObjTracerOwnershipMask(tr_owner_obj1,iobj)) for iobj in range(nobj1)]
    print(f'n_owner1: {[len(ownset) for ownset in tr_ownersets1]}')
    for iobj in range(nobj1):
        assert tr_ownersets1[iobj].issubset(tr_containsets1[iobj])
    #
    leafs2, oct2 = cz.LoadPlotfile(pf2)
    objm2 = clfind.get_cmasks(leafs2, oct2, criterion)
    nobj2 = len(objm2)
    Moo_contain2 = cztrown.ObjObjContainmentMatrix(objm2)
    trdata2 = cz.PartfileData(pd2)
    tr_containsets2 = [ParticleIDSetFromMask(trdata2,cztrown.ObjTracerContainmentMask(oct2,om,trdata2)) for om in objm2]
    print(f'n_contain2: {[len(contset) for contset in tr_containsets2]}')
    tr_owner_obj2 = cztrown.TracerOwnershipObjLUT(oct2, objm2, Moo_contain2, trdata2)
    tr_ownersets2 = [ParticleIDSetFromMask(trdata2,cztrown.ObjTracerOwnershipMask(tr_owner_obj2,iobj)) for iobj in range(nobj2)]
    print(f'n_owner2: {[len(ownset) for ownset in tr_ownersets2]}')
    for iobj in range(nobj2):
        assert tr_ownersets2[iobj].issubset(tr_containsets2[iobj])
    #
    Mcc_f, Mcc_r, Mcc_j = IntersectionMatrix(tr_containsets1,tr_containsets2)
    Mco_f, Mco_r, Mco_j = IntersectionMatrix(tr_containsets1,tr_ownersets2)
    Moc_f, Moc_r, Moc_j = IntersectionMatrix(tr_ownersets1,tr_containsets2)
    Moo_f, Moo_r, Moo_j = IntersectionMatrix(tr_ownersets1,tr_ownersets2)

    matrices = [Mcc_f, Mcc_r, Mcc_j, Mco_f, Mco_r, Mco_j, Moc_f, Moc_r, Moc_j, Moo_f, Moo_r, Moo_j]
    for im,matrix in enumerate(matrices):
        print(f'#{im} | min {np.min(matrix)} | med {np.median(matrix)} | avg {np.average(matrix)} | max {np.max(matrix)} | shape {matrix.shape}')


#===============================================================================
# ==== LOBBY ===================================================================
#===============================================================================

if __name__ == '__main__':
    criterion = sys.argv[1]
    print('>>> COMPILING MMC <<<')
    dirs = sys.argv[2:]
    if len(dirs)>1:
        print('CAUTION: Please make sure all given directories belong to the same simulation!')
    fprint('Collecting plotfiles...')
    plotfiles = CollectPlotfiles(dirs)
    fprint('Sorting plotfiles...')
    plotfiles_sorted = OrderPlotfiles(plotfiles)
    compile_matchmatrices_collection(plotfiles_sorted, criterion)

    print('FINISHED!')
