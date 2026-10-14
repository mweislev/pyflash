# -*- coding: utf-8 -*-
import fnmatch,os,sys
import pickle
import gzip
import time
import numpy as np
import matplotlib.pylab as plt
from collections import OrderedDict
from ast import literal_eval
import csv
import pickle
import subprocess

from py27hash.hash import hash27

__author__ = "Michael Weis"
__version__ = "1.0.0.0"

#===============================================================================
# ==== CONSTANTS ===============================================================
#===============================================================================

datadir = 'cldata'

homepath = os.path.expanduser("~")
wall = str(int(time.time()))

Myr = 3.15576e+13


#===============================================================================
# ==== HELPER ==================================================================
#===============================================================================

def deepmap(func, ll):
   return list(map(lambda l: list(map(func,l)), ll))

def lmap(*args, **kwargs):
    return list(map(*args,**kwargs))

def lzip(*args, **kwargs):
    return list(zip(*args,**kwargs))

def select(cluster):
    return tuple(map(np.array, list(zip(*cluster))))

def modclip(a, a_min, a_max):
    return a_min + np.mod(a-a_min, a_max-a_min)

def tstring(hdfdata):
    time = hdfdata['real scalars/time'] / Myr
    return r'$t=$'+'%.2f'%time+r'$\,$Myr'

def fprint(*args, **kwargs):
    ret = print(*args, **kwargs)
    sys.stdout.flush()
    return ret

def flushprint(*args):
    for arg in args:
        print(arg, end=' ')
    print()    
    sys.stdout.flush()

def verboseprint(verbose, *args):
    if verbose:
        flushprint(*args)

def notify(message):
    return subprocess.Popen(['notify-send', message])

def timestr():
    import time
    t = time.gmtime()
    tstr  = "%04d%02d%02d"%(t.tm_year,t.tm_mon,t.tm_mday)+'-'
    tstr += "%02d%02d%02d"%(t.tm_hour,t.tm_min,t.tm_sec)+'-'
    tstr += 'R'+str(np.random.randint(100))
    return tstr
    
def datestr():
    from datetime import date
    today = date.today()
    dstr = today.strftime("%Y-%m-%d")
    return dstr


#===============================================================================
# ==== PARAMETER FILE PARSER ===================================================
#===============================================================================

def safe_eval(value):
    value_dict = {'.true.': True, '.false.': False}
    if value in value_dict:
        return value_dict[value]
    try:
        # known to crash on '.true.', '.false.', may go wrong in other cases.
        return literal_eval(value)
    except:
        # no clue how to evaluate (e.g. string like 'periodic'). Leave it as is.
        return value
        
def readpar(filename):
    par_dict = {'plot_vars': set()}
    with open(filename, 'r') as infile:
        for line in infile:
            # cut line prior to first # :
            line = line.rsplit('#')[0]
            # omit line if not containing = :
            if '=' not in line:
                continue
            # Transfer key and value from line to dictionary:
            key, foo, value = [word.strip() for word in line.rpartition('=')]
            if key.startswith('plot_var_'):
                par_dict['plot_vars'].add(value)
            else:
                par_dict[key] = safe_eval(value)
    return par_dict

def sim_param(hdfdata):
    pardict = dict()
    rrp = hdfdata['real runtime parameters']
    pardict.update(rrp)
    irp = hdfdata['integer runtime parameters']
    pardict.update(irp)
    lrp = hdfdata['logical runtime parameters']
    pardict.update(lrp)
    return pardict

def sim_scalars(hdfdata):
    scdict = dict()
    rsc = hdfdata['real scalars']
    scdict.update(rsc)
    isc = hdfdata['integer scalars']
    scdict.update(isc)
    lsc = hdfdata['logical scalars']
    scdict.update(lsc)
    return scdict


#===============================================================================
# ==== FINGERPRINTING ==========================================================
#===============================================================================

def get_strlist_fingerprint(lstr):
    concstr = ' '.join(lstr)
    fingerprint = format(hash27(concstr), 'x')
    return fingerprint

def get_array_fingerprint(a, fast=False):
    if fast:
        # Idea: It is very unprobable for two different fields/masks
        # to have exactly the same sum in each block, respectively
        arr = np.sum(a, axis=(1,2,3))
        fingerprint = get_array_fingerprint(arr, fast=False)
    else:
        try:
            hashable_repr = a.tostring()
        except AttributeError:
            hashable_repr = a.tobytes()
        fingerprint = format(hash27(hashable_repr), 'x')
    return fingerprint

def get_grid_fingerprint(hdfdata):
    block_bbox = hdfdata['bounding box']
    fingerprint = get_array_fingerprint(block_bbox)
    return fingerprint

def get_plotfile_fingerprint(hdfdata, fast=True):
    if fast:
        # Idea: It is very unprobable for the density field of the first block
        # to exactly identical for all low-corners,
        # so we will only check those (not the entire domain)
        arr = hdfdata['dens'][:,0,0,0]
    else:
        arr = hdfdata['dens']
    fingerprint = get_array_fingerprint(arr)
    return fingerprint


#===============================================================================
# ==== HUMAN READABLE DATA I/O =================================================
#===============================================================================

def save_csv(dictlist, outfile, force=False):
    ''' Write a list of dictionaries containing value sets into a human-readable text file '''
    # Make sure target directory exists
    filedir = os.path.dirname(outfile)
    filedir = makedir(os.path.dirname(outfile))
    # Avoid overwriting an already existing file
    if os.path.exists(outfile) and not force:
        raise IOError('Output file already exists: %s' % outfile)
    # Write csv file
    keys = list(dictlist[0].keys())
    with open(outfile, 'wb') as f:
        print(f'File (W-CSV): {outfile}')
        writer = csv.DictWriter(f, keys)
        writer.writeheader()
        writer.writerows(dictlist)
    #
    return outfile
    
def load_csv(infile):
    dictlist = list()
    with open(infile, mode='rb') as f:
        print(f'File (R-CSV): {infile}')
        reader = csv.DictReader(f)
        for row in reader:
            dictlist.append(row)
    return dictlist
    
def load_csv_cols(infile):
    dictlist = list()
    with open(infile, mode='rb') as f:
        print(f'File (R-CSV-cols): {infile}')
        reader = csv.DictReader(f)
        varkeys = reader.fieldnames
        for row in reader:
            dictlist.append(row)
    vardict = OrderedDict()
    for var in varkeys:
        vardict[var] = np.array([safe_eval(row[var]) for row in dictlist])
    return vardict


#===============================================================================
# ==== GENERALIZED FILE I/O ====================================================
#===============================================================================

def makedir(filedir):
    # Try to create datafile target directory; 
    # if this fails, the directory should already be there.
    # Otherwise something is seriously wrong!
    # (File with dirname? No write access?)
    try: 
        os.makedirs(filedir)
    except OSError:
        if not os.path.isdir(filedir):
            raise
    return filedir

def save_data(data, outfile, force=False, compress=False):
    filedir = os.path.dirname(outfile)
    filedir = makedir(os.path.dirname(outfile))
    # Create a preliminary filename to write the file (rename after write);
    # because the write-process may be incomplete due to error/interruption,
    # to avoid yielding a defective file of the desired name then.
    prefile = outfile +'.pre'
    # Avoid overwriting an already existing file
    if os.path.exists(outfile) and not force:
        raise IOError('Output file already exists: %s' % outfile)
    # Write data to preliminary file
    if compress:
        with gzip.open(prefile, 'wb') as f:
            print(f'File (WC): {outfile}')
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    else:
        with open(prefile, 'wb') as f:
            print(f'File (W): {outfile}')
            pickle.dump(data, f, pickle.HIGHEST_PROTOCOL)
    # Rename preliminary file to desired filename
    os.rename(prefile, outfile)
    return outfile

def load_data(infile):
    if not os.path.exists(infile):
        raise IOError('Input file does not exist: %s' % infile)
    try:
        with open(infile, 'rb') as f:
            print(f'File (R): {infile}')
            data = pickle.load(f)
    except:
        with gzip.open(infile, 'rb') as f:
            print(f'File (RC): {infile}')
            data = pickle.load(f)
    return data

def get_datafile_path(hdfdata, prefix, suffix='', extension='.pkl', outdir=None):
    # Build path to data file from plotfile path
    plotfile = hdfdata.filename()
    basename = os.path.basename(plotfile).replace('_hdf5', '')
    filename = prefix +basename +suffix +extension
    if outdir is None: outdir = datadir
    basedir = os.path.dirname(plotfile)
    filedir = os.path.join(basedir, outdir)
    filepath = os.path.join(filedir, filename)
    return filepath

def get_datafile_path(hdfdata, prefix, suffix='', extension='.pkl', outdir=None):
    # Build path to data file from plotfile path
    plotfile = hdfdata.filename()
    basename = os.path.basename(plotfile).replace('_hdf5', '')
    filename = prefix +basename +suffix +extension
    if outdir is None: outdir = datadir
    basedir = os.path.dirname(plotfile)
    filedir = os.path.join(basedir, outdir)
    filepath = os.path.join(filedir, filename)
    return filepath

def get_collection_path(simdir, prefix, suffix='', extension='.pkl', outdir=None):
    # Build path to data file from plotfile path
    filename = prefix +suffix +extension
    if outdir is None: outdir = datadir
    # Note that the default datadir will be a subdirectory of the simulation directory,
    # while an outdir might be an absolute path (beginning with /), and as such would
    # override the simdir argument in path.join
    filedir = os.path.join(simdir, outdir)
    filepath = os.path.join(filedir, filename)
    return filepath

def CheckFiles(filelist):
    print('>> Checking existence of files:')
    file_ok = [os.path.isfile(f) for f in filelist]
    for fname,fstatus in zip(filelist,file_ok):
        print(f'[{fstatus}] {fname}')
    if not np.all(file_ok):
        raise IOError('Missing input files')
    return file_ok


#===============================================================================
# ==== DIRECTORY TRAVERSAL =====================================================
#===============================================================================

def scanpath(simpath, pattern, exclude_pattern='*_forced_*'):
    dirdict = dict()
    if os.path.isfile(simpath):
        dirpath = os.path.dirname(simpath)
        filename = os.path.basename(simpath)
        dirdict[dirpath] = [filename,]
    elif os.path.isdir(simpath):
        for dirpath, dirnames, files in os.walk(simpath, followlinks=True):
            if datadir in dirpath: continue
            if 'particle_files' in dirpath: continue
            filenames_include = set(fnmatch.filter(files, pattern))
            filenames_exclude = set(fnmatch.filter(files, exclude_pattern))
            filenames = list(filenames_include-filenames_exclude)
            if len(filenames) == 0:
                continue
            filenames.sort()
            dirdict[dirpath] = filenames
    return dirdict


#===============================================================================
# ==== BATCH PROCESSING HELPER =================================================
#===============================================================================

def multifilearg(*arg):
    from os.path import split, join, isdir, isfile
    files = list()
    for pattern in arg:
        dirname, filepattern = (arg, '*') if isdir(arg) else split(arg)
        dirfiles = [f for f in os.listdir(dirname) if isfile(f)]
        matchfiles = fnmatch.filter(dirfiles, filepattern)
        filepathes = [join(dirname, fn) for fn in matchfiles]
        files.extend(filepathes)
    return files

def process(function, filename, verbose=False, strict=False):
    verboseprint(verbose, '\n-> Opening :', filename)
    if not strict:
        try:
            result = function(filename)
        except Exception as e:
            result = None
            flushprint('-> Error   :', e)
            flushprint('-> Skipped :', filename)
        else:
            verboseprint(verbose, '-> Success :', filename)
    else:
        result = function(filename)
        verboseprint(verbose, '-> Success :', filename)        
    return result

def batchprocess(function, filebatch, verbose=False, strict=False):
#    if hasattr(filebatch, '__iter__'):
    if isinstance(filebatch, list):
        resultlist = []
        for entry in filebatch:
            resultlist.append(batchprocess(function, entry, verbose, strict=strict))
        return resultlist
    elif isinstance(filebatch, dict):
        resultdict = {}
        for simpath, dirfiles in filebatch.items():
            filelist = [os.path.join(simpath,fn) for fn in dirfiles]
            resultdict[simpath] = batchprocess(function, filelist, verbose, strict=strict)
        return resultdict
    else:
        filename = filebatch
        return process(function, filename, verbose=verbose, strict=strict)

def countbatch(path, basename, cntrange, cntdigits=4):
    """ Scan the given directory for plotfiles in subdirectories count wise"""
    batch = []
    counts = []
    for cnt in cntrange:
        plotfiles = []
        pattern = basename+str(cnt).zfill(cntdigits)
        for dirpath, dirnames, files in os.walk(path, followlinks=True):
            simname = os.path.basename(dirpath)
            if '.' in simname:
                continue
            for filename in fnmatch.filter(files, pattern):
                if '_forced_' in filename:
                    continue
                plotfiles.append(os.path.join(dirpath, filename))
        if plotfiles:
            batch.append(plotfiles)
            counts.append(cnt)
    return counts, batch
  
def dirbatch(path, basename, cntrange, cntdigits=4):
    """ Scan the given directory for plotfiles subdirectory wise"""
    dirdict = {}
    patternlist = [basename+str(cnt).zfill(cntdigits) for cnt in cntrange]
    for dirpath, dirnames, files in os.walk(path, followlinks=True):
        simname = os.path.basename(dirpath)
        #if '.' in simname:
        #    print(f'Skipping {dirpath} (dot in name)')
        #    continue
        for pattern in patternlist:
            for filename in fnmatch.filter(files, pattern):
                if '_forced_' in filename:
                    continue
                plotfile = os.path.join(dirpath, filename)
                if not dirpath in dirdict:   
                    dirdict[dirpath] = [plotfile,]
                else:
                    dirdict[dirpath].append(plotfile)        
    return dirdict

def matrixbatch(path, basename, cntrange, cntdigits=4, skipdot=True):
    """ ??? """
    dirdict = {}
    patternlist = [basename+str(cnt).zfill(cntdigits) for cnt in cntrange]
    for dirpath, dirnames, files in os.walk(path, followlinks=True):
        simname = os.path.basename(dirpath)
        if skipdot and '.' in simname: # WANTED COMMENT: WHY THIS SKIP?
            print(f'Skipping {dirpath} ("." in dirname)')
            continue
        for ipa, pattern in enumerate(patternlist):
            for filename in fnmatch.filter(files, pattern):
                if '_forced_' in filename:
                    continue
                else:
                    if not dirpath in dirdict:   
                        dirdict[dirpath] = len(patternlist)*[None,]
                    dirdict[dirpath][ipa] = os.path.join(dirpath, filename)
                    break
    return dirdict
    
def tilebatch(path, basename, cntrange, cntdigits=4, skipdot=True):
    """ Scan the given directory for plotfiles subdirectory wise,
        either one file or NONE per subdir*count, respectively. """
    dirdict = {}
    for dirpath, subdirs, files in os.walk(path, followlinks=True):
        simname = os.path.basename(dirpath)
        if skipdot and '.' in simname:
            print(f'Skipping {dirpath} ("." in dirname)')
            continue
        basefiles = fnmatch.filter(files, basename+'?'*cntdigits)
        tilefiles = [fn for fn in basefiles if '_forced_' not in fn]
        if not len(tilefiles):
            continue
        #print len(basefiles), dirpath
        dirdict[dirpath] = len(cntrange)*[None,]
        cntpatternlist = [basename+str(cnt).zfill(cntdigits) for cnt in cntrange]
        for icnt, cntpattern in enumerate(cntpatternlist):
            tile_applicants = fnmatch.filter(tilefiles, cntpattern)
            if len(tile_applicants):
                dirdict[dirpath][icnt] = tile_applicants[0]                
    return dirdict
    
