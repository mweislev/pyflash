# -*- coding: utf-8 -*-
import subprocess, sys, fnmatch, os, re

__author__ = "Michael Weis"
__version__ = '0.3.0.0'

# ==== SETTINGS ================================================================
patternbase = '*.png'
FFMPEG_PATH = 'ffmpeg'
THREADS = 8
FRAMERATE = 6#15#30#15#6#24

discard_str = ['_hdf5', '_cnt']

# ==== HELPER ==================================================================
def trycreate(dirname):
    try: 
        os.makedirs(dirname)
    except OSError:
        if not os.path.isdir(dirname):
            raise

# ==== MAIN ====================================================================
def scandir(path, pattern, cntdigits=4):
    ffmpeg_filelist = {}
    ffmpeg_cntpattern = '%?%?%?%?'
    for dirpath, subdirs, files in os.walk(path):
        if not files: continue
        if 'automov' in dirpath: continue
        ffmpeg_patterns = list()
        for f in files:
            numbers = re.findall(r'\d+',f)
            if not numbers: continue
            cnts = numbers[-1]
            f_ = f.split(cnts)
            ffmpeg_patterns.append(''.join(f_[:-1])+ffmpeg_cntpattern+f_[-1])
        #filenames, fileexts = zip(*[os.path.splitext(f) for f in files])
        #filenamecores = [fn[:-cntdigits] for fn in filenames]
        #ffmpeg_patterns = [Z[0]+ffmpeg_cntpattern+Z[1] for Z in zip(filenamecores, fileexts)]
        dir_ffmpeg_patterns = set(fnmatch.filter(ffmpeg_patterns, pattern))
        if len(dir_ffmpeg_patterns)>0:
            if '_raw' in dirpath: continue
            ffmpeg_filelist[dirpath] = dir_ffmpeg_patterns
    return ffmpeg_filelist

def encode(datadir, ffmpeg_filelist, cntdigits=4):
    outpath = os.path.join(datadir, 'automov')
    if ffmpeg_filelist: trycreate(outpath)
    for dirpath in ffmpeg_filelist:
        for ffmpeg_pattern in ffmpeg_filelist[dirpath]:
            ffmpeg_in = os.path.join(dirpath, ffmpeg_pattern)

            subpath = dirpath.replace(datadir,'')
            sbpname = subpath.replace('/','_').replace(r'\\','_') +'_'
            outfile = sbpname+os.path.splitext(ffmpeg_pattern)[0][:-(2*cntdigits+1)]
            for s in discard_str:
                outfile = outfile.replace(s,'')
            ffmpeg_out = os.path.join(outpath, outfile+'.mp4')            
            if os.path.exists(ffmpeg_out):
                print(f'Skipping {ffmpeg_out}')
                continue
            command  = [FFMPEG_PATH, '-y']
            command += ['-framerate', str(FRAMERATE),]
            command += ['-i', ffmpeg_in,]
            command += ['-c:v', 'libx264',]
            command += ['-preset', 'slow']
            command += ['-crf', '18']
            command += ['-tune', 'animation']
            command += ['-pix_fmt', 'yuv420p']
            command += ['-vf', 'scale=trunc(iw/2)*2:trunc(ih/2)*2',]
            command += ['-threads', str(THREADS),]
            command += [ffmpeg_out,]
            print(command)
            subprocess.call(command)

def autovid(datadir,cntdigits):
    filedict = scandir(datadir, patternbase, cntdigits)
    encode(datadir, filedict)


# ==== ENTRANCE ================================================================
if __name__=='__main__':
    datadir = sys.argv[1]
    try:
        cntdigits = sys.argv[2]
    except:
        cntdigits = 4
    autovid(datadir,cntdigits)

# ==== FINISH LINE =============================================================
print('FINISHED!')

