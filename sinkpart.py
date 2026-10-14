import sys,os
import fnmatch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib
import matplotlib.cm as mpcm
import re
from libmf.rpnparser import rpn_program

#===============================================================================
# ==== CONSTANTS ===============================================================
#===============================================================================

import constants
time_conv = constants.Myr/1e+3
mass_conv = constants.M_sol
mdot_conv = mass_conv/time_conv
angmom_conv = 1.
MSun = constants.M_sol
LSun = constants.L_sol
G = constants.G_g


#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

testfile = './output-sinks_evolES.dat'

angmom_ulabel = '$\mathrm{cgs}$'

var_settings = {
    'time': {'objtext':'Sim.', 'utext':'Time', 'unit':time_conv, 'ulabel':'$\mathrm{kyr}$', 'logscale':False},
    'sinkage': {'objtext':'Sink', 'utext':'Age', 'unit':time_conv, 'ulabel':'$\mathrm{kyr}$', 'logscale':True},
    'mass': {'objtext':'Sink', 'utext':'Mass', 'unit':mass_conv, 'ulabel':'$M_\odot$'},
    'starmass': {'objtext':'Star', 'utext':'Mass', 'unit':mass_conv, 'ulabel':'$M_\odot$'},
    'ofmass': {'objtext':'Cumulative Outflow', 'utext':'Mass', 'unit':mass_conv, 'ulabel':'$M_\odot$'},
    'iadmass': {'objtext':'IAD', 'utext':'Mass', 'unit':mass_conv, 'ulabel':'$M_\odot$'},
    'oadmass': {'objtext':'OAD', 'utext':'Mass', 'unit':mass_conv, 'ulabel':'$M_\odot$'},
    'windmass': {'objtext':'Wind', 'utext':'Mass', 'unit':mass_conv, 'ulabel':'$M_\odot$'},
    'M_ratio': {'objtext':'Star', 'utext':'Mass Fraction', 'unit':1., 'ulabel':'/', 'logscale':True},
    'subrad': {'objtext':'Star', 'utext':'Radius', 'unit':1e+11, 'ulabel':'$\mathrm{Gm}$'},
    'mdot': {'objtext':'Sink', 'utext':'Accr.', 'unit':mdot_conv, 'ulabel':'$M_\odot\,\mathrm{kyr}^{-1}$'},
    'starmdot': {'objtext':'Star', 'utext':'Accr.', 'unit':mdot_conv, 'ulabel':'$M_\odot\,\mathrm{kyr}^{-1}$'},
    'mdot_num': {'unit':mdot_conv, 'ulabel':'$M_\odot\,\mathrm{kyr}^{-1}$'},
    'ltot': {'unit':LSun, 'ulabel':r'$L_\odot$'},
    'lint': {'unit':LSun, 'ulabel':'$L_\odot$'},
    'l_accr': {'unit':LSun, 'ulabel':'$L_\odot$'},
    'temp': {'unit':1., 'ulabel':'$\mathrm{Kelvin}$'},
    'ltemp': {'unit':1., 'ulabel':'$\mathrm{Kelvin}$'},
    'wind_warm': {'unit':(MSun/3.15576e+10), 'ulabel':'$M_\odot\,\mathrm{kyr}^{-1}$'},
    'wind_cold': {'unit':(MSun/3.15576e+10), 'ulabel':'$M_\odot\,\mathrm{kyr}^{-1}$'},
    'wind': {'unit':(MSun/3.15576e+10), 'ulabel':'$M_\odot\,\mathrm{kyr}^{-1}$'},
    'windvel': {'unit':1e5, 'ulabel':'$\mathrm{km/s}$'},
    'outflow': {'unit':(MSun/3.15576e+10), 'ulabel':'$M_\odot\,\mathrm{kyr}^{-1}$'},
    'L_breakup': {'objtext':'Stellar Breakup', 'utext':'Angular Momentum', 'unit':angmom_conv, 'ulabel':angmom_ulabel, 'logscale':True},
    'L_star': {'objtext':'Star', 'utext':'Angular Momentum', 'unit':angmom_conv, 'ulabel':angmom_ulabel},
    'L_sink': {'objtext':'Sink', 'utext':'Angular Momentum', 'unit':angmom_conv, 'ulabel':angmom_ulabel},
    'L_iad': {'objtext':'IAD', 'utext':'Angular Momentum', 'unit':angmom_conv, 'ulabel':angmom_ulabel, 'logscale':True},
    'L_oad': {'objtext':'OAD', 'utext':'Angular Momentum', 'unit':angmom_conv, 'ulabel':angmom_ulabel, 'logscale':True},
    'L_ratio': {'objtext':'Star', 'utext':'Angular Momentum Fraction', 'unit':1., 'ulabel':'/', 'logscale':True},
    'DEFAULT': {'unit':1., 'ulabel':'cgs'},
    'posx': {'objtext':'Sink', 'utext':'X', 'ulabel':'pc', 'unit':constants.pc, 'logscale':False},
    'posy': {'objtext':'Sink', 'utext':'Y', 'ulabel':'pc', 'unit':constants.pc, 'logscale':False},
    'posz': {'objtext':'Sink', 'utext':'Z', 'ulabel':'pc', 'unit':constants.pc, 'logscale':False},
}

std_cmap = mpcm.get_cmap('jet')

sinkfile_pattern = '*output-sinks_evolES.dat'
exclude_pattern = ''


#===============================================================================
# ==== Particle Database Class =================================================  
#===============================================================================

int_keys = ['part_tag',]

class SinkData:
    def __init__(self, fn):

        self.filepath = fn
        self.rawdata, self.keys = self._ReadData(fn)
        
        line_parttags = self._GetColumn('part_tag')
        self.parttags = list(set(line_parttags))
        
        partmasks = [line_parttags==tag for tag in self.parttags]     
        self.particles = [Particle(self.rawdata[M], self.keys) for M in partmasks]
        
        for p in self.particles:
            p.learn(var_part_std)

    def __len__(self):
        return len(self.particles)
        
    def __getitem__(self, key):
        if isinstance(key,int):
            idx = key
            return self.particles[idx]
        else:
            assert isinstance(key,str)
            return [p[key] for p in self.particles]
            
    def __iter__(self):
        return iter(self.particles)

    def _ReadData(self, fn):
        linelist = []
        linetime = []
        keys = []
        keyset = set()
        sim_restart = False
        f =  open(fn, "r")
        for line in f:
            strs = line.replace('] ', ']').split()
            if "[" in line: # this is a header line
                for i,item in enumerate(strs):
                    if '#' in item:
                        keys.clear() # The header of the columns might have changed
                        # The keyset is not cleared. This is on purpose, because
                        # it is intended to inform the user about all keys that
                        # exist somewhere in the entire database
                        # (anytime, not only since the last restart).
                        sim_restart = True
                        continue                    
                    tagtext = re.sub('[\[\]]', ' ', item)
                    istr, key = tagtext.split()
                    # Check if the given number in the header is consistent with
                    # actually counting the keys (this might be overly cautious):
                    assert len(keys)==int(istr) 
                    keys.append(key)
                keyset |= set(keys)
            else: # this is a data line
                if not keys:
                    raise IOError('Trying to read lines without header')
                dataline = {}
                for counter, item in enumerate(strs):
                    key = keys[counter]
                    if key in int_keys:
                        dataline[key] = int(item)
                    else:
                        try:
                            dataline[key] = float(item)
                        except ValueError:
                            try:
                                dataline[key] = float(f'{item[:-4]}E{item[-4:]}')
                            except ValueError:
                                raise

                time = dataline['time']

                if sim_restart:
                    iReplace = np.searchsorted(linetime,time)
                    del linetime[iReplace:]
                    del linelist[iReplace:]
                    sim_restart = False

                linetime.append(time)
                linelist.append(dataline)

        return np.array(linelist), list(keyset)
    
    def _GetColumn(self, key):
        if not key in self.keys:
            raise KeyError('SinkData._GetColumn: Unknown key %s'%key)
        data = np.array([d.get(key, np.nan) for d in self.rawdata])
        return data
        
    def GetTimeData(self, time, varkey):
        sptdata = [sp.tget(time,varkey) for sp in self]
        return np.array(sptdata)
        
class Particle:
    def __init__(self, linearr, dkeys):
        self.keys = dkeys
        self.datadict = {}
        for key in self.keys:
            self.datadict[key] = np.array([d.get(key, np.nan) for d in linearr])
        self.__formulae = dict()
        
        if not 'time' in self.datadict:
            raise RuntimeError('Time data missing in sink particle')
            
        self.time = self.datadict['time']
        # Attention: This assumes that a sink particle is only
        # created and destroyed once per tag.
        # It also assumes that the SinkData._ReadData function guarantees
        # the particle data arrays to be already t-ordered.
        # This is not necessarily the case for the underlying datafile,
        # as data (e.g. from restarts) is always appended to the file.
        self.tmin = self.time[0]
        self.tmax = self.time[-1]
        # Get a minimum stepsize. This is used as an additional margin
        # when checking if the time from e.g. a plotfile is inside the
        # window of existence of the particle, because the time from
        # the sinkfile and from another source are not guaranteed to match
        # precisely (e.g. due to differing numerical precision)
        tdiff = self.time[1:]-self.time[:-1]
        self.dtmin = np.min(tdiff[tdiff>0.])
        self.tmin -= .5*self.dtmin
        self.tmax += .5*self.dtmin

    def __contains__(self, key_in):
        result = False
        result |= key_in in self.keys
        result |= key_in in self.__formulae
        return result

    def __getitem__(self, key):
        if key in self.keys:
            return self.datadict[key]
        elif key in self.__formulae:
            return self.__formulae[key].eval()
        else:
            return self.__formulae['MISSING'].eval()
            #raise KeyError('Particle: Unknown key %s'%key)

    def learnvar(self, key, rpn_form, force=True):
        self.__formulae[key] = rpn_program(rpn_form, str_eval=self.__getitem__)            
    def learn(self, vardict, force=True):
        for key in vardict:
            self.learnvar(key, vardict[key], force=force)
            
    def tget(self, obstime, varkey):
        if not varkey in self:
            raise KeyError('Particle: Unknown key %s'%varkey)        
        if (obstime < self.tmin) or (obstime > self.tmax):
            return np.nan*obstime
            # *obstime is a hack to put the return in the same format as obstime
        else:
            I = np.searchsorted(self.time, obstime-.5*self.dtmin)
            return self.__getitem__(varkey)[I]
            

# ==============================================================================
# ==== RPN Formulae for some sink vars. ========================================
# ==============================================================================

def diff(a, extrapolate=False):
    da = np.full_like(a, np.nan)
    da[1:-1] = a[2:]-a[:-2]
    return da
    
def cs(a):
    an = np.nan_to_num(a)
    return np.cumsum(an)
    
sq = np.square
sqrt = np.sqrt
nan = lambda a: np.full_like(a, np.nan)
rand = .5+.5*np.random.rand()

var_part_std = {
    'mdot_num': ('mass', diff, 'time', diff, '/'),
    'mdot': ('mdot(wo_wind)', ),
    'ofmass': ('outflow', 'dt', '*', cs),
    'starmdot': ('starmdot(wo_wind)', ),
    'l_accr': ('ltot', 'lint', '-'),
    'vel': ('velx', sq, 'vely', sq, 'velz', sq, '+', '+', sqrt),
    'posrad': ('subrad', 1., '+'),
    'temp': ('ltot', 7.125562866e-4, '/', 'posrad', sq, '/', sqrt, sqrt),
    'iadmass': ('mass', 'starmass', '-'),
    'M_ratio': ('starmass', 'mass', '/'),
    
    'dt': ('time', diff),

    'wwa': ('ltot', (1e+5*LSun), '/', np.log10, 2.194, '*'),
    'wwb': ('mass', (30.*MSun), '/', np.log10, -1.313, '*'),
    'wwc': ('temp', 40000., '/', np.log10, 0.933, '*'),
    'wwd': ('temp', 40000., '/', np.log10, sq, -10.92, '*'),
    'wwx': ((2.6/2.), np.log10, -1.226, '*'),
    'wwexp': (-6.697, 'wwa', '+', 'wwb', '+', 'wwc', '+', 'wwd', '+', 'wwx', '+', ),
    'wind_warm': (10., 'wwexp', '**', (MSun/3.15576e+7), '*'),

    'wca': ('ltot', (1e+5*LSun), '/', np.log10, 2.210, '*'),
    'wcb': ('mass', (30.*MSun), '/', np.log10, -1.339, '*'),
    'wcc': ('temp', 20000., '/', np.log10, 1.070, '*'),
    'wcx': ((1.3/2.), np.log10, -1.601, '*'),
    'wcexp': (-6.688, 'wca', '+', 'wcb', '+', 'wcc', '+', 'wcx', '+', ),
    'wind_cold': (10., 'wcexp', '**', (MSun/3.15576e+7), '*'),

    'tcold':    ('temp', 25000., '<'),
    'twarm':    (1., 'tcold', '-'),
    'wind':     ('wind_cold', 'tcold', '*', 'wind_warm', 'twarm', '*', '+'),
    'L_breakup_pms': (6.67430e-8, 'starmass', '*', 'subrad', '*', sqrt, 'starmass', '*'),
    'L_breakup_wolff': ('starmass', .2*MSun, '/', .3, '**', 'starmass', '*', 3e+17, '*'),
    'L_breakup': ('L_breakup_pms', 0., '<=', 'L_breakup_wolff', '*', 'L_breakup_pms', '+'),
    'L_star': ('starangx', sq, 'starangy', sq, 'starangz', sq, '+', '+', sqrt),
    'L_sink': ('anglx', sq, 'angly', sq, 'anglz', sq, '+', '+', sqrt),
    'L_iad': ('iadangx', sq, 'iadangy', sq, 'iadangz', sq, '+', '+', sqrt),
    'L_oad': ('oadangx', sq, 'oadangy', sq, 'oadangz', sq, '+', '+', sqrt),
    'L_ratio': ('L_star', 'L_sink', '/'),
    'sinkage': ('time', 'ptime', '-'),
    #'L_iad': ('anglx', 'starangx', '-', sq, 'angly', 'starangy', '-', sq, 'anglz', 'starangz', '-', sq, '+', '+', sqrt),
    
    #(star_lint /5.67037e-5 /4.0 /3.14159 /star_radius**2)**(0.25)
    'lint_temp': ('lint', 5.67037e-5*4.0*3.14159, 'subrad', sq, '*', '/', sqrt, sqrt),
    'ltot_temp': ('ltot', 5.67037e-5*4.0*3.14159, 'subrad', sq, '*', '/', sqrt, sqrt),
    'acc_heat': ('ltot_temp', 'lint_temp', '-'),
    'wind_active': ('lint_temp', 12500., '>', 'starmass', 8.*MSun, '>', '*', rand, '*'),
    'wind_active_old': ('temp', 12500., '>', 'starmass', 8.*MSun, '>', '*', rand, '*'),

#    'wind_active': ('lint_temp', 12500., '>'),
    
    'MISSING': ('time', nan),
}


#===============================================================================
# ==== MAIN ====================================================================  
#===============================================================================

def datascale(sink, key):
    vs = var_settings.get(key, var_settings['DEFAULT'])
    print(vs)
    logscale = vs.get('logscale', None)
    minval = np.nanmin(pldata(sink, key))
    print(logscale, logscale is None)
    if logscale is None:
        logscale = (minval > 0.)
    scale = 'linear' if (not logscale) else 'log'
    print(key, scale, minval, logscale, not logscale)
    return scale        

def datalabel(key):
    vs = var_settings.get(key, var_settings['DEFAULT'])
    #varsym, varlabel = vs['varsym'], vs['varlabel']
    utext = vs.get('utext', key)
    ulabel = vs['ulabel']
    datalabel = f"{utext} [{ulabel}]"
    return datalabel
    
def objlabel(key):
    vs = var_settings.get(key, var_settings['DEFAULT'])
    objlabel = vs.get('objtext', key)
    return objlabel

def pldata(part, key):
    data_cgs = part[key]
    vs = var_settings.get(key, var_settings['DEFAULT'])
    data = data_cgs/vs['unit']
    return data
        
def axPlotSinkData(ax, sink, xkey, ykey, **kwargs):
    pd = lambda key: pldata(sink, key)
    xdata, data = pd(xkey), pd(ykey)
    ax.plot(xdata, data, **kwargs)
    print(f'{xkey}, Min.:{np.nanmin(xdata)} Max.:{np.nanmax(xdata)}')
    print(f'{ykey}, Min.:{np.nanmin(data)} Max.:{np.nanmax(data)}')

def main(files, xkey, varkeys):
    plargs = dict()
    dlabels = [datalabel(key) for key in varkeys]
    axlabel = dlabels[0] if np.all([dlabels[0]==dl for dl in dlabels]) else None
    
    xdlabel = datalabel(xkey)
    xobjlabel = objlabel(xkey)

    merge_var_plots = (len(varkeys) > 1) and not (len(varkeys) > 4) and axlabel is not None
    if merge_var_plots:
        figs_axs = [plt.subplots(1,1), ]
    else:
        figs_axs = [plt.subplots(1,1) for k in varkeys]
        
    for i,filepath in enumerate(files):
        dirname = os.path.dirname(filepath)
        simname = os.path.basename(dirname)
        sinks = SinkData(filepath)
        print(dirname)
        if not sinks:
            print(f'No sinks found in {simname}. Skipping.')
            continue

        for k,ykey in enumerate(varkeys):
            fig, ax = figs_axs[(not merge_var_plots)*k]
            plargs['ls'] = ['-', '--', ':', '-.'][merge_var_plots*k]    
            plargs['alpha'] = 1./(len(files)*len(sinks))**.4

            for si,sink in enumerate(sinks):
                if len(files) > 1:
                    plargs['color'] = std_cmap((1.*i)/(len(files)-.99))
                    plargs['label'] = '_' if (si+k) else simname 
                elif len(sinks) > 1:
                    plargs['color'] = std_cmap((1.*si)/(len(sinks)-.99))
                    plargs['label'] = f'#{si+1}'
                    plargs['ls'] = ('-',':','--','-.')[si%min(4,1+len(sinks)//5)]
                    ax.set_title(simname)
                else:
                    plargs['label'] = '_'
                    plargs['color'] = 'black'
                    print(simname)
                    ax.set_title(simname)
                axPlotSinkData(ax, sink, xkey, ykey, **plargs)

            xscale = datascale(sink, xkey)
            yscale = datascale(sink, ykey)

            objl = objlabel(ykey)
            ax.set(xlabel=f'{xobjlabel} {xdlabel}', xscale=xscale)
            ax.set(ylabel=f'{objl} {dlabels[k]}', yscale=yscale)
            ax.legend()
                
    if merge_var_plots:
        for k,ykey in enumerate(varkeys):
            objl = objlabel(ykey)
            plargs['ls'] = ['-', '--', ':', '-.'][k]    
            plargs['label'] = objl
            plargs['color'] = 'black'
            #ax.set(xlabel=f'{xobjlabel} {xdlabel}', xscale=xscale)
            ax.set(ylabel=f'{dlabels[k]}', yscale=yscale)
            ax.plot([np.nan, np.nan], [np.nan, np.nan], **plargs)
            ax.legend()

    plt.show()


#===============================================================================
# ==== LOBBY ====================================================================
#===============================================================================

def getarg(i, default):
    try:
        return sys.argv[i]
    except:
        return default

if __name__ == '__main__':
    path = sys.argv[1]
    xkey = sys.argv[2]
    varkeys = sys.argv[3:]
    if len(varkeys)==0:
        raise RuntimeError('No varkey arguments given.')
    if os.path.isfile(path):
        files = [path,]
    elif os.path.isdir(path):
        files = list()
        for dirpath, dirnames, dirfiles in os.walk(path, followlinks=True):
            if 'out' in os.path.basename(dirpath):
                print(f'Ignoring path: {dirpath}')
                continue
            if 'old' in os.path.basename(dirpath):
                if not 'old' in path:
                    print(f'Ignoring path: {dirpath}')
                    continue
            filenames_include = set(fnmatch.filter(dirfiles, sinkfile_pattern))
            filenames_exclude = set(fnmatch.filter(dirfiles, exclude_pattern))
            filenames = list(filenames_include-filenames_exclude)
            for fn in filenames:
                filepath = os.path.join(dirpath, fn)
                files.append(filepath)
    else:
        raise IOError(f'Not a file/directory: {path}')
    main(files, xkey, varkeys)
    plt.show()
    


