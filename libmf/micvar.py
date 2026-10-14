# -*- coding: utf-8 -*-
import h5py
import numpy as np
from .rpnparser import rpn_program
import sys, os
from functools import reduce

__author__ = "Michael Weis"
__version__ = "1.0.0.1"


# ==============================================================================
# ==== CLASS: flashfile data reader ============================================
# ==============================================================================

class flashfile(object):
    def __init__(self, filename):
        self.__filename = filename
        self.__plotfile = h5py.File(self.__filename, 'r')
        self.__closed = False
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.close()
        return False
    def __del__(self):
        self.close()
    def __repr__(self, level=0):
        answer  = 'HDF5 Flash Datafile' + '\n'
        answer += self.__filename+'\n'
        return answer        
    def read_raw(self, path):
        if self.__closed:
            raise IOError('Trying to read_raw from closed flashfile')
        route = [subdir for subdir in path.split('/') if subdir!='']
        dset = self.__plotfile
        for step in route:
            if not hasattr(dset, 'keys'):
                try:
                    dset_dict = {key.decode().rstrip():value for key,value in dset}
                    dset = dset_dict
                except AttributeError:
                    dset_dict = {key.rstrip():value for key,value in dset}
                    dset = dset_dict
                except ValueError:
                    return dset
            keys = list(dset.keys())
            try:
                stepkey = [k for k in keys if step.rstrip()==k.rstrip()][0]
            except IndexError:
                raise KeyError(f'Key {step} not found on path {path}')
            else:
                dset = dset[stepkey]
        return dset

    def read(self, path, dtype=None):
        if self.__closed:
            raise IOError('Trying to read from closed flashfile')
        data = self.read_raw(path)
        dtype_in = np.dtype(data.dtype)
        if dtype is None:
            if np.issubdtype(dtype_in, np.floating):
                return np.array(data, dtype=np.float64)
            elif np.issubdtype(dtype_in, np.complexfloating):
                return np.array(data, dtype=np.cdouble)
            elif np.issubdtype(dtype_in, np.unsignedinteger):
                return np.array(data, dtype=np.uint64)
            elif np.issubdtype(dtype_in, np.signedinteger):
                return np.array(data, dtype=np.int64)
            elif np.issubdtype(dtype_in, np.number):
                return np.array(data)
            elif np.issubdtype(dtype_in, np.dtype(str).type):
                assert np.all(np.array(map(len,data)) == 1)
                strlist = [item[0] for item in data]
                return strlist
            elif np.issubdtype(dtype_in, np.void):
                dsdict = dict()
                if 'name' in data.dtype.names and 'value' in data.dtype.names:
                    for item in data:
                        subkey = item['name'].rstrip().decode()
                        value = item['value']
                        val = value.rstrip() if isinstance(value, str) else value
                        dsdict[subkey] = val
                else:
                    for subkey in data.dtype.names:
                        assert len(data[subkey]) == 1
                        value = data[subkey][0]
                        val = value.rstrip() if isinstance(value, str) else value
                        dsdict[subkey] = val
                return dsdict
            elif np.issubdtype(dtype_in, np.bytes_):
                strarr_raw = np.array(data, dtype=np.bytes_)
                stripfunc = lambda s: s.strip()
                arrstrip = np.vectorize(stripfunc, [np.str_])
                strarr = arrstrip(np.squeeze(strarr_raw))
                return strarr
            else:
                raise NotImplementedError(f'Datatype {data.dtype} unknown')
        else:
            return np.array(data, dtype=dtype)
                
    def list(self, path=None):
        if path is None:
            path = ''
        dataset = self.read_raw(path)
        return [key.rstrip() for key in dataset]
    def basename(self):
        bn = os.path.basename(self.__filename)
        return bn
    def filename(self):
        fp = os.path.normpath(self.__filename)
        return fp
    def dirname(self):
        dn = os.path.dirname(os.path.abspath(self.__filename))
        return dn
    def basedir(self):
        dn = os.path.dirname(os.path.abspath(self.__filename))
        bd = os.path.basename(dn.removesuffix('/out'))
        return bd
    def close(self):
        try:
            if not self.__closed:
                try:
                    self.__plotfile.close()
                except IOError:
                    pass
                finally:
                    self.__closed = True
        except AttributeError:
            pass


# ==============================================================================
# ==== CLASS: memorizing plotfile evaluator ====================================
# ==============================================================================

class plotfile(flashfile):
    def __init__(self, filename, leafdata=True, memorize=True):
        flashfile.__init__(self, filename)
        nodetype = np.array(flashfile.read(self, 'node type'))
        self.__nodeselect = (nodetype==1) if leafdata else (nodetype==nodetype)
        self.__nodecount = len(self.__nodeselect)
        self.__blockcount = np.sum(self.__nodeselect)
        self.__memodict = dict()
        self.__formulae = dict()
        self.__memorize = memorize
        self.cache('blid', self.blockid())
        self.cache('ndid', self.nodeid())
        self.shape = self.blockshape()
    def __enter__(self):
        flashfile.__enter__(self)
        return self
    def __exit__(self, *exc):
        self.close()
        return False
    def __del__(self):
        self.close()
    def __repr__(self, level=0):
        return flashfile.__repr__(self, level)
    def read(self, key):
        """ Remove inner nodes (if applicable) prior to returning flashfile data """
        data = flashfile.read(self, key)
        if isinstance(data, np.ndarray) and np.any(self.__nodeselect):
            if data.ndim:
                if data.shape[0] == self.__nodecount:
                    return data[self.__nodeselect]
        return data
    def flush(self, key=None):
        if key is None:
            self.__memodict.clear()
            return None
        else:
            return self.__memodict.pop(key, None)
    def iscached(self, key):
        return key in self.__memodict
    def cache(self, key, dataset, force=True):
        if self.__memorize or force:
            self.__memodict[key] = dataset
        return dataset
    def cache_fields(self, fields):
        for key in fields:
            self.cache(key, fields[key], force=True)
        return list(fields.keys())
    def __getitem__(self, key_in):
        #print('getitem %s'%key_in)
        #sys.stdout.flush()
        # Remove plot setting specifier (indicated by '::') from key:
        key = key_in.split('::')[0]
        # 1. Look up, if key is memorized in cache dictionary
        if key in self.__memodict:
            #print('* memodict')
            return self.__memodict[key]
        # 2. Try to obtain data from formula
        if key in self.__formulae:
            #print('* formula')
            #print(self.__formulae[key])
            ret = self.__formulae[key].eval()
            return self.cache(key, ret, force=False)
        # 3. Try to obtain data from file
        try:
            #print('* plotfile')
            dataset = plotfile.read(self, key)
        except KeyError:
            raise KeyError(f'Plotfile: Unknown variable {key}')
        else:
            return self.cache(key, dataset, force=False)
    def __contains__(self, key_in):
        key = key_in.split('::')[0]
        result = False
        result |= key in self.list()
        result |= key in self.__memodict
        result |= key in self.__formulae
        return result
    def blockshape(self, fulltree=None):
        if fulltree is None:
            nblocks = self.__blockcount
        elif fulltree:
            nblocks = self.__nodecount
        else:
            nodetype = np.array(flashfile.read(self, 'node type'))
            nblocks = np.sum(nodetype==1)
            if nblocks < self.__blockcount:
                print('WARNING: MICVAR: Leaf block shape requested, but reader NOT in leaf-only mode.')
        nzb = (self.__getitem__('integer scalars/nzb')).item()
        nyb = (self.__getitem__('integer scalars/nyb')).item()
        nxb = (self.__getitem__('integer scalars/nxb')).item()
        blshape = (nblocks, nzb, nyb, nxb)
        return blshape
    def dshape(self):
        return self.blockshape()
    def blockid(self):
        return np.arange(self.__blockcount)
    def nodeid(self):
        nodeid = np.arange(self.__nodecount)
        return nodeid[self.__nodeselect]
    def get(self, key_in, default=None, default_shape=None):
        key = key_in.split('::')[0]
        try:
            return self.__getitem__(key)
        except KeyError:
            if default is None:
                raise
            else:
                if default_shape is None:
                    default_shape = self.blockshape()
                dataset = np.full(default_shape, default)
                return self.cache(key, dataset, force=False)
    def nblocks(self, leafonly=False):
        if leafonly:
            return self.__blockcount
        else:
            return self.__nodecount
    def learnvar(self, key, rpn_form, force=True):
        if key in self.list():
            if force:
                print(f'WARNING: Overriding plotfile variable {key} with formula!')
                self.__formulae[key] = rpn_program(rpn_form, str_eval=self.__getitem__)
            else:
                print(f'WARNING: Ignoring formula {key} (conflicting plotfile variable)!')
        else:
            self.__formulae[key] = rpn_program(rpn_form, str_eval=self.__getitem__)
    def learn(self, vardict, force=True):
        for key in vardict:
            self.learnvar(key, vardict[key], force=force)
    def close(self):
        flashfile.close(self)
        self.__memodict.clear()
        self.__formulae.clear()


# ==============================================================================
# ==== RPN Formulae for some ubiquitous mhd./chem. vars. =======================
# ==============================================================================

def enum(a):
    dshape = a.shape
    ones = np.ones(dshape, dtype=int)
    idx = np.cumsum(ones).reshape(dshape)
    return idx


k_b = 1.38064852e-16    # Boltzmann constant (erg/K)
st_b = 5.670374419e-5   # Stefan-Boltzmann constant (erg/K)
m_a = 1.660539040e-24     # atomic mass unit (g)/(Da)
G_g = 6.67408e-8    # Gravitational constant (cgs)
sq = np.square
lg = np.log10

#sclip = lambda x: np.clip(x, -1, 1)
var_grid = {
    'zeros': ('dens', np.zeros_like, ),
    'ones': ('dens', np.ones_like, ),
    'cellID': ('dens', enum),
    'intones': ('dens', lambda a: np.ones_like(a, dtype=int), ),
    'nxb': ('integer scalars/nxb', ),
    'nyb': ('integer scalars/nyb', ),
    'nzb': ('integer scalars/nzb', ),
    'vol': ('block size', lambda a: np.reshape(np.prod(a, axis=-1), (-1,1,1,1)),
        'ones', '*', 'nxb', '/', 'nyb', '/', 'nzb', '/'),
    'lenx': ('block size', lambda a: np.reshape(a[...,0], (-1,1,1,1)), 'ones', '*', 'nxb', '/'),
    'leny': ('block size', lambda a: np.reshape(a[...,1], (-1,1,1,1)), 'ones', '*', 'nyb', '/'),
    'lenz': ('block size', lambda a: np.reshape(a[...,2], (-1,1,1,1)), 'ones', '*', 'nzb', '/'),
    'dl': ('lenx','leny','lenz',6.,'*','*','/'),
    'areax': ('vol', 'lenx', '/'),
    'areay': ('vol', 'leny', '/'),
    'areaz': ('vol', 'lenz', '/'),
    'length': ('vol', 1./3., '**'),
    'mass': ('dens', 'vol', '*'),
    'parts': ('numdens', 'vol', '*'),
    'rlevel': ('refine level', lambda x: np.reshape(x, (-1,1,1,1)), 'ones', '*'),
    'denscontrast_m': ('mass', np.sum, 'dens', 'mass', '*', np.sum, '/', 'dens', '*'),
    'denscontrast_v': ('vol', np.sum, 'dens', 'vol', '*', np.sum, '/', 'dens', '*'),
    'cellcount': ('ones', ),
    'blockcount': ('cellcount', 'nxb', '/', 'nyb', '/', 'nzb', '/'),
    'nodeindex': ('ndid', lambda x: np.reshape(x, (-1,1,1,1)), 'intones', '*'),
    'blockindex': ('blid', lambda x: np.reshape(x, (-1,1,1,1)), 'intones', '*'),
    'jl_ref':   ('integer runtime parameters/ncell_ref_jeans', ),
    'jl_deref': ('integer runtime parameters/ncell_deref_jeans', ),
    'hsml': ('length', 2., '*'), # Assumed smoothing length for converting to sph
}

var_2d = {
    'mag_2d_sq': ('magx', sq, 'magy', sq, '+'),
    'magp_2d': ('mag_2d_sq', (1./(8.*np.pi)), '*'),
    'vel_2d_sq': ('velx', sq, 'vely', sq, '+'),
    'c_s_2d': ('pres', 'dens', '/', np.sqrt),    
    'mach_s_2d': ('vel_2d_sq', np.sqrt, 'c_s_2d', '/'),   
}

var_mhd = {
    'jx': ('dens', 'velx', '*'),
    'jy': ('dens', 'vely', '*'),
    'jz': ('dens', 'velz', '*'),
    'numdens': ('pres', 'temp', k_b, '*', '/'), # Particle number density
    'mu': ('dens', 'numdens', '/'), # Mean particle mass
    'Ai': ('mu', m_a, '/'), # Mean nucleon number
    'magpres': ('mag_sq', (1./(8.*np.pi)), '*'),
    'beta': ('pres', 'magpres', '/'),

    'mag_sq': ('magx', sq, 'magy', sq, 'magz', sq, '+', '+'),
    'magx_sq': ('magx', sq),
    'magy_sq': ('magy', sq),
    'magz_sq': ('magz', sq),
    'mag_yz_sq': ('magy_sq', 'magz_sq', '+'),
    'mag': ('mag_sq', np.sqrt),
    'mag_yz': ('mag_yz_sq', np.sqrt),
    'mag_sinphi': ('mag_yz', 'mag', '/'),
    'mag_cosphi': ('magx', 'mag', '/'),
    'magp': ('mag_sq', (1./(8.*np.pi)), '*'),

    'vel_sq': ('velx', sq, 'vely', sq, 'velz', sq, '+', '+'),
    'velx_sq': ('velx', sq),
    'vely_sq': ('vely', sq),
    'velz_sq': ('velz', sq),
    'vel_xy_sq': ('velx_sq', 'vely_sq', '+'),
    'vel_yz_sq': ('vely_sq', 'velz_sq', '+'),
    'vel': ('vel_sq', np.sqrt),
    'vel_xy': ('vel_xy_sq', np.sqrt),
    'vel_yz': ('vel_yz_sq', np.sqrt),
    'vel_sinphi': ('vel_yz', 'vel', '/'),
    'vel_cosphi': ('velx', 'vel', '/'),
    'velp': ('vel_sq', 'dens', '*', .5, '*'),
    
    'momx': ('dens', 'velx', '*'),
    'momy': ('dens', 'vely', '*'),
    'momz': ('dens', 'velz', '*'),

    'crossp': ('vely', sq, 'velz', sq, '+', 'dens', '*', .5, '*'),
    'beta_v': ('velp', 'magpres', '/'),
    'epot': ('gpot', 'gpot', np.max, '-'),
    'epotabs': ('epot', np.fabs),
    'gravp': ('gpot', 'dens', '*'),
    'gravep': ('epot', 'dens', '*'),
    'p_sum': ('magpres', 'velp', '+', 'gravp', '+', 'pres', '+'),
    'p_hdyn': ('velp', 'gravp', 'pres', '+', '+'),
    'c_s': ('pres', 'dens', '/', np.sqrt),    
    'c_a': ('mag', 4*np.pi, 'dens', '*', np.sqrt, '/'), # Alven velocity
    'c_ma': (5./3., 'pres', '*', 'dens', '/', np.sqrt),
    'mach_s': ('vel_sq', np.sqrt, 'c_s', '/'),
    'mach_a': ('vel_sq', np.sqrt, 'c_a', '/'),
    't_ff': (1., G_g, 'dens', '*', np.sqrt, '/'),
    'l_jeans': ('c_s', 't_ff', '*', np.sqrt(np.pi), '*'), # Jeans length as oscillation wavelength
    'l_jH_ratio': ('c_a','c_s','/', sq, 1., '+', np.sqrt),
    'l_jH': ('l_jH_ratio', 'l_jeans', '*'),

    't_ff_sphere': (3.*np.pi/32., G_g, 'dens', '*', '/', np.sqrt), # Kippenhahn, Weigert & Weiss, 1990
    
    'dens_jet': ('ofmi', 'dens', '*'),
    'dens_wind': ('ofwi', 'dens', '*'),

    'mass_jet': ('dens_jet', 'vol', '*'),
    'mass_wind': ('dens_wind', 'vol', '*'),

    # The factors (1./8.) come from the jeans length relating to the diameter
    # of the collapsing sphere (as opposed to the radius)
    'm_j': ('l_jeans', 3, '**', 4./(3.*8.)*np.pi, '*', 'dens', '*'), # Local jeans mass
    # Local jeans mass in magnetic field (hypothetical, not able to collapse over field lines)
    'm_jh': ('l_jeans', 'l_jH', sq, '*', 4./(3.*8.)*np.pi, '*', 'dens', '*'),
    # Effective collapse mass in magnetic field
    'm_c': ('l_jH', 3, '**', 4./(3.*8.)*np.pi, '*', 'dens', '*'),
    
    'jeans_res': ('l_jeans', 'length', '/'),
    'sel_slab': ('p_sum', 0., '<'),
    'sel_inflow': ('p_sum', 0., '>'),
    'vb_cos': ('magx', 'velx', '*', 'magy', 'vely', '*', '+',
        'magz', 'velz', '*', '+', 'mag_sq', 'vel_sq', '*', np.sqrt, '/'),
    'vb_oangle': ('vb_cos', lambda x:np.clip(x,-1,1), np.arccos, 180./np.pi, '*'),
    'vb_angle': ('vb_cos', lambda x:np.clip(np.fabs(x),0,1), np.arccos, 180./np.pi, '*'),
    
    'jref_ltarget': ('rlevel', 'length', 'l_jeans', 'jl_ref', '/', '/', np.log2, np.ceil, '+'),
    'Tj': ('irxe', np.pi, st_b, '*', '/', np.sqrt, np.sqrt),
    'rpa': ('rpax', sq, 'rpay', sq, 'rpaz', sq, '+', '+', np.sqrt),
    
    'rpg_ratio': ('rpa', 'gpa', '/'),
    'tdiff': ('temp', 'tdus', '-'),
    
    'sndspd': ('c_s',np.sqrt(5./3.), '*'),
    'dt_ltempx': ('sndspd', 'velx', np.fabs, '+', 'lenx', '/'),
    'dt_ltempy': ('sndspd', 'vely', np.fabs, '+', 'leny', '/'),
    'dt_ltempz': ('sndspd', 'velz', np.fabs, '+', 'lenz', '/'),
    'dt_ltemp': ('dt_ltempx', 'dt_ltempy', np.maximum, 'dt_ltempz', np.maximum),
    'dthydro': (.5, 'dt_ltemp', '/'),

    'mhdspd': ('sndspd', 'c_a', np.maximum),
    'dt_ltmhdx': ('mhdspd', 'velx', np.fabs, '+', 'lenx', '/'),
    'dt_ltmhdy': ('mhdspd', 'vely', np.fabs, '+', 'leny', '/'),
    'dt_ltmhdz': ('mhdspd', 'velz', np.fabs, '+', 'lenz', '/'),
    'dt_ltmhd': ('dt_ltmhdx', 'dt_ltmhdy', np.maximum, 'dt_ltmhdz', np.maximum),
    'dtmhd': (.5, 'dt_ltmhd', '/'),
    
}

ch_mu5 = {
    'Ha': 1.0, 'He': 4.002602, 'C': 12.011, 'O': 15.9994, 'Si': 28.0855,
    'H2': 2.0, 'Hp': 1.0, 'Cp': 12.011, 'CO': 28.01,}
var_ch5 = {
    'ch_abundhe': ('real runtime parameters/ch_abundhe',),
    'ch_abundc': ('real runtime parameters/ch_abundc',),
    'ch_abundo': ('real runtime parameters/ch_abundo',),
    'ch_abundsi': ('real runtime parameters/ch_abundsi',),
    'ch_abar': (
            ch_mu5['Ha'],
            ch_mu5['He'], 'ch_abundhe', '*', '+',
            ch_mu5['C'], 'ch_abundc', '*', '+',
            ch_mu5['O'], 'ch_abundo', '*', '+',
            ch_mu5['Si'], 'ch_abundsi', '*', '+',),
    'ch_ihp': ('ihp', 'ch_abar', '/'),
    'ch_iha': ('iha', 'ch_abar', '/'),
    'ch_ih2': ('ih2', 'ch_abar', '/'),
    'ch_icp': ('icp', 'ch_abar', '/'),
    'ch_ico': ('ico', ch_mu5['CO']/ch_mu5['Cp'], '*', 'ch_abar', '/'), 
    'ch_dih2': ('dih2', 'ch_abar', '/'),
    'ch_dico': ('dico', ch_mu5['CO']/ch_mu5['Cp'], '*', 'ch_abar', '/'),
    'n_hp': ('dens', 'ch_ihp', '*', m_a*ch_mu5['Hp'], '/'),
    'n_ha': ('dens', 'ch_iha', '*', m_a*ch_mu5['Ha'], '/'),
    'n_h2': ('dens', 'ch_ih2', '*', m_a*ch_mu5['H2'], '/'),
    'n_cp': ('dens', 'ch_icp', '*', m_a*ch_mu5['Cp'], '/'),
    'n_co': ('dens', 'ch_ico', '*', m_a*ch_mu5['CO'], '/'),
    'n_h1': ('n_ha', 'n_hp', '+'),
    'n_e': ('n_hp', 'n_cp', '+'),
    'n_hx': ('n_ha', 'n_hp', '+', 'n_h2', 2, '*', '+'),
    'n_cx': ('n_co', 'n_cp', '+'),
    'relrate_h2': ('ch_dih2', 'ch_iha', 'ch_ihp', '+', '/'),
    'relrate_co': ('ch_dico', 'ch_icp', '/'),
    'densrate_h2': ('relrate_h2', 'dens_ha', 'dens_hp', '+', '*'),
    'densrate_co': ('relrate_co', 'dens_cp', '*', ch_mu5['CO']/ch_mu5['Cp'], '*'),
    'nrate_h2': ('densrate_h2', m_a*ch_mu5['H2'], '/'),
    'nrate_co': ('densrate_co', m_a*ch_mu5['CO'], '/'),
    'madens': ('dens', m_a, '/', 'ch_abar', '/'), # density in units of H-atom-m/ccm
    'f_hp': ('n_hp', 'n_hx', '/'),
    'f_ha': ('n_ha', 'n_hx', '/'),
    'f_h2': ('n_h2', 'n_hx', '/'),
    'f_cp': ('n_cp', 'n_cx', '/'),
    'f_co': ('n_co', 'n_cx', '/'),
    'nhtot': ('cdto', 1.8e+21, '*'), # Effective column density
    'ch_ihx': ('ch_ihp', 'ch_iha', 'ch_ih2', '+', '+'),
    'hstate': ('ch_ihp', 'ch_ih2', '-', 'ch_ihx', '/'),
    'cnt_hx': ('n_hx', 'vol', '*'),
}

ch_mu15 = {
    'Ha': 1.008, 'He': 4.002602, 'C': 12.011, 'O': 15.9994, 'Si': 28.0855,
    'H2': 2.01588, 'Hp': 1.0072, 'Hep': 4.002602, 'CHx': 13., 'OHx': 17., 
    'Cp': 12.011, 'HCOp': 29., 'CO': 28.01, 'Mp': 28., 'M':28.}

var_ch15 = { ### TODO: WORK IN PROGRESS!
    'ch_ihp': ('ihp',),
    'ch_iha': ('iha',),
    'ch_ih2': ('ih2',),
    'ch_icp': ('icp',),
    'ch_ico': ('ico',), 
    'ch_dih2': ('dih2',),
    'ch_dico': ('dico',),
    'n_hp': ('dens', 'ch_ihp', '*', m_a*ch_mu15['Hp'], '/'),
    'n_ha': ('dens', 'ch_iha', '*', m_a*ch_mu15['Ha'], '/'),
    'n_h2': ('dens', 'ch_ih2', '*', m_a*ch_mu15['H2'], '/'),
    'n_cp': ('dens', 'ch_icp', '*', m_a*ch_mu15['Cp'], '/'),
    'n_co': ('dens', 'ch_ico', '*', m_a*ch_mu15['CO'], '/'),
    'n_h1': ('n_ha', 'n_hp', '+'),
    'n_e': ('n_hp', 'n_cp', '+'),
    'n_hx': ('n_ha', 'n_hp', '+', 'n_h2', 2, '*', '+'),
    'n_cx': ('n_co', 'n_cp', '+'),
    'relrate_h2': ('dih2', 'iha', 'ihp', '+', '/'),
    'relrate_co': ('dico', 'icp', '/'),
    'densrate_h2': ('relrate_h2', 'dens_h1', '*'),
    'densrate_co': ('relrate_co', 'dens_cp', '*'),
    'nrate_h2': ('densrate_h2', m_a*ch_mu15['H2'], '/'),
    'nrate_co': ('densrate_co', m_a*ch_mu15['CO'], '/'),
    'madens': ('dens', m_a*ch_mu5['Ha'], '/'), # density in units of H-atom-m/ccm
}

var_chx = {
    'dens_hp': ('dens', 'ch_ihp', '*'),
    'dens_ha': ('dens', 'ch_iha', '*'),
    'dens_h2': ('dens', 'ch_ih2', '*'),
    'dens_cp': ('dens', 'ch_icp', '*'),
    'dens_co': ('dens', 'ch_ico', '*'),
    'dens_h1': ('dens_ha', 'dens_hp', '+'),
    'mass_hp': ('dens_hp', 'vol', '*'),
    'mass_ha': ('dens_ha', 'vol', '*'),
    'mass_h2': ('dens_h2', 'vol', '*'),
    'mass_cp': ('dens_cp', 'vol', '*'),
    'mass_co': ('dens_co', 'vol', '*'),
    'mass_h1': ('dens_h1', 'vol', '*'),
    'massfrac_hp': ('mass_hp', 'mass', '/'),
    'massfrac_ha': ('mass_ha', 'mass', '/'),
    'massfrac_h2': ('mass_h2', 'mass', '/'),
    'massfrac_cp': ('mass_cp', 'mass', '/'),
    'massfrac_co': ('mass_co', 'mass', '/'),
    'massfrac_h1': ('mass_h1', 'mass', '/'),
    'abund_hp': ('n_hp', 'madens', '/'),
    'abund_ha': ('n_ha', 'madens', '/'),
    'abund_h2': ('n_h2', 'madens', '/'),
    'abund_cp': ('n_cp', 'madens', '/'),
    'abund_co': ('n_co', 'madens', '/'),
    'abund_h1': ('n_h1', 'madens', '/'),
    'abund_e': ('n_e', 'madens', '/'),
    'abco_sel': ('abund_co', 1e-4, '>'),
    'abco_dens': ('abco_sel', 'dens', '*'),
    'ionfrac': ('n_e', 'numdens', '/', 2., '*'),
    
    #'massrate_h2': ('dih2', 'ih2', '/', 'mass_h2', '*'),
    #'massrate_co': ('dico', 'ico', '/', 'mass_co', '*'),
    'densrate_h2': ('ch_dih2', 'dens', '*'),
    'densrate_co': ('ch_dico', 'dens', '*'),
    'massrate_h2': ('densrate_h2', 'vol', '*'),
    'massrate_co': ('densrate_co', 'vol', '*'),
}

var_ch5.update(var_chx)
var_ch15.update(var_chx)

# ==== TEST ====================================================================
if __name__ == '__main__':
    filename = 'M:/scratchbin/MW_CF97_supermuc/CF97J/1.TEST/CF97T1_hdf5_plt_cnt_0048'
    ffile = plotfile(filename)
    ffile.learn(var_grid)
    ffile.learn(var_mhd)
    ffile.learn(var_ch5)
