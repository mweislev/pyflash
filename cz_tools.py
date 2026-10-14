import os, sys
import numpy as np
import libmf.micflash as mflash
import libmf.micplot as mplot
import libmf.tools as tools
import sinkpart
import constants as cu
from operator import itemgetter
from py27hash.hash import hash27
from multiprocessing import Pool

#===============================================================================
# ==== SETTINGS ================================================================
#===============================================================================

sinkfile = 'output-sinks_evolES.dat'
regionfile = 'stdzoom_regions.dat'
var_ch = mflash.var_ch5
ax_xy_order = {0:[1,2], 1:[0,2], 2:[0,1]}
ax_units = {'pc':cu.pc, 'Au':cu.au, 'km':1e+5, 'cm':1.}
usorter = dict(key=itemgetter(1), reverse=True)


#===============================================================================
# ==== HELPER ==================================================================    
#===============================================================================

def MoveWindowPeriodic(posx_raw, w_center, w_size, x_per):
    if x_per:
        posx = tools.modclip(posx_raw, w_center-.5*w_size, w_center+.5*w_size)
    else:
        posx = posx_raw
    return posx

def PeriodicDiff(x_raw, x_ctr, x_size, x_per):
    if x_per:
        x_rel = tools.modclip(x_raw -x_ctr, -.5*x_size, +.5*x_size)
    else:
        x_rel = x_raw -x_ctr
    return x_rel

def RelCoords(pos, ctr, octree):
    ctr_x, ctr_y, ctr_z = ctr
    pos_x, pos_y, pos_z = pos
    x_per, y_per, z_per = octree.periodic()
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    x_size, y_size, z_size = xmax -xmin, ymax -ymin, zmax -zmin
    rel_x = PeriodicDiff(pos_x, ctr_x, x_size, x_per)
    rel_y = PeriodicDiff(pos_y, ctr_y, y_size, y_per)
    rel_z = PeriodicDiff(pos_z, ctr_z, z_size, z_per)
    return rel_x, rel_y, rel_z


def BG_UnwindPeriodic(octree, coords_raw):
    x_raw, y_raw, z_raw = coords_raw
    ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
    x_unw = tools.modclip(x_raw, xmin, xmax)
    y_unw = tools.modclip(y_raw, ymin, ymax)
    z_unw = tools.modclip(z_raw, zmin, zmax)
    return x_unw, y_unw, z_unw

def BG_UnrelCoords(octree, relcoords, center):
   coords_raw = [relcoords[j]+center[j] for j in range(3)]
   abscoords = BG_UnwindPeriodic(octree, coords_raw)
   return abscoords

def BG_RelCoords(octree, center, ng=0, force=False):
   ''' Determine radial vector from center to cells using geometric cell centers,
   while rotating the coordinate system window to center the center '''
   coords = octree.coords(force=force, ng=ng)
   x_raw, y_raw, z_raw = [coords[:,i,:,:,:] for i in range(3)]
   x_per, y_per, z_per = octree.periodic()
   x_ctr, y_ctr, z_ctr = center
   ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
   x_size, y_size, z_size = xmax -xmin, ymax -ymin, zmax -zmin
   x_rel = PeriodicDiff(x_raw, x_ctr, x_size, x_per)
   y_rel = PeriodicDiff(y_raw, y_ctr, y_size, y_per)
   z_rel = PeriodicDiff(z_raw, z_ctr, z_size, z_per)
   return x_rel, y_rel, z_rel

def BG_RelCoords_D(leafdata, octree, center):
   ''' Determine radial vector from center to cells using cached cell center of masses,
   while rotating the coordinate system window to center the center '''
   x_raw, y_raw, z_raw = leafdata['posx'], leafdata['posy'], leafdata['posz']
   x_per, y_per, z_per = octree.periodic()
   x_ctr, y_ctr, z_ctr = center
   ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
   x_size, y_size, z_size = xmax -xmin, ymax -ymin, zmax -zmin
   x_rel = PeriodicDiff(x_raw, x_ctr, x_size, x_per)
   y_rel = PeriodicDiff(y_raw, y_ctr, y_size, y_per)
   z_rel = PeriodicDiff(z_raw, z_ctr, z_size, z_per)
   return x_rel, y_rel, z_rel

def PickAxisUnit(extent, rth=.1):
   exmax = np.max(np.abs(extent))
   for axulabel, axunit in sorted(ax_units.items(), **usorter):
       if exmax > rth*axunit: break
   return axunit, axulabel

def PickAxUnit(axis, regdict):
   ax_x, ax_y = ax_xy_order[axis]
   extdict = RegionExtent(regdict, scale=1.0)
   xyz_extent = extdict['x_extent'], extdict['y_extent'], extdict['z_extent']
   regextent_cgs = xyz_extent[ax_x], xyz_extent[ax_y]
   axunit, axulabel = PickAxisUnit(regextent_cgs, rth=.1)
   return axunit, axulabel


#===============================================================================
# ==== PLOTFILE I/O ============================================================ 
#===============================================================================

def LoadCellPos(hdfdata, octree):
   # Put down cell center coordinates into cell data cache.
   # We want it as a field instead of being hard-coded to be able
   # to account for a possible displacement of each cells center of mass,
   # e.g. when accounting for sink particles.
   # ( Note: Technically, we perhaps should reconstruct the Jacobian of the
   #   density field in each cell while using a flux limiter, and then 
   #   calculate each cells center of mass accordingly.)
   coords = octree.coords(force=True)
   posx, posy, posz = [coords[:,i,:,:,:] for i in range(3)]
   hdfdata.cache('posx', posx)
   hdfdata.cache('posy', posy)
   hdfdata.cache('posz', posz)
   return None

def LoadPlotfile(dirpath, filename=None, verbose=False, memorize=True):
   vv = max(0,int(verbose)-1)
   if filename is None:
      plotfile = dirpath
   else:
      plotfile = os.path.join(dirpath, filename)
   if verbose > 0:
      print(f'\n>> Processing {plotfile}')
   assert os.path.isfile(plotfile)
   # Open plotfile data
   leafdata = mflash.plotfile(plotfile, memorize=memorize)
   leafdata.learn(mflash.var_mhd)
   leafdata.learn(mflash.var_grid)
   leafdata.learn(var_ch)
   # Get octree from plotfile
   octree = mflash.pm3dgrid(leafdata, verbose=vv)
   # Load cell center positions (posx,posy,posz) into cell data cache
   LoadCellPos(leafdata, octree)
   #
   return leafdata, octree

def Simname(leafdata):
   simdir, pfname = os.path.split(leafdata.filename())
   simname = os.path.basename(simdir)
   while simname in ['out','old']:
      simdir, subdir = os.path.split(simdir)
      simname = os.path.basename(simdir)
   return simname


#===============================================================================
# ==== SINK PARTICLE I/O =======================================================  
#===============================================================================

def SinkTable(simdir):
   fp = os.path.join(simdir, sinkfile)
   sinkdata = sinkpart.SinkData(fp)
   return sinkdata

sinklist_props = {
   'x' : 1, 'y' : 2, 'z' : 3,
   'posx' : 1, 'posy' : 2, 'posz' : 3,
   'vx' : 4, 'vy' : 5, 'vz' : 6,
   'velx' : 4, 'vely' : 5, 'velz' : 6,
   'prx' : 7, 'pry' : 8, 'prz' : 9,
   'vprx' : 10, 'vpry' : 11, 'vprz' : 12,
   'd0nx' : 13, 'd0ny' : 14, 'd0nz' : 15,
   'd1nx' : 16, 'd1ny' : 17, 'd1nz' : 18,
   'd2nx' : 19, 'd2ny' : 20, 'd2nz' : 21,
   'd3nx' : 22, 'd3ny' : 23, 'd3nz' : 24,
   'nd0nx' : 25, 'nd0ny' : 26, 'nd0nz' : 27,
   'nd1nx' : 28, 'nd1ny' : 29, 'nd1nz' : 30,
   'd0dx' : 31, 'd0dy' : 32, 'd0dz' : 33,
   'd1dx' : 34, 'd1dy' : 35, 'd1dz' : 36,
   'd2dx' : 37, 'd2dy' : 38, 'd2dz' : 39,
   'd3dx' : 40, 'd3dy' : 41, 'd3dz' : 42,
   'nd0dx' : 43, 'nd0dy' : 44, 'nd0dz' : 45,
   'nd1dx' : 46, 'nd1dy' : 47, 'nd1dz' : 48,
   'acc' : 49,
   'stepInt' : 50,
   'dtkLev' : 51,
   'tnewInt' : 52,
   'told' : 53,
   'inti' : 54,
   'gpot' : 55,
   'angx' : 56, 'angy' : 57, 'angz' : 58,
   'massDot' : 59,
   'mass' : 60,
   't0' : 61,
   'uvlum' : 62,
   'zams' : 63,
   'clusterSink' : 64,
   'accrActive' : 65,
   'activeMass' : 66,
   'bp' : 67,
   'originalMass' : 68,
   'enex' : 69,
   'radTemp' : 70,
   'snN' : 71,
   'snMass' : 72,
   'windMass' : 73,
   'windVel' : 74,
   'windEk' : 75,
   'activeN' : 76,
   'activeOB' : 77,
   'trMassSN' : 78,
   'trMassWind' : 79,
   'gpotGas' : 84,
   'tag' : 85,
   'part_tag' : 85,
   'softGroup' : 86,
   'proc' : 87,
   'blk' : 88,
   'subrad' : 89,
   'stage' : 90,
   'mdeut' : 91,
   'lint' : 92,
   'ltot' : 93,
   'idk' : 94,
   'fuvlum' : 95,
   'nonlum' : 96,
   'stellmass' : 97,
   'stellmdot' : 98,
   'stellangx' : 99, 'stellangy' : 100, 'stellangz' : 101,
   'accevact' : 102,
   'stellangdotx' : 103, 'stellangdoty' : 104, 'stellangdotz' : 105,
   'outflow' : 106,
   'angdotx' : 107, 'angdoty' : 108, 'angdotz' : 109,
   'temp' : 110,
   'ltemp' : 110,
   'iadmass' : 111,
   'iadangx' : 112, 'iadangy' : 113, 'iadangz' : 114,
   'oadmass' : 115,
   'oadangx' : 116, 'oadangy' : 117, 'oadangz' : 118,
}
sinklist_intprops = [71,85,86,87,88,90]

def SinkData(hdfdata):
   try:
      nsink = hdfdata['integer scalars/hs_nsink']
   except KeyError:
      nsink = 0
   if nsink:
      sldata = hdfdata['sinkList']
      nsink_list = 100*(1+(nsink-1)//100)
      nprops = len(sldata)//nsink_list
   else:
      sldata = np.array(list())
      nprops = int(np.max([val for key,val in sinklist_props.items()]))
   idxstride = nprops
   idxmax = nsink*idxstride
   sinkdict = dict(nsink=nsink, nprops=nprops)
   for iprop in range(1,nprops+1):
      idxoffset = iprop-1
      propdata = sldata[idxoffset:idxmax:idxstride]
      if iprop in sinklist_intprops: propdata = propdata.astype(int)
      sinkdict[iprop] = propdata
   for key,iprop in sinklist_props.items():
      idxoffset = iprop-1
      if iprop>nprops:
         print(f'Warning: Missing HS properties > {nprops}')
         break
      propdata = sldata[idxoffset:idxmax:idxstride]
      if iprop in sinklist_intprops: propdata = propdata.astype(int)
      sinkdict[key] = propdata
   return sinkdict

def BG_SinkMass(hdfdata, octree, sinkdict):
   dshape = hdfdata.dshape()
   msink = np.zeros(dshape)
   nsink = np.zeros(dshape, dtype=int)
   if sinkdict['nsink'] > 0:
      Sm = sinkdict['mass']
      Sx, Sy, Sz = sinkdict['x'], sinkdict['y'], sinkdict['z']
      Scells = octree.find(Sx,Sy,Sz)
      for isink, (bl,k,j,i) in enumerate(zip(*Scells)):
         # Store mass of sink
         sinkmass = Sm[isink]
         msink[bl,k,j,i] += sinkmass
         nsink[bl,k,j,i] += 1
   hdfdata.cache('msink', msink)
   hdfdata.cache('nsink', nsink)
   return None

def BG_MapSinksToGrid(hdfdata, octree, sinkdict, debug=False):
   '''
   Modify the grid data (cell positions, cell velocities, densities)
   to account for the masses and momenta stored in sink particles.
   '''
   # We need to flush the leafdata cache since there might be quite some
   # information that depends on mass, which will be changed here!
   # Note that we must run other cache loads only after this routine!
   hdfdata.flush()
   # First load back in the grid cell center position data
   LoadCellPos(hdfdata, octree)
   # Now locally store all fields that we want to change.
   posx, posy, posz = hdfdata['posx'], hdfdata['posy'], hdfdata['posz']
   velx, vely, velz = hdfdata['velx'], hdfdata['vely'], hdfdata['velz']
   msink = np.zeros_like(posx)
   nsink = np.zeros_like(posx, dtype=int)
   dens = hdfdata['dens']
   # Also fetch other fields that are needed for our calculations
   vol = hdfdata['vol']
   # Find out which leaf cells the sink particles correspond to
   Sx, Sy, Sz = sinkdict['x'], sinkdict['y'], sinkdict['z']
   Scells = octree.find(Sx,Sy,Sz)
   # Add sinks iteratively. Note that we technically account for the possibility
   # that multiple sinks might be contained in the same cell, even though
   # this should hopefully not happen when using a sensible sink and refinement setup.
   Svx, Svy, Svz = sinkdict['vx'], sinkdict['vy'], sinkdict['vz']
   Sm = sinkdict['mass']
   for isink, (bl,k,j,i) in enumerate(zip(*Scells)):
      sinkmass = Sm[isink]
      cellmass = dens[bl,k,j,i] * vol[bl,k,j,i]
      totmass = sinkmass + cellmass
      fcell = cellmass / totmass
      fsink = sinkmass / totmass
      assert np.isclose(fcell+fsink, 1.)
      # Modify cell center of mass position
      sinkposx, sinkposy, sinkposz = Sx[isink], Sy[isink], Sz[isink]
      cellposx, cellposy, cellposz = posx[bl,k,j,i], posy[bl,k,j,i], posz[bl,k,j,i]
      posx[bl,k,j,i] = cellposx*fcell + sinkposx*fsink
      posy[bl,k,j,i] = cellposy*fcell + sinkposy*fsink
      posz[bl,k,j,i] = cellposz*fcell + sinkposz*fsink
      # Modify cell center of mass velocity
      sinkvelx, sinkvely, sinkvelz = Svx[isink], Svy[isink], Svz[isink]
      cellvelx, cellvely, cellvelz = velx[bl,k,j,i], vely[bl,k,j,i], velz[bl,k,j,i]
      velx[bl,k,j,i] = cellvelx*fcell + sinkvelx*fsink
      vely[bl,k,j,i] = cellvely*fcell + sinkvely*fsink
      velz[bl,k,j,i] = cellvelz*fcell + sinkvelz*fsink
      # Modify cell density to account for sink mass
      dens[bl,k,j,i] += sinkmass / vol[bl,k,j,i]
      # Store mass of sink
      msink[bl,k,j,i] += sinkmass
      nsink[bl,k,j,i] += 1
   # Now write the modified fields into the cell data cache.
   # When retrieving data, this will override the information from the plotfile.
   # XXX CHECK THIS ASSERTION, I MIGHT HAVE CHANGED THAT FOR A CERTAIN REASON! XXX
   hdfdata.cache('posx', posx)
   hdfdata.cache('posy', posy)
   hdfdata.cache('posz', posz)
   hdfdata.cache('velx', velx)
   hdfdata.cache('vely', vely)
   hdfdata.cache('velz', velz)
   hdfdata.cache('dens', dens)
   hdfdata.cache('msink', msink)
   hdfdata.cache('nsink', nsink)
   #
   if debug:
      # Verify that the displaced cell centers are still inside the active cell.
      # Since the sink masses used to calculate the new cell center of mass
      # are always inside the respective cell, this should always be true from a
      # mathematical standpoint; But:
      # Note that this assumes careful management of the sink position inside Flash
      # when using a periodic boundary condition
      print('Verifying cell position integrity... ', end='')
      sys.stdout.flush()
      Ic = octree.find(posx,posy,posz)
      assert np.all(posx[Ic]==posx)
      assert np.all(posy[Ic]==posy)
      assert np.all(posz[Ic]==posz)
      print('OK')
      sys.stdout.flush()
   #
   return None

def MapSinkMass(hdfdata, octree, sinkdict):
   dshape = hdfdata['mass'].shape
   sinkmass = np.zeros_like(dshape)
   X = sinkdict['x']
   Y = sinkdict['y']
   Z = sinkdict['z']
   M = sinkdict['mass']
   cells = octree.find(X,Y,Z)
   for idx,(bl,k,j,i) in enumerate(zip(*cells)):
      sinkmass[bl,k,j,i] += M[idx]
   return sinkmass

def ApproxStarTemp(lum,rad):
   """
   Solve Stefan-Boltzmann law for black body temp. given a sphere radius and luminosity
   """
   sb = 7.125562866e-4
   temp = (lum/(sb*rad**2))**(.25)
   return temp


#===============================================================================
# ==== PARTICLE FILE READER ====================================================    
#===============================================================================

partdata_dtypes = {'hash':np.int_, 'tag':np.int_, 'blk':np.int_, 'proc':np.int_}

def PartfileFromPlotfile(plotfile):
   dn, plotfn = os.path.split(plotfile)
   if '_chk_' in plotfn:
      partfile = plotfile
   else:
      basenm, cnt = plotfn.split('_plt_cnt_')
      partfn = f'{basenm}_part_{cnt}'
      partfile = os.path.join(dn, partfn)
   return partfile

def PartfileName(leafdata):
   plotfile = leafdata.filename()
   return PartfileFromPlotfile(plotfile)

def _PD_sighash(s):
   return hash27(str(s))

def PartfileData(partfile, verbose=2, sort=False):
   if verbose>0:
      print(f'Reading particle data: {partfile}...')
      sys.stdout.flush()
   assert os.path.isfile(partfile)
   with mflash.flashfile(partfile) as partdata:
      trdata = partdata.read('/tracer particles')
      partnames = partdata.read('/particle names')#, dtype=np.string_)
   l_partnames = list(partnames)+['hash',]
   formats = [partdata_dtypes.get(pn,np.float64) for pn in l_partnames]
   data = np.zeros(len(trdata), dtype={'names':l_partnames, 'formats':formats})
   for i,key in enumerate(partnames):
      data[key] = trdata[:,i]
   # Add a unique hash to each tracer particle
   npart = len(trdata)
   if verbose>0:
      print(f'Hashing {npart} particles...', end=' ')
      sys.stdout.flush()
   #data['hash'] = np.array([hash27(str(s)) for s in signatures])
   pool = Pool()
   signatures = zip(data['tag'], data['tcreat'])
   hashmap = pool.map(_PD_sighash, signatures)
   data['hash'] = np.array(list(hashmap))
   #
   if verbose>0:
      print('OK')
      sys.stdout.flush()
   #
   if sort:
      I = np.argsort(data['tag'])
      return data[I]
   else:
      return data

def PartCoords(leafdata):
   pfn = PartfileName(leafdata)
   partdata = PartfileData(pfn)
   partX = partdata['posx']
   partY = partdata['posy']
   partZ = partdata['posz']
   partcoords = np.array([partX,partY,partZ])
   return partcoords

def PartColors(leafdata):
   pfn = PartfileName(leafdata)
   partdata = PartfileData(pfn)
   fdbtype = partdata['fdbtype']
   fdbcolors = np.array(['green' if t<0.5 else 'white' for t in fdbtype])
   print(f'fdbtype.min: {fdbtype.min()}, fdbtype.max: {fdbtype.max()}')
   return fdbcolors


#===============================================================================
# ==== REGION FILE READER ======================================================    
#===============================================================================

def ReadHeader(line):
   items = line.split()
   if len(items) < 20:
      return None # Too short, not a header (misplaced comment?)
   return items[1:]

def ReadValue(item):
   try:
      value = float(item)
   except ValueError:
      value = item
   else:
      if value.is_integer() and abs(value)<1e+7:
         value = int(value)
   return value

def ReadData(line):
   items = line.split()
   if len(items) < 20:
      return None # Too short, not a data line (empty line?)
   data = [ReadValue(it) for it in items]
   return data
   
def SimDomainRegion(simdir):
   parfile = os.path.join(simdir,'flash.par')
   pardict = tools.readpar(parfile)
   regdict = dict(nreg=0, velx=0., vely=0., velz=0., shape=16, nstep=-1, tref=-1.)
   xmin, xmax = pardict['xmin'], pardict['xmax']
   ymin, ymax = pardict['ymin'], pardict['ymax']
   zmin, zmax = pardict['zmin'], pardict['zmax']
   regdict['posx'] = .5*(xmax+xmin)
   regdict['posy'] = .5*(ymax+ymin)
   regdict['posz'] = .5*(zmax+zmin)
   regdict['radx'] = .5*(xmax-xmin)
   regdict['rady'] = .5*(ymax-ymin)
   regdict['radz'] = .5*(zmax-zmin)
   return regdict
   
def RegionTable(simdir, domainreg=False):
   keys = list()
   rt = list()
   if domainreg:
      domain_regdict = SimDomainRegion(simdir)
      rt.append(domain_regdict)
   fp = os.path.join(simdir, regionfile)
   if not os.path.isfile(fp):
      if not domainreg:
         raise IOError(f'RegionTable: File not found: {fp}')
      else:
         print(f'Warning: RegionTable: File not found: {fp}')
         return rt
   f =  open(fp, "r")
   for l,line in enumerate(f):
      if '!' in line: # this is a comment line
         continue
      elif '#' in line: # this is supposed to be a header line
         header = ReadHeader(line)
         if header is None: # ReadHeader did not like it
            continue
         else:
            keys = header.copy()
      else: # this might be a data line
         data = ReadData(line)
         if data is None:
            continue
         elif not (len(data)==len(keys)):
            raise IOError(f'ReadRegions: Inconsistent number of values (line {l+1})')
         elif not keys:
            raise IOError(f'ReadRegions: Header not found (line {l+1})')
         else:
            regdict = {k:v for k,v in zip(keys,data)}
            rt.append(regdict)
   return rt

def StepRegions(leafdata, regtable=None, debug=False):
   if regtable is None:
      # Open region data
      simdir = leafdata.dirname().removesuffix('/out')
      regtable = RegionTable(simdir, domainreg=False)
   if debug:
      for item in regtable: print(item)
   sr = dict()
   nstep_plt = leafdata['integer scalars/nstep']
   simtime_plt = leafdata['real scalars/time']
   nstep_ = np.array([ld.get('nstep', np.nan) for ld in regtable])
   use_nstep =  np.all(np.isfinite(nstep_))
   if use_nstep:
      # Search nstep closest to nstep_plt,
      # requiring nstep <= nstep_plt to omit future regions
      nstep_candidates = nstep_[nstep_<=nstep_plt]
      imatch = np.argmin(nstep_plt-nstep_candidates)
      nmatch = nstep_candidates[imatch]
      # Note: n<0 for persistent regions (eg. n=-1 for domain)
      I = [i for i,n in enumerate(nstep_) if (n==nmatch or n<0)]
   else:
      dt = leafdata['real scalars/dt']
      tref = np.array([ld['tref'] for ld in regtable])
      tref -= 0.5*dt # Adjust for limited resolution in output file.
      tref_candidates = tref[tref<=simtime_plt]
      imatch = np.argmin(simtime_plt-tref_candidates)
      tmatch = tref_candidates[imatch]
      # Note: tref<0.0 for persistent regions (eg. tref=-1.0 for domain)
      I = [i for i,n in enumerate(tref) if (tref==tmatch or tref<0.)]
   sr = [regtable[idx] for idx in I]
   return sr


#===============================================================================
# ==== ZOOMREG HELPER ==========================================================
#===============================================================================

def RegionExtent(regdict, scale=1.0, dictout=True):
   d = dict()
   xctr, xrad = regdict['posx'], regdict['radx']
   yctr, yrad = regdict['posy'], regdict['rady']
   zctr, zrad = regdict['posz'], regdict['radz']
   d['x_extent'] = xctr-scale*xrad, xctr+scale*xrad
   d['y_extent'] = yctr-scale*yrad, yctr+scale*yrad
   d['z_extent'] = zctr-scale*zrad, zctr+scale*zrad
   if dictout:
      return d
   else:
      return d['x_extent'], d['y_extent'], d['z_extent']

def RegionMask(leafdata, octree, regdict):
   baseshape = leafdata.dshape() # This works, but is a bit expensive...
   mask = np.ones(baseshape, dtype=bool)
   center = regdict['posx'], regdict['posy'], regdict['posz']
   x_rad, y_rad, z_rad = regdict['radx'], regdict['rady'], regdict['radz']
   x_rel, y_rel, z_rel = BG_RelCoords(octree, center, force=True)
   mask *= (-x_rad<=x_rel) * (x_rel<x_rad)
   mask *= (-y_rad<=y_rel) * (y_rel<y_rad)
   mask *= (-z_rad<=z_rel) * (z_rel<z_rad)
   return mask

def DomainToRegdict(leafdata, octree=None, maxaspect=20.,offset=None):
   if offset is None:
      offset = np.zeros(3)
   block_bbox = leafdata['bounding box']
   bbox_min = block_bbox.min(axis=2)
   bbox_max = block_bbox.max(axis=2)
   domain_min = bbox_min.min(axis=0)
   domain_max = bbox_max.max(axis=0)
   xmin, xmax = (domain_min[0], domain_max[0])
   ymin, ymax = (domain_min[1], domain_max[1])
   zmin, zmax = (domain_min[2], domain_max[2])
   regdict = dict()
   regdict['posx'] = .5*(xmax+xmin) +offset[0]
   regdict['posy'] = .5*(ymax+ymin) +offset[1]
   regdict['posz'] = .5*(zmax+zmin) +offset[2]
   radx = .5*(xmax-xmin)
   rady = .5*(ymax-ymin)
   radz = .5*(zmax-zmin)
   radmax = np.min((radx,rady,radz))*maxaspect
   regdict['radx'] = min(radx,radmax)
   regdict['rady'] = min(rady,radmax)
   regdict['radz'] = min(radz,radmax)
   return regdict

def MaskToRegdict(leafdata, octree, mask):
   if not np.any(mask):
      raise ValueError('MaskToRegdict: Received empty mask')
   if np.all(mask):
      return DomainToRegdict(leafdata, octree)
   block_bbox = leafdata['bounding box']
   bbox_min = block_bbox.min(axis=2)
   bbox_max = block_bbox.max(axis=2)
   domain_min = bbox_min.min(axis=0)
   domain_max = bbox_max.max(axis=0)
   domain_radius = .5*(domain_max-domain_min)
   drx, dry, drz = domain_radius
   csx, csy, csz = leafdata['lenx'], leafdata['leny'], leafdata['lenz']
   coords = octree.coords(force=True)
   x, y, z = (coords[:,i] for i in range(3))
   x_avg = np.median(x[mask])
   y_avg = np.median(y[mask])
   z_avg = np.median(z[mask])
   ip = np.argmin((x[mask]-x_avg)**2+(y[mask]-y_avg)**2+(z[mask]-z_avg)**2)
   x_pivot = np.ravel(x[mask])[ip]
   y_pivot = np.ravel(y[mask])[ip]
   z_pivot = np.ravel(z[mask])[ip]
   x_wrapped = mflash.modclip(x, x_pivot-drx, x_pivot+drx)
   y_wrapped = mflash.modclip(y, y_pivot-dry, y_pivot+dry)
   z_wrapped = mflash.modclip(z, z_pivot-drz, z_pivot+drz)
   xmin_cell, xmax_cell = x_wrapped -.5*csx, x_wrapped +.5*csx
   ymin_cell, ymax_cell = y_wrapped -.5*csy, y_wrapped +.5*csy
   zmin_cell, zmax_cell = z_wrapped -.5*csz, z_wrapped +.5*csz
   xmin_mask, xmax_mask = (np.min(xmin_cell[mask]), np.max(xmax_cell[mask]))
   ymin_mask, ymax_mask = (np.min(ymin_cell[mask]), np.max(ymax_cell[mask]))
   zmin_mask, zmax_mask = (np.min(zmin_cell[mask]), np.max(zmax_cell[mask]))
   regdict = dict()
   regdict['posx'] = .5*(xmax_mask+xmin_mask)
   regdict['posy'] = .5*(ymax_mask+ymin_mask)
   regdict['posz'] = .5*(zmax_mask+zmin_mask)
   regdict['radx'] = .5*(xmax_mask-xmin_mask)
   regdict['rady'] = .5*(ymax_mask-ymin_mask)
   regdict['radz'] = .5*(zmax_mask-zmin_mask)
   return regdict

def ExtentToMask(hdfdata, octree, extent):
   x_ctr, y_ctr, z_ctr = [.5*(ext_j[1]+ext_j[0]) for ext_j in extent]
   coords = octree.coords(force=True)
   x,y,z = [coords[:,d] for d in range(3)]
   x_per, y_per, z_per = octree.periodic()
   ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
   x_size, y_size, z_size = xmax -xmin, ymax -ymin, zmax -zmin
   x_winded = MoveWindowPeriodic(x, x_ctr, x_size, x_per)
   y_winded = MoveWindowPeriodic(y, y_ctr, y_size, y_per)
   z_winded = MoveWindowPeriodic(z, z_ctr, z_size, z_per)
   x_extent, y_extent, z_extent = extent
   x_extmin, x_extmax = x_extent
   y_extmin, y_extmax = y_extent
   z_extmin, z_extmax = z_extent
   extmask  = (x_extmin<=x_winded)*(x_winded<=x_extmax)
   extmask *= (y_extmin<=y_winded)*(y_winded<=y_extmax)
   extmask *= (z_extmin<=z_winded)*(z_winded<=z_extmax)
   return extmask


#===============================================================================
# ==== MASKED OBJECT HELPER ====================================================
#===============================================================================

def ObjCenter(hdfdata, octree, objwhere):
   # Get object cell raw positions
   posx_raw = hdfdata['posx'][objwhere]
   posy_raw = hdfdata['posy'][objwhere]
   posz_raw = hdfdata['posz'][objwhere]
   # Rotate the coordinate system window to center one of the object
   # grid points (i.e. select the first one) according to per. bounds.
   ((xmin, xmax), (ymin, ymax), (zmin, zmax)) = octree.extent()
   xsize, ysize, zsize = xmax -xmin, ymax -ymin, zmax -zmin
   x_per, y_per, z_per = octree.periodic()
   posx = MoveWindowPeriodic(posx_raw, posx_raw[0], xsize, x_per)
   posy = MoveWindowPeriodic(posy_raw, posy_raw[0], ysize, y_per)
   posz = MoveWindowPeriodic(posz_raw, posz_raw[0], zsize, z_per)
   cellpos = (posx,posy,posz)
   # Calculate center of mass using rotated grid window coordinates
   mass = hdfdata['mass'][objwhere]
   center = [np.average(cellpos[i], weights=mass) for i in range(3)]
   center_unw = BG_UnwindPeriodic(octree, center)
   return np.array(center_unw)

def RadiusExtent(center, radius, a=2.):
   # Why did I put this factor 2 here? Smells wrong!
   x_extent = (center-a*radius)[0], (center+a*radius)[0]
   y_extent = (center-a*radius)[1], (center+a*radius)[1]
   z_extent = (center-a*radius)[2], (center+a*radius)[2]
   return x_extent, y_extent, z_extent

def ObjBoundingBox(hdfdata, octree, objwhere, objcenter, margin=1.1):
   relcoords = BG_RelCoords(octree, objcenter)
   xrel, yrel, zrel = [relcoords[i][objwhere] for i in range(3)]
   rx = np.max(np.fabs(xrel))
   ry = np.max(np.fabs(yrel))
   rz = np.max(np.fabs(zrel))
   r3d = np.array([rx,ry,rz])
   bbox = RadiusExtent(objcenter, r3d, a=margin)
   return bbox

def ObjBboxMask(hdfdata, octree, objwhere, debug=True):
   dshape = hdfdata.dshape()
   mask = np.zeros(dshape, dtype=bool)
   mask[objwhere] = True
   if debug: print(f'Object cells: {np.sum(mask)}')
   assert np.any(mask)
   obj_center = ObjCenter(hdfdata, octree, mask)
   if debug: print(f'Object center: {obj_center}')
   obj_bbox = ObjBoundingBox(hdfdata, octree, objwhere, obj_center, margin=2.0)
   if debug: print(f'Object bbox: {obj_bbox}')
   bbox_mask = ExtentToMask(hdfdata, octree, obj_bbox)
   if debug: print(f'Object bbox cells: {np.sum(bbox_mask)}')
   assert np.any(bbox_mask)
   return bbox_mask


#===============================================================================
# ==== COLOR HELPER ============================================================
#===============================================================================
# This should go to libmf/micplot.py,
# but there is some similar old (incompatible?) code there,
# and I got no time for cleanup.

cR, cG, cB = .299, .587, .114 # PAL
# Define corresponding orthonormal (R,G,B) vectors describing a plane of
# constant luminance, i.e. Y(R,G,B) = 0:
v1  = np.array((-cR, -cG, (cR**2+cG**2)*1./cB))
v1 /= np.linalg.norm(v1)
v2  = np.array((cG, -cR, 0.))
v2 /= np.linalg.norm(v2)
Rv = np.vstack((v1,v2))

def HSLtoRGB(Y, sat=1., hue=1., gamma=.7, relsat=True):
   dshape = (Y+sat+hue+gamma).shape
   # Calculate Luma
   Ys = Y**gamma * np.ones(dshape)
   # Calculate Hue vector
   H = (1./3.)*np.pi*hue * np.ones(dshape)
   n = np.array((np.cos(H), np.sin(H)))
   Hn = np.dot(n.T, Rv).T
   # Calculate Chroma
   Cmax = np.min(np.maximum((1.-Ys)/Hn,(0.-Ys)/Hn), axis=0)
   if relsat:
      Cx = sat*Cmax
   else:
      Cx = sat*Ys*(1.-Ys)
   C = np.clip(Cx,0.,Cmax)
   # Calculate RGB
   RGB = Ys + C*Hn
   return RGB.T

def RGBtoHSL(R,G,B, gamma=.7, relsat=True):
   # Calculate Luminance
   Ys = cR*R + cG*G + cB*B
   Y = Ys**(1./gamma)
   # Calculate Hue
   RH, GH, BH = R-Ys, G-Ys, B-Ys
   Hv1 = RH*v1[0] + GH*v1[1] + BH*v1[2]
   Hv2 = RH*v2[0] + GH*v2[1] + BH*v2[2]
   H = np.arctan2(Hv2,Hv1)
   hue = tools.modclip(3.*H/np.pi, 0., 6.)
   # Calculate Saturation
   Cx = np.sqrt(RH**2+GH**2+BH**2)
   if relsat:
      n = np.array((np.cos(H), np.sin(H)))
      Hn = np.dot(n.T, Rv).T
      Cmax = np.min(np.maximum((1.-Ys)/Hn,(0.-Ys)/Hn), axis=0)
      sat = np.nan_to_num(Cx/Cmax)
   else:
      sat = np.nan_to_num(Cx/(Ys*(1.-Ys)))
   #
   return Y,sat,hue

#===============================================================================
# ==== DATA-SPECIFIC COLORBAR HELPER ===========================================
#===============================================================================

# Idea: Log-colorbar with clearly visible orders of magnitude
# P1: Use (Semi-)progressive Hue/Saturation/Lightness
# P2: Align principal colors with order of mag bounds
# P3a: Use increased saturation at mag bounds
# P3b: Use uniform saturation everywhere else
# P4: Hue diffs easy to spot, light diffs harder
#     * Use continuous Hue ramp
#     * Use lightness stair profile
#       Align stair profile (2--3 stairs/mag) with mag bounds
# P5: Ignore spurious very low/high values

def makeCmap(X, RGB, mapname):
   R, G, B = RGB.T
   cdict = {
      'red':  [(x,R[i],R[i]) for i,x in enumerate(X)],
      'green': [(x,G[i],G[i]) for i,x in enumerate(X)],
      'blue':  [(x,B[i],B[i]) for i,x in enumerate(X)],}
   mplot.register_cmap(name=mapname, data=cdict)
   return mapname

def GetAdaptiveLogCbar(norm, H0=0., Hf=1., L0=.1, Lf=.8, sat0=.35, satt=1.0, satf=7., relsat=True, gamma=.7, mapname='LogAdaptive'):
   vlmin, vlmax = np.log10(norm.vmin), np.log10(norm.vmax)
   nmag = vlmax-vlmin
   mag0 = tools.modclip(vlmin,-1.,0.)
   #
   if nmag<5:
      Cfunc = lambda X: H0+2.*Hf*(nmag*X+mag0)
      Yfunc = lambda X: L0+Lf/np.ceil(nmag)*((3.*(nmag*X+mag0))//1-3.*np.floor(mag0))/3.
   else:
      Cfunc = lambda X: H0+Hf*(nmag*X+mag0)
      Yfunc = lambda X: L0+Lf/np.ceil(nmag)*((2.*(nmag*X+mag0))//1-2.*np.floor(mag0))/2.
   Pfunc = lambda X: sat0+(satt-sat0)*np.exp(-satf*np.minimum(np.abs(-(nmag*X+mag0)%1),np.abs((nmag*X+mag0)%1)))
   RGBfunc = lambda X: HSLtoRGB(Yfunc(X), sat=Pfunc(X), hue=Cfunc(X), relsat=relsat, gamma=.7)
   X = np.linspace(0., 1., 1024)
   return  makeCmap(X, RGBfunc(X), mapname)
   

def GetNormCbar(data, vmin=None, vmax=None):
   v = np.ravel(data)
   vl = np.log10(v)
   p15, p50, p85 = np.percentile(v, [5,50,95])
   vl_width = max((p50-p15),(p85-p50))
   if vmin is None:
      vl_min = p50 +3.*vl_width
   else:
      vl_min = np.log10(vmin)
   if vmax is None:
      vl_max = p50 -3.*vl_width
   else:
      vl_min = np.log10(vmax)
   # XXX TODO XXX
   raise NotImplementedError()
