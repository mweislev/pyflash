import numpy as np
cimport numpy as np
cimport cython
from cython.parallel cimport parallel, prange

cdef extern from "cppblocktree.h":
    cdef struct rect3d:
        double xmin
        double ymin
        double zmin
        double xmax
        double ymax
        double zmax

    cdef cppclass amr_tree:
        amr_tree(rect3d extent, size_t Nx, size_t Ny, size_t Nz) nogil except +
        size_t addblock(size_t index, double x, double y, double z, size_t node_type, size_t node_level) nogil except +
        size_t findblock(double x, double y, double z) nogil except +
        size_t findblock(double x, double y, double z, size_t maxlevel) nogil except +
        size_t flush() nogil except +

cdef class Cy_amr_tree:
    cdef amr_tree * c_amr_tree
    def __cinit__(self, double xmin, double ymin, double zmin, double xmax,
    double ymax, double zmax, size_t Nx, size_t Ny, size_t Nz):
        cdef rect3d extent
        extent.xmin = xmin
        extent.ymin = ymin
        extent.zmin = zmin
        extent.xmax = xmax
        extent.ymax = ymax
        extent.zmax = zmax
        self.c_amr_tree = new amr_tree(extent, Nx, Ny, Nz)
    def __dealloc__(self):
        del self.c_amr_tree
    def flush(self):
        self.c_amr_tree.flush()
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef Findblock_c(self, np.ndarray[np.float64_t, ndim=1] x,
    np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] z):
        #
        cdef int i
        cdef int N = len(x)
        cdef np.ndarray[size_t, ndim=1] block = np.empty(N, dtype=np.uint64)
        with nogil, parallel():
            for i in prange(N, schedule='guided'):
                block[i] = self.c_amr_tree.findblock(x[i], y[i], z[i])        
        return block    
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef Findblock_lmax_c(self, np.ndarray[np.float64_t, ndim=1] x,
    np.ndarray[np.float64_t, ndim=1] y, np.ndarray[np.float64_t, ndim=1] z,
    size_t lmax):
        #
        cdef int i
        cdef int N = len(x)
        cdef np.ndarray[size_t, ndim=1] block = np.empty(N, dtype=np.uint64)
        with nogil, parallel():
            for i in prange(N, schedule='guided'):
                block[i] = self.c_amr_tree.findblock(x[i], y[i], z[i], lmax)        
        return block
    @cython.wraparound(False)
    @cython.boundscheck(False)
    cpdef Build(self, np.ndarray[np.float64_t, ndim=3] block_bbox,
    np.ndarray[np.uint64_t, ndim=1] nodetype,
    np.ndarray[size_t, ndim=1] nodelevel):
        # Baum zusammenfuegen:
        cdef np.ndarray[np.float64_t, ndim=2] block_coord = .5*(block_bbox[:,:,0]+block_bbox[:,:,1])
        cdef size_t N = len(block_bbox)
        # !Attention!: Tree buildup parallelization left out on purpose.
        for i in xrange(N):
            self.c_amr_tree.addblock(i, block_coord[i][0], block_coord[i][1],
                block_coord[i][2], nodetype[i], nodelevel[i])
        return block_coord

class blocktree:
    @cython.wraparound(False)
    @cython.boundscheck(False)
    @cython.cdivision(True)
    def __init__(self, np.ndarray[np.float64_t, ndim=3] block_bbox, np.ndarray[np.uint64_t, ndim=1] nodetype):
        cdef size_t i
        cdef np.ndarray[np.float64_t, ndim=2] block_len = block_bbox[:,:,1]-block_bbox[:,:,0]
        cdef np.ndarray[np.float64_t, ndim=1] block_len_max = np.max(block_len, axis=0)
        cdef np.ndarray[size_t, ndim=1] nodelevel = (np.log2(block_len_max[0]/block_len[:,0]) + 1.5).astype(np.uint64)    
        cdef np.float64_t xmin = np.min(block_bbox[:,0,0])
        cdef np.float64_t ymin = np.min(block_bbox[:,1,0])
        cdef np.float64_t zmin = np.min(block_bbox[:,2,0])
        cdef np.float64_t xmax = np.max(block_bbox[:,0,1])
        cdef np.float64_t ymax = np.max(block_bbox[:,1,1])
        cdef np.float64_t zmax = np.max(block_bbox[:,2,1])    
        cdef size_t Nx = np.uint64((xmax-xmin)/block_len_max[0]+.5)
        cdef size_t Ny = np.uint64((ymax-ymin)/block_len_max[1]+.5)
        cdef size_t Nz = np.uint64((zmax-zmin)/block_len_max[2]+.5)        
        # Grundstein fuer Baum legen:
        self.Tree = Cy_amr_tree(xmin, ymin, zmin, xmax, ymax, zmax, Nx, Ny, Nz)
        # Baum zusammenfuegen:
        self.Tree.Build(block_bbox, nodetype, nodelevel)
    def __enter__(self):
        return self
    def __exit__(self, *exc):
        self.Flush()
        return False
    def __del__(self):
        self.Flush()
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def Findblock(self, x, y, z):
        cdef np.ndarray[np.float64_t, ndim=1] xflat = x.flatten()
        cdef np.ndarray[np.float64_t, ndim=1] yflat = y.flatten()
        cdef np.ndarray[np.float64_t, ndim=1] zflat = z.flatten()
        cdef np.ndarray[size_t, ndim=1] blocklist
        blocklist = self.Tree.Findblock_c(xflat, yflat, zflat)
        return blocklist.reshape([(x.shape)[i] for i in range(x.ndim)])        
    @cython.wraparound(False)
    @cython.boundscheck(False)
    def Findblock_lmax(self, x, y, z, size_t lmax):
        cdef np.ndarray[np.float64_t, ndim=1] xflat = x.flatten()
        cdef np.ndarray[np.float64_t, ndim=1] yflat = y.flatten()
        cdef np.ndarray[np.float64_t, ndim=1] zflat = z.flatten()
        cdef np.ndarray[size_t, ndim=1] blocklist
        blocklist = self.Tree.Findblock_lmax_c(xflat, yflat, zflat, lmax)
        return blocklist.reshape([(x.shape)[i] for i in range(x.ndim)])    
    def Flush(self):
        if self.Tree is not None:
            self.Tree.flush()
        self.Tree = None
        return 1