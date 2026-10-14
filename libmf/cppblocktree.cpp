//
//  blocktree.cpp
//
//  Created by Michael Weis on 10.09.16.
//

#include <new>
#include <algorithm>
#include <stdexcept>
#include "cppblocktree.h"

using namespace std;

inline size_t partindex(double a, double amin, double partsize_recp){
    return size_t((a - amin)*partsize_recp);
}

inline size_t part3dindex(double * node_min, double * d_recp, size_t * N, double x, double y, double z){
    size_t I = min(partindex(x, node_min[0], d_recp[0]), N[0]-1);
    I += N[0]*min(partindex(y, node_min[1], d_recp[1]), N[1]-1);
    I += N[0]*N[1]*min(partindex(z, node_min[2], d_recp[2]), N[2]-1);
    return I;
}

rect3d part3dextent(double * extent_min, double * extent_max, size_t Nx, size_t Ny, size_t Nz, size_t index){
    size_t ix = index%Nx;
    size_t iy = (index/Nx)%Ny;
    size_t iz = (index/(Nx*Ny))%Nz;
    double dx = (extent_max[0]-extent_min[0])/Nx;
    double dy = (extent_max[1]-extent_min[1])/Ny;
    double dz = (extent_max[2]-extent_min[2])/Nz;
    double part_xmin = extent_min[0] + ix*dx;
    double part_ymin = extent_min[1] + iy*dy;
    double part_zmin = extent_min[2] + iz*dz;
    return{ part_xmin, part_ymin, part_zmin, part_xmin+dx, part_ymin+dy, part_zmin+dz };
}

amr_tree_node::amr_tree_node(size_t index, rect3d extent, size_t type, size_t level) {
    node_min[0] = extent.xmin;
    node_min[1] = extent.ymin;
    node_min[2] = extent.zmin;
    node_max[0] = extent.xmax;
    node_max[1] = extent.ymax;
    node_max[2] = extent.zmax;
    node_level = level;
    node_type = type;
    data_index = index;
    node_center[0] = .5*(extent.xmin+extent.xmax);
    node_center[1] = .5*(extent.ymin+extent.ymax);
    node_center[2] = .5*(extent.zmin+extent.zmax);
    for (int i = 0; i<8; i++) {
        child_ptr[i] = nullptr;
    }
}

amr_tree_node::~amr_tree_node(void) {
    for (int i = 0; i < 8; i++) {
        delete child_ptr[i];
        child_ptr[i] = nullptr;
    }
}

size_t amr_tree_node::addnode(size_t index, double x, double y, double z, size_t type, size_t level){
    if (node_level >= level) {
        // Reached desired node depth. Deliver payload.
        if (node_type == -1) {
            node_type = type;
            data_index = index;
            return size_t(0);
        }
        else {
            // Something went horribly wrong: There is already data here!
            throw logic_error("Tried to overwrite existing part of AMR-tree.");
        }
    }
    else {
        // Desired node depth not yet reached. Traverse deeper.
        int child_index = 1*int(x>node_center[0]) + 2*int(y>node_center[1]) + 4*int(z>node_center[2]);
        if (child_ptr[child_index] == nullptr) {
            // Child node does not yet exist. Create it.
            rect3d child_extent = part3dextent(node_min, node_max, 2, 2, 2, child_index);
            child_ptr[child_index] = new amr_tree_node(-1, child_extent, -1, node_level+1);
        }
        return child_ptr[child_index]->addnode(index, x, y, z, type, level);
    }
}

size_t amr_tree_node::findindex(const double x, const double y, const double z){
    if (node_type == 1) {
        // Found a leave. Return the payload.
        return data_index;
    }
    else {
        // This is not a leave. Traverse deeper.
        int child_index = 1*int(x>node_center[0]) + 2*int(y>node_center[1]) + 4*int(z>node_center[2]);
        if (child_ptr[child_index] == nullptr) {
            // Something went horribly wrong: This is not a leave (node_type 1), but there is no child node!
            throw logic_error("Tried to access missing part of AMR-tree.");
        }
        return child_ptr[child_index]->findindex(x, y, z);
    }
}

size_t amr_tree_node::findindex(const double x, const double y, const double z, const size_t maxlevel){
    if (node_type == 1) {
        // Found a leave. Return the payload.
        return data_index;
    }
    else if (node_level >= maxlevel) {
        // Not a leave, but should not traverse deeper. Return payload (may or may not be valid (-1 then)).
        return data_index;
    }
    else {
        // This is not a leave. Traverse deeper.
        int child_index = 1*int(x>node_center[0]) + 2*int(y>node_center[1]) + 4*int(z>node_center[2]);
        if (child_ptr[child_index] == nullptr) {
            // Something went horribly wrong: This is not a leave (node_type 1), but there is no child node!
            throw logic_error("Tried to access missing part of AMR-tree.");
        }
        return child_ptr[child_index]->findindex(x, y, z, maxlevel);
    }
}

amr_tree::amr_tree(rect3d extent, size_t Nx, size_t Ny, size_t Nz){
    rootblockN[0] = Nx;
    rootblockN[1] = Ny;
    rootblockN[2] = Nz;
    tree_min[0] = extent.xmin;
    tree_min[1] = extent.ymin;
    tree_min[2] = extent.zmin;
    tree_max[0] = extent.xmax;
    tree_max[1] = extent.ymax;
    tree_max[2] = extent.zmax;
    rootblock_d_recp[0] = Nx*1./(extent.xmax-extent.xmin);
    rootblock_d_recp[1] = Ny*1./(extent.ymax-extent.ymin);
    rootblock_d_recp[2] = Nz*1./(extent.zmax-extent.zmin);
    
    for (size_t i=0; i<(Nx*Ny*Nz); i++){
        rect3d block_extent = part3dextent(tree_min, tree_max, Nx, Ny, Nz, i);
        rootblock_ptr.push_back(new amr_tree_node(-1, block_extent, -1, 1));
    }
}

amr_tree::~amr_tree(){
    this->flush();
}

size_t amr_tree::addblock(size_t index, double x, double y, double z, size_t node_type, size_t node_level){
    // TODO: As the rootblock_index is only valid for x,y,z (which are provided by user) inside tree_extent,
    //       it may be invalid. Luckily this is fetched through std::vector [] bound checking,
    //       but should be complemented by actually checking the inputs validity!
    size_t rootblock_index = part3dindex(tree_min, rootblock_d_recp, rootblockN, x, y, z);
    return rootblock_ptr[rootblock_index]->addnode(index, x, y, z, node_type, node_level);
}

size_t amr_tree::findblock(double x, double y, double z){
    size_t rootblock_index = part3dindex(tree_min, rootblock_d_recp, rootblockN, x, y, z);
    return rootblock_ptr[rootblock_index]->findindex(x, y, z);
}

size_t amr_tree::findblock(double x, double y, double z, size_t maxlevel){
    size_t rootblock_index = part3dindex(tree_min, rootblock_d_recp, rootblockN, x, y, z);
    return rootblock_ptr[rootblock_index]->findindex(x, y, z, maxlevel);
}

size_t amr_tree::flush(){
    for (size_t i=0; i<(rootblockN[0]*rootblockN[1]*rootblockN[2]); i++){
	    delete rootblock_ptr[i];
        rootblock_ptr[i] = nullptr;
    }
    return 0;
}