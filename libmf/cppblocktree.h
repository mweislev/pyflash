//
//  blocktree.h
//
//  Created by Michael Weis on 10.09.16.
//

#ifndef __blocktree__
#define __blocktree__

#include <vector>

struct rect3d{
    double xmin, ymin, zmin;
    double xmax, ymax, zmax;
};

class amr_tree_node{
public:
    amr_tree_node(size_t index, rect3d extent, size_t type, size_t level);
    ~amr_tree_node();
    size_t addnode(size_t index, double x, double y, double z, size_t type, size_t level);
    size_t findindex(const double x, const double y, const double z);
    size_t findindex(const double x, const double y, const double z, const size_t maxlevel);
private:
    amr_tree_node *child_ptr[8];
    size_t data_index;
    size_t node_level, node_type;
    double node_min[3];
    double node_max[3];
    double node_center[3];
};

class amr_tree{
public:
    amr_tree(rect3d extent, size_t Nx, size_t Ny, size_t Nz);
    ~amr_tree();
    size_t addblock(size_t index, double x, double y, double z, size_t node_type, size_t node_level);
    size_t findblock(double x, double y, double z);
    size_t findblock(double x, double y, double z, size_t maxlevel);
	size_t flush();
private:
    std::vector<amr_tree_node*> rootblock_ptr;
    size_t rootblockN[3];
    double tree_min[3];
    double tree_max[3];
    double rootblock_d_recp[3];
};


#endif