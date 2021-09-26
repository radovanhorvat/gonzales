#ifndef OCTNODE_H
#define OCTNODE_H

#include "data_structs.h"


typedef struct octnode_t {
	double w, m, r_x, r_y, r_z, R_x, R_y, R_z; // width, mass, pos, com
	int leaf_cap; // leaf capacity
	particle** pnts; // points
	struct octnode_t* children[8];

} octnode;


octnode* octnode_make(double w, double x, double y, double z);

void octnode_print(octnode* nd);

void octree_print(octnode* root);

void octnode_insert_point(octnode* nd, particle* pnt);

int octnode_get_child_id(octnode* nd, double x, double y, double z);

void octnode_make_child(octnode* nd, int idx);

void octree_build(octnode* root, particle** pcont, int n);

void octree_build_omp(octnode* root, particle** pcont, int n);

void octree_calc_accs(octnode* nd, particle** psub, int k, params* par);

void octree_calc_accs_omp(octnode* nd, particle** psub, int k, params* par);

double* calc_accs_wrap(int n, double* points, double* masses, double G, double eps, double theta, double root_width, double root_x, double root_y, double root_z);


#endif