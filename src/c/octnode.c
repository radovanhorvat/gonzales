#include <stdlib.h>
#include <stdio.h>
#include <math.h>
#include <omp.h>
#include "octnode.h"
#include "brute_force.h"


#define LEAF_SIZE 8
const short int child_signs[8][3] = {{-1, -1, -1},{-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
		                               {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}};


octnode* octnode_make(double w, double x, double y, double z) {
	octnode* nd = malloc(sizeof(octnode));
	nd->w = w;
	nd->m = 0.;
	nd->r_x = x;
	nd->r_y = y;
	nd->r_z = z;
	nd->R_x = 0.;
	nd->R_y = 0.;
	nd->R_z = 0.;
	nd->leaf_cap = LEAF_SIZE;
	nd->pnts = malloc(LEAF_SIZE * sizeof(particle*));	
	for (int i = 0; i < LEAF_SIZE; i++)
		nd->pnts[i] = NULL;
	for (int i = 0; i < 8; i++)
		nd->children[i] = NULL;
	return nd;
}

void octnode_print(octnode* nd) {
	printf("<octnode: %f, %f, %f [%f, %d]>\n", nd->r_x, nd->r_y, nd->r_z, nd->w, nd->leaf_cap);
}


void octree_print(octnode* root) {
	octnode_print(root);
	for (int i = 0; i < 8; i++) {
		if (root->children[i] == NULL)
			continue;
		octree_print(root->children[i]);
	}
}


void octnode_insert_point(octnode* nd, particle* pnt) {
	if (nd->leaf_cap > 0) {
		nd->pnts[LEAF_SIZE - nd->leaf_cap] = pnt;
		nd->leaf_cap -= 1;
		nd->m += pnt->m;
		return;
	}
	else if (nd->leaf_cap == 0) {
		int n = LEAF_SIZE - nd->leaf_cap;
		for (int i = 0; i < n; i++) {
			int cid = octnode_get_child_id(nd, nd->pnts[i]->x, nd->pnts[i]->y, nd->pnts[i]->z);
			if (nd->children[cid] == NULL) {
				octnode_make_child(nd, cid);			
			}
			octnode_insert_point(nd->children[cid], nd->pnts[i]);
			nd->R_x += nd->pnts[i]->x * nd->pnts[i]->m;
			nd->R_y += nd->pnts[i]->y * nd->pnts[i]->m;
			nd->R_z += nd->pnts[i]->z * nd->pnts[i]->m;
		}
		double ff = 1. / nd->m;
		nd->R_x *= ff;
		nd->R_y *= ff;
		nd->R_z *= ff;
		nd->leaf_cap -= 1;
		free(nd->pnts);
	}
	if (nd->leaf_cap == -1) {
		double pmass = pnt->m;
	    double k1 = pmass / (nd->m + pmass);
	    double k2 = nd->m / (nd->m + pmass);
		nd->R_x *= k2;
		nd->R_y *= k2;
		nd->R_z *= k2;
		nd->R_x += pnt->x * k1;
		nd->R_y += pnt->y * k1;
		nd->R_z += pnt->z * k1;
		nd->m += pmass;
		int cid = octnode_get_child_id(nd, pnt->x, pnt->y, pnt->z);
		if (nd->children[cid] == NULL) {
			octnode_make_child(nd, cid);			
		}
		octnode_insert_point(nd->children[cid], pnt);
	}
}


int octnode_get_child_id(octnode* nd, double x, double y, double z) {
	int idx = (x >= nd->r_x) << 2 | (y >= nd->r_y) << 1 | (z >= nd->r_z);
	return idx;
}


void octnode_make_child(octnode* nd, int idx) {
	double k = 0.25 * nd->w;
	double x = nd->r_x + k * child_signs[idx][0];
	double y = nd->r_y + k * child_signs[idx][1];
	double z = nd->r_z + k * child_signs[idx][2];
	octnode* child = octnode_make(0.5 * nd->w, x, y, z);
	nd->children[idx] = child;
}


void octree_build(octnode* root, particle** pcont, int n) {	
	for (int i = 0; i < n; ++i) {
		octnode_insert_point(root, pcont[i]);
	}
}


void octree_build_omp(octnode* root, particle** pcont, int n) {	
	if (n <= LEAF_SIZE) {
		octree_build(root, pcont, n);
		return;
	}
	int max_threads = omp_get_max_threads();
	int n_cores = (max_threads <= 8) ? max_threads : 8;
	int* child_ids = malloc(n * sizeof(int));
	int** thread_assign = malloc(n_cores * sizeof(int*));
	int* thread_assign_cnt = malloc(n_cores * sizeof(int));
	for (int i = 0; i < n_cores; i++) {
		thread_assign[i] = malloc(n * sizeof(int));
		thread_assign_cnt[i] = 0;
	}

	for (int i = 0; i < n; i++) {
		int cid = octnode_get_child_id(root, pcont[i]->x, pcont[i]->y, pcont[i]->z);
		if (root->children[cid] == NULL) {
			octnode_make_child(root, cid);			
		}
		child_ids[i] = cid;
		int tid = cid % n_cores;
		thread_assign[tid][thread_assign_cnt[tid]] = i;
		thread_assign_cnt[tid] += 1;

	}
	#pragma omp parallel num_threads(n_cores)
	{
		int tid = omp_get_thread_num();
		int k = thread_assign_cnt[tid];
		int* t_arr = thread_assign[tid];		
		for (int i = 0; i < k; i++) {
			int pid = t_arr[i];
			octnode_insert_point(root->children[child_ids[pid]], pcont[pid]);
		}

	}
	root->leaf_cap = -1;
	for (int i = 0; i < 8; i++) {
		if (root->children[i] == NULL)
			continue;
		root->m += root->children[i]->m;
		root->R_x += root->children[i]->R_x * root->children[i]->m;
		root->R_y += root->children[i]->R_y * root->children[i]->m;
		root->R_z += root->children[i]->R_z * root->children[i]->m;
	}
	double ff = 1. / root->m;
	root->R_x *= ff;
	root->R_y *= ff;
	root->R_z *= ff;

}


void octree_calc_accs(octnode* nd, particle** psub, int k, params* par) {
	if (k == 0)	
		return;
	double d, f, mac, dx, dy, dz, d_squared, mac_squared;
	int cnt;
	if (nd->leaf_cap >= 0) {
		int n = LEAF_SIZE - nd->leaf_cap;
		for (int i = 0; i < k; i++) {
			for (int j = 0; j < n; j++) {
				calc_accs_particles(psub[i], nd->pnts[j], par);
			}			
		}
		return;
	}
    cnt = 0;	
	mac = nd->w / par->theta;
	mac_squared = mac * mac;
	particle** new_psub = malloc(k * sizeof(particle*));
	for (int i = 0; i < k; ++i) {
		particle* part = psub[i];
		dx = nd->R_x - part->x;
		dy = nd->R_y - part->y;
		dz = nd->R_z - part->z;
		d_squared = dx * dx + dy * dy + dz * dz;
        // case 1: MAC satisfied
        if (d_squared > mac_squared) {
            d = sqrt(d_squared);
            f = par->G * nd->m / (d_squared * d + par->eps);
            part->a_x += f * dx;
            part->a_y += f * dy;
            part->a_z += f * dz;
        }
        // case 2: MAC not satisfied
        else {
            new_psub[cnt] = part;
            cnt++;
        }
	}
    for (int i = 0; i < 8; ++i) {
		if (nd->children[i] == NULL)
		    continue;
		octree_calc_accs(nd->children[i], new_psub, cnt, par);
	}
	free(new_psub);
	
}

void octree_calc_accs_omp(octnode* nd, particle** psub, int k, params* par) {
	if (k <= LEAF_SIZE) {
		calc_accs(psub, k, par);
		return;
	}       
	int n_cores = omp_get_max_threads();
    int chunk_size = k / n_cores;
    # pragma omp parallel num_threads(n_cores)
    {
        int tid, start_ind, end_ind, m;
        tid = omp_get_thread_num();
        start_ind = tid * chunk_size;
        end_ind = (tid < n_cores - 1) ? (tid + 1) * chunk_size : k;
        m = end_ind - start_ind;        
        particle** new_psub = make_slice(psub, start_ind, end_ind);
        octree_calc_accs(nd, new_psub, m, par);
        free(new_psub);
    }
}


double* calc_accs_wrap(int n, double* points, double* masses, double G, double eps, double theta, double root_width, double root_x, double root_y, double root_z) {
	particle** pcont = make_from_arrays(n, points, masses);
	octnode* root = octnode_make(root_width, root_x, root_y, root_z);
	params* par = params_make(G, eps, theta);
	octree_build_omp(root, pcont, n);
	octree_calc_accs_omp(root, pcont, n, par);
	return accs_from_pcont(pcont, n);
}
