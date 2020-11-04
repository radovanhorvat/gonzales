#pragma once
#include <math.h>
#include <omp.h>
#include "octree.h"

namespace octree {

Octree::Octree(double x, double y, double z, double w, double u_G, double u_eps, double u_theta, double* u_points,
               double* u_masses, int n_points) {
	root_node = new OctNode(w, x, y, z);
	points = u_points;
	masses = u_masses;
	n = n_points;
	accs = new double[3 * n]();
	num_nodes = 1;
	num_leaves = 0;
	a_x = 0;
	a_y = 0;
	a_z = 0;
	G = u_G;
	eps = u_eps;
	theta = u_theta;
}

Octree::~Octree() {
    delete root_node;
}

void Octree::insert_particle(OctNode* nd, double r_x, double r_y, double r_z, double mass) {
    if (nd->is_empty == true) {
        nd->m = mass;
        nd->R_x = r_x;
        nd->R_y = r_y;
        nd->R_z = r_z;
        nd->is_empty = false;
        nd->is_leaf = true;
        num_leaves += 1;
        return;
    }
    if (nd->is_leaf == true) {
        nd->is_leaf = false;
        num_leaves -= 1;
        int child_id = nd->get_child_id(nd->R_x, nd->R_y, nd->R_z);
        double f1 = 0.25 * nd->w;
        double cc_x = nd->c_x + f1 * child_signs[child_id][0];
        double cc_y = nd->c_y + f1 * child_signs[child_id][1];
        double cc_z = nd->c_z + f1 * child_signs[child_id][2];
        double c_w = 0.5 * nd->w;
        nd->children[child_id] = new OctNode(c_w, cc_x, cc_y, cc_z);
        num_nodes += 1;
        insert_particle(nd->children[child_id], nd->R_x, nd->R_y, nd->R_z, nd->m);
    }
    double k1 = mass / (nd->m + mass);
    double k2 = nd->m / (nd->m + mass);
    nd->R_x = k1 * r_x + k2 * nd->R_x;
    nd->R_y = k1 * r_y + k2 * nd->R_y;
    nd->R_z = k1 * r_z + k2 * nd->R_z;
    nd->m += mass;
    int child_id = nd->get_child_id(r_x, r_y, r_z);
    double f1 = 0.25 * nd->w;
    if (nd->children[child_id] == NULL) {
        double cc_x = nd->c_x + f1 * child_signs[child_id][0];
        double cc_y = nd->c_y + f1 * child_signs[child_id][1];
        double cc_z = nd->c_z + f1 * child_signs[child_id][2];
        double c_w = 0.5 * nd->w;
        nd->children[child_id] = new OctNode(c_w, cc_x, cc_y, cc_z);
        num_nodes += 1;
    }
    insert_particle(nd->children[child_id], r_x, r_y, r_z, mass);
}


void Octree::traverse_tree(OctNode* nd, double r_x, double r_y, double r_z, double mass, int j) {
    double dist = nd->distance(r_x, r_y, r_z, nd->R_x, nd->R_y, nd->R_z);
    if (dist == 0)
        return;
    double dx = nd->R_x - r_x;
    double dy = nd->R_y - r_y;
    double dz = nd->R_z - r_z;
    double f = G * nd->m / (dist * dist * dist + eps);
    if (nd->is_leaf == true) {
        accs[3 * j] += f * dx;
        accs[3 * j + 1] += f * dy;
        accs[3 * j + 2] += f * dz;
        return;
    }
    if (nd->w / dist < theta) {
        accs[3 * j] += f * dx;
        accs[3 * j + 1] += f * dy;
        accs[3 * j + 2] += f * dz;
        return;
    }
    for (int i = 0; i < 8; ++i) {
        if (nd->children[i] == NULL)
            continue;
        traverse_tree(nd->children[i], r_x, r_y, r_z, mass, j);
    }
}

void Octree::calculate_accs() {
    int k;
    for (int i = 0; i < n; ++i) {
        k = 3 * i;
        traverse_tree(root_node, points[k], points[k + 1], points[k + 2], masses[i], i);
    }
}

void Octree::calculate_accs_st() {
    std::vector<int> indices(n);
    for (int i = 0; i < indices.size(); ++i)
        indices[i] = i;
    traverse_tree_st(root_node, indices);
}

void Octree::calculate_accs_st_parallel() {
    int n_cores = omp_get_max_threads();
    int chunk_size = n / n_cores;
    # pragma omp parallel num_threads(n_cores)
    {
        int tid, start_ind, end_ind, vec_size;
        tid = omp_get_thread_num();
        start_ind = tid * chunk_size;
        end_ind = (tid < n_cores - 1) ? (tid + 1) * chunk_size : n;
        vec_size = end_ind - start_ind;
        std::vector<int> indices(vec_size);
        for (int i = 0; i < indices.size(); ++i)
            indices[i] = start_ind + i;
        traverse_tree_st(root_node, indices);
    }
}

void Octree::traverse_tree_st(OctNode* nd, std::vector<int>& indices) {
    if (indices.size() == 0)
        return;
    double r_x, r_y, r_z, dx, dy, dz, dist, f;
    if (nd->is_leaf == true) {
        for(const int& i : indices) {
            r_x = points[3 * i];
            r_y = points[3 * i + 1];
            r_z = points[3 * i + 2];
            dx = nd->R_x - r_x;
            dy = nd->R_y - r_y;
            dz = nd->R_z - r_z;
            dist = sqrt(dx * dx + dy * dy + dz * dz);
            f = G * nd->m / (dist * dist * dist + eps);
            accs[3 * i] += f * dx;
            accs[3 * i + 1] += f * dy;
            accs[3 * i + 2] += f * dz;
        }
        return;
    }
    std::vector<int> new_indices(indices.size());
    int k = 0;
    for(const int& i : indices) {
        r_x = points[3 * i];
        r_y = points[3 * i + 1];
        r_z = points[3 * i + 2];
        dx = nd->R_x - r_x;
        dy = nd->R_y - r_y;
        dz = nd->R_z - r_z;
        dist = sqrt(dx * dx + dy * dy + dz * dz);
        // case 1: MAC satisfied
        if (nd->w / dist < theta) {
            f = G * nd->m / (dist * dist * dist + eps);
            accs[3 * i] += f * dx;
            accs[3 * i + 1] += f * dy;
            accs[3 * i + 2] += f * dz;
        }
        // case 2: MAC not satisfied
        else {
            new_indices[k] = i;
            k++;
        }
    }
    new_indices.resize(k);
    for (int i = 0; i < 8; ++i) {
        if (nd->children[i] == NULL)
            continue;
        traverse_tree_st(nd->children[i], new_indices);
    }

}

void Octree::build() {
    int k;
    for (int i = 0; i < n; ++i) {
        k = 3 * i;
        insert_particle(root_node, points[k], points[k + 1], points[k + 2], masses[i]);
    }
}

}