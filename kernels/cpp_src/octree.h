#include <iostream>
#include <cmath>
#include <vector>
#include "octnode.h"


namespace octree {


class Octree {
	public:
		OctNode* root_node;
		double* points;
		double* masses;
		double* accs;
		int n;
		int num_nodes;
		int num_leaves;
		double a_x, a_y, a_z, G, eps, theta;
		short int child_signs[8][3] = {{-1, -1, -1}, {-1, -1, 1}, {-1, 1, -1}, {-1, 1, 1},
		                               {1, -1, -1}, {1, -1, 1}, {1, 1, -1}, {1, 1, 1}};

		Octree(double x, double y, double z, double w, double u_G, double u_eps, double u_theta, double* points,
		       double* masses, int n_points);
		~Octree();

        void build();
		void calculate_accs();
		void calculate_accs_st();

		void insert_particle(OctNode* nd, double r_x, double r_y, double r_z, double mass);
		void traverse_tree(OctNode* nd, double r_x, double r_y, double r_z, double mass, int j);

		void traverse_tree_st(OctNode* nd, std::vector<int>& indices);

};

}