#include <iostream>
#include <cmath>


class OctNode {
	public:
		bool is_empty;
		bool is_leaf;
		double w, m, c_x, c_y, c_z, R_x, R_y, R_z;
		OctNode* children[8] = { NULL };
		OctNode(double width, double center_x, double center_y, double center_z);
		~OctNode();

		int get_child_id(double r_x, double r_y, double r_z);
		double distance(double x1, double y1, double z1, double x2, double y2, double z2);

};
