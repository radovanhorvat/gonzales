#pragma once
#include "octnode.h"


OctNode::OctNode(double width, double center_x, double center_y, double center_z) {
	w = width;
	c_x = center_x;
	c_y = center_y;
	c_z = center_z;
	is_empty = true;
	is_leaf = false;
	m = 0;
	R_x = 0;
	R_y = 0;
	R_z = 0;
}

OctNode::~OctNode() {
    for (int i = 0; i < 8; i++) {
        delete children[i];
    }
}

double OctNode::distance(double x1, double y1, double z1, double x2, double y2, double z2) {
    double dx = x2 - x1;
    double dy = y2 - y1;
    double dz = z2 - z1;
    return sqrt(dx * dx + dy * dy + dz * dz);
}


int OctNode::get_child_id(double r_x, double r_y, double r_z) {
	int index = (r_x >= c_x) << 2 | (r_y >= c_y) << 1 | (r_z >= c_z);
	return index;
}

