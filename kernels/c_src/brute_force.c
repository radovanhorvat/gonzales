#include <math.h>
#include <stdio.h>
#include "brute_force.h"
#include "data_structs.h"


void calc_accs(particle** pcont, int n, params* par) {
    // pairwise interaction of all particles in pcont
    double k1, k2, d, f, dx, dy, dz;
    for (int i = 0; i < n; ++i) {        
        k2 = pcont[i]->m * par->G;
        for (int j = i + 1; j < n; ++j) {
            k1 = pcont[j]->m * par->G;
            dx = pcont[j]->x - pcont[i]->x;
            dy = pcont[j]->y - pcont[i]->y;
            dz = pcont[j]->z - pcont[i]->z;
            d = sqrt(dx * dx + dy * dy + dz * dz);
        	f = 1.0 / (d * d * d + par->eps);
            pcont[i]->a_x += f * k1 * dx;
            pcont[i]->a_y += f * k1 * dy;
            pcont[i]->a_z += f * k1 * dz;
            pcont[j]->a_x -= f * k2 * dx;
            pcont[j]->a_y -= f * k2 * dy;
            pcont[j]->a_z -= f * k2 * dz;
        }
    }
}


void calc_accs_particles(particle* p1, particle* p2, params* par) {
    // acceleration on particle p1 due to p2
    if (p1 == p2)
        return;
    double k1, d, f, dx, dy, dz;
    dx = p2->x - p1->x;
    dy = p2->y - p1->y;
    dz = p2->z - p1->z;
    d = sqrt(dx * dx + dy * dy + dz * dz);
    f = 1.0 / (d * d * d + par->eps);
    k1 = p2->m * par->G;
    p1->a_x += f * k1 * dx;
    p1->a_y += f * k1 * dy;
    p1->a_z += f * k1 * dz;
}
