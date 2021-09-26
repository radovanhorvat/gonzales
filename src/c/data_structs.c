#include <stdio.h>
#include <stdlib.h>
#include <math.h>
#include "data_structs.h"


/* ---------------------------------
	particle functions
------------------------------------ */	

particle* particle_make(double x, double y, double z, double m) {
	particle* p = malloc(sizeof(particle));
	p->x = x;
	p->y = y;
	p->z = z;
	p->a_x = 0.;
	p->a_y = 0.;
	p->a_z = 0.;
	p->m = m;
	return p;
}


void particle_print(particle* p) {
	printf("<particle: r=(%f, %f, %f), m=%f, a=(%f, %f, %f)>\n", p->x, p->y, p->z, p->m, p->a_x, p->a_y, p->a_z);
}


void particles_print(particle** pcont, int n) {
	for (int i = 0; i < n; i++)
		particle_print(pcont[i]);
	printf("\n");
}


double randrange(double min, double max) {
    double range = (max - min); 
    double div = RAND_MAX / range;
    return min + (rand() / div);
}


particle** make_random_particles(int n, double min_r, double max_r, double min_m, double max_m) {
	particle** pcont = malloc(n * sizeof(particle*));
	for (int i = 0; i < n; i++) {
		double x = randrange(min_r, max_r);
		double y = randrange(min_r, max_r);
		double z = randrange(min_r, max_r);
		double m = randrange(min_m, max_m);
		pcont[i] = particle_make(x, y, z, m);
	}
	return pcont;
}


particle** make_slice(particle** pcont, int i1, int i2) {
	int n = i2 - i1;
	particle** pslice = malloc(n * sizeof(particle*));
	for (int i = 0; i < n; i++)
		pslice[i] = pcont[i1 + i];
	return pslice;
}


particle** make_from_arrays(int n, double* points, double* masses) {
	particle** pcont = malloc(n * sizeof(particle*));
	int k;
	for (int i = 0; i < n; i++) {
		k = 3 * i;
		pcont[i] = particle_make(points[k], points[k + 1], points[k + 2], masses[i]);
	}
	return pcont;
}


double* accs_from_pcont(particle** pcont, int n) {
	double* accs = malloc(3 * n * sizeof(double));
	int k;
	for (int i = 0; i < n; i++) {
		k = 3 * i;
		accs[k] = pcont[i]->a_x;
		accs[k + 1] = pcont[i]->a_y;
		accs[k + 2] = pcont[i]->a_z;
	}
	return accs;
}


/* ---------------------------------
	params functions
------------------------------------ */	

params* params_make(double G, double eps, double theta) {
	params* par = malloc(sizeof(params));	
	par->G = G;
	par->eps = eps;
	par->theta = theta;
	return par;
}
