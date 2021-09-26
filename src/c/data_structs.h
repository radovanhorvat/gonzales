#ifndef DATA_STRUCTS_H
#define DATA_STRUCTS_H


/* ---------------------------------
	particle
------------------------------------ */

typedef struct {
	double x, y, z, a_x, a_y, a_z, m;

} particle;


particle* particle_make(double x, double y, double z, double m);

void particle_print(particle* p);

void particles_print(particle** pcont, int n);

double randrange(double min, double max);

particle** make_random_particles(int n, double min_r, double max_r, double min_m, double max_m);

particle** make_slice(particle** pcont, int i1, int i2);

particle** make_from_arrays(int n, double* points, double* masses);

double* accs_from_pcont(particle** pcont, int n);


/* ---------------------------------
	params
------------------------------------ */

typedef struct {
	double G, eps, theta;

} params;


params* params_make(double G, double eps, double theta);


#endif