#include <math.h>

namespace bfcpp {

double* accs_bf(double* points, double* masses, int n, double G, double eps) {
    double* accs = new double[3 * n]();
    double k1, k2, dx, dy, dz, ds, d, f;
    for (int i = 0; i < n; ++i) {
        k1 = masses[i] * G;
        for (int j = i + 1; j < n; ++j) {
            k2 = masses[j] * G;
            dx = points[3 * j] - points[3 * i];
            dy = points[3 * j + 1] - points[3 * i + 1];
            dz = points[3 * j + 2] - points[3 * i + 2];
            ds = dx * dx + dy * dy + dz * dz;
            d = sqrt(ds);
            f = 1.0 / (d * d * d + eps);
            accs[3 * i] += dx * f * k1;
            accs[3 * i + 1] += dy * f * k1;
            accs[3 * i + 2] += dz * f * k1;
            accs[3 * j] -= dx * f * k2;
            accs[3 * j + 1] -= dy * f * k2;
            accs[3 * j + 2] -= dz * f * k2;
        }
    }
    return accs;
}

}