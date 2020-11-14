import numpy as np

import kernels.numeric as kernum


def test_com():
    # test center of mass calculation
    # 1. case
    r = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 3.])
    com = kernum.calc_com_wrap(r, m)
    np.testing.assert_equal(com, np.array([5/6., 0.5, 0.5]))
    # 2. case - one of the masses dominates
    r = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 1.0e15])
    com = kernum.calc_com_wrap(r, m)
    np.testing.assert_almost_equal(com, np.array([1., 1., 1.]))


def test_ke():
    # test kinetic energy calculation
    # 1. trivial case
    v = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    m = np.array([1., 2., 3.])
    ke = kernum.calc_ke_wrap(v, m)
    np.testing.assert_equal(ke, 0.)
    # 2. other cases
    v = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 3.])
    ke = kernum.calc_ke_wrap(v, m)
    np.testing.assert_equal(ke, 11 / 2.)


def test_pe():
    # test potential energy calculation
    # 1. case
    r = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 3.])
    pe = kernum.calc_pe_wrap(r, m, 1.0, 0.)
    np.testing.assert_equal(pe, - (2. + 3 / np.sqrt(3) + 6 / np.sqrt(2)))
    # 2. case - particles at huge distances
    r = np.array([[0., 0., 0.], [1.0e15, 0., 0.], [1., 1.0e15, 1.]])
    m = np.array([1., 2., 3.])
    pe = kernum.calc_pe_wrap(r, m, 1.0, 0.)
    np.testing.assert_almost_equal(pe, 0.)


def test_te():
    # test total energy calculation
    r = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    v = np.array([[-1., 1., 0.], [1., -1., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 3.])
    pe = kernum.calc_pe_wrap(r, m, 1.0, 0.)
    ke = kernum.calc_ke_wrap(v, m)
    te = kernum.calc_te_wrap(r, v, m, 1.0, 0.)
    np.testing.assert_equal(pe + ke, te)


def test_ang_mom():
    # test angular momentum
    r = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    v = np.array([[-1., 1., 0.], [1., -1., 0.], [1., -2., 1.]])
    m = np.array([1., 2., 3.])
    am = kernum.calc_ang_mom_wrap(r, v, m)
    np.testing.assert_equal(am, np.array([9., 0., -11.]))
