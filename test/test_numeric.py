import numpy as np

import gonzales.lib.physics as phy


def test_com():
    # test center of mass calculation
    # 1. case
    r = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 3.])
    com = phy.calc_com(r, m)
    np.testing.assert_almost_equal(com, np.array([5/6., 0.5, 0.5]))
    # 2. case - one of the masses dominates
    r = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 1.0e15])
    com = phy.calc_com(r, m)
    np.testing.assert_almost_equal(com, np.array([1., 1., 1.]))


def test_ke():
    # test kinetic energy calculation
    # 1. trivial case
    v = np.array([[0., 0., 0.], [0., 0., 0.], [0., 0., 0.]])
    m = np.array([1., 2., 3.])
    ke = phy.calc_ke(v, m)
    np.testing.assert_equal(ke, 0.)
    # 2. other cases
    v = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 3.])
    ke = phy.calc_ke(v, m)
    np.testing.assert_equal(ke, 11 / 2.)


def test_pe():
    # test potential energy calculation
    # 1. case
    r = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 3.])
    pe = phy.calc_pe(r, m, 1.0, 0.)
    np.testing.assert_equal(pe, - (2. + 3 / np.sqrt(3) + 6 / np.sqrt(2)))
    # 2. case - particles at huge distances
    r = np.array([[0., 0., 0.], [1.0e15, 0., 0.], [1., 1.0e15, 1.]])
    m = np.array([1., 2., 3.])
    pe = phy.calc_pe(r, m, 1.0, 0.)
    np.testing.assert_almost_equal(pe, 0.)


def test_te():
    # test total energy calculation
    r = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    v = np.array([[-1., 1., 0.], [1., -1., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 3.])
    pe = phy.calc_pe(r, m, 1.0, 0.)
    ke = phy.calc_ke(v, m)
    te = phy.calc_te(r, v, m, 1.0, 0.)
    np.testing.assert_equal(pe + ke, te)


def test_ang_mom():
    # test angular momentum
    r = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    v = np.array([[-1., 1., 0.], [1., -1., 0.], [1., -2., 1.]])
    m = np.array([1., 2., 3.])
    am = phy.calc_ang_mom(r, v, m)
    np.testing.assert_equal(am, np.array([9., 0., -11.]))
