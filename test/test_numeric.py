import numpy as np

from simulator.space import Space
from simulator.utils import calculate_relative_error
import kernels.numeric as kernum


def test_com():
    # test center of mass calculation
    r = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 3.])
    com = kernum.calc_com_wrap(r, m)
    expected = np.array([5/6., 0.5, 0.5])
    np.testing.assert_equal(com, expected)


def test_ke():
    # test kinetic energy calculation
    v = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 3.])
    ke = kernum.calc_ke_wrap(v, m)
    assert ke == 11 / 2.


def test_pe():
    # test potential energy calculation
    r = np.array([[0., 0., 0.], [1., 0., 0.], [1., 1., 1.]])
    m = np.array([1., 2., 3.])
    pe = kernum.calc_pe_wrap(r, m, 1.0, 0.)
    assert pe == - (2. + 3 / np.sqrt(3) + 6 / np.sqrt(2))
