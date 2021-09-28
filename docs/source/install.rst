Installation
============

`gonzales` is released as a source distribution, which means that the C and Cython extensions will
be built during the installation process. Hence, a C compiler (typically GCC for Unix-based systems or Microsoft
Visual Studio for Windows) should be available. Other installation and setup dependencies shall be handled by
``pip``.

Prior to installation, it is strongly recommended to create and activate a virtual environment.


.. rubric:: Latest release

To install the latest release from https://pypi.org/project/gonzales/, simply install with:

``pip install gonzales``

.. rubric:: Development version

To install the current development version from https://github.com/radovanhorvat/gonzales, do:

``pip install git+https://github.com/radovanhorvat/gonzales.git``

.. rubric:: For developers

If you plan to experiment with the code, you should:

1. Clone the repository with ``git clone https://github.com/radovanhorvat/gonzales.git``
2. Create and activate a virtual environment
3. Run ``pip install -e .``

If extensions need to be re-compiled, this should be done with ``python setup.py build_ext --build-lib=src/python``.
