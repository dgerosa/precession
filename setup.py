import os
import precession

from setuptools import setup

def readme():
    with open('README.rst') as f:
        return f.read()


setup(
    name='precession',
    version=0,#precession.__version__,
    description='Dynamics of precessing black-hole binaries',
    long_description=readme(),
    classifiers=[
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='black hole spin inspiral precession post-Newtonian',
    url='http://davidegerosa.com/spinprecession/',
    author='Davide Gerosa',
    author_email='d.gerosa@damtp.cam.ac.uk',
    license='CC by-nc-sa 3.0',
    packages=['precession'],
    install_requires=[
          'numpy',
          'scipy',
          'parmap',
      ],
    include_package_data=True,
    zip_safe=False,
)