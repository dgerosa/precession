'''
To publish on Pypi:
    python setup.py sdist
    python setup.py sdist upload
'''

from setuptools import setup
from pypandoc import convert
    
read_md = lambda f: convert(f, 'rst') # Convert markdown to rst for pypi

# Extract code version from __init__.py 
def get_version():
    with open('precession/__init__.py') as f:
        for line in f.readlines():
            if "__version__" in line:
                return line.split('"')[1]

setup(
    name='precession',
    version=get_version(),
    description='Dynamics of precessing black-hole binaries',
    long_description=read_md('README.md'),
    classifiers=[
        'Topic :: Scientific/Engineering :: Astronomy',
        'Topic :: Scientific/Engineering :: Physics',
        'Topic :: Scientific/Engineering :: Mathematics',
        'Intended Audience :: Science/Research',
        'Programming Language :: Python :: 2.7',
        'Topic :: Software Development :: Libraries :: Python Modules',
    ],
    keywords='black hole spin inspiral precession post-Newtonian',
    url='https://github.com/dgerosa/precession/',
    author='Davide Gerosa',
    author_email='d.gerosa@damtp.cam.ac.uk',
    license='CC by-nc-sa 3.0',
    packages=['precession','precession.test'],
    install_requires=[
          'numpy',
          'scipy',
          'matplotlib',
          'parmap',
      ],
    include_package_data=True,
    zip_safe=False,
)