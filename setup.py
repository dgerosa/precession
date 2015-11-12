'''
To publish on Pypi type
    python setup.py sdist
    python setup.py sdist upload
'''

from setuptools import setup

# Fill "long_description" using the README file
def readme():
    with open('README.rst') as f:
        return f.read()
        
# Extract code version from __init__.py 
def get_version():
    with open('precession/__init__.py') as f:
        for line in f.readlines():
            if "version" in line:
                return line.split('"')[1]

setup(
    name='precession',
    version=get_version(),
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
    packages=['precession','precession.tutorial'],
    install_requires=[
          'numpy',
          'scipy',
          'parmap',
      ],
    include_package_data=True,
    zip_safe=False,
)