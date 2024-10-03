from setuptools import setup, find_packages

VERSION = '0.1.0'
DESCRIPTION = 'Companion code for https://doi.org/10.1101/2024.09.22.614271'

# Setting up
setup(
    name="Creamer_LDS_2024",
    version=VERSION,
    author="Matthew S. Creamer",
    author_email="matthew.s.creamer@gmail.com",
    description=DESCRIPTION,
    packages=find_packages(),
    install_requires=['numpy',
                      'matplotlib',
                      'scipy',
                      'mpi4py',
                      'PyYAML',
                      'simple-slurm',
                      ],

    keywords=['python', 'calcium', 'ssm', 'linear', 'dynamical', 'system', 'lds'],
    classifiers=[
        "Programming Language :: Python :: 3",
    ]
)
