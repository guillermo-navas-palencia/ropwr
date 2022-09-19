#!/usr/bin/env python

import os
import sys

from setuptools import find_packages, setup, Command
from setuptools.command.test import test as TestCommand


long_description = '''
**RoPWR** is a library written in Python implementing several mathematical
programming formulations to compute the optimal continuous/discontinuous
piecewise polynomial regression given a list of split points.

Read the documentation at: http://gnpalencia.org/ropwr/

RoPWR is distributed under the Apache Software License (Apache 2.0).
'''


class CleanCommand(Command):
    user_options = []

    def initialize_options(self):
        pass

    def finalize_options(self):
        pass

    def run(self):
        os.system('rm -vrf ./build ./dist ./*.pyc ./*.tgz ./*.egg-info')


# test suites
class PyTest(TestCommand):
    def finalize_options(self):
        TestCommand.finalize_options(self)
        self.test_args = []
        self.test_suite = []

    def run_tests(self):
        # import here, because outside the eggs aren't loaded
        import pytest
        errcode = pytest.main(self.test_args)
        sys.exit(errcode)


# install requirements
install_requires = [
    'cvxpy>=1.1.14',
    'numpy>=1.16',
    'scikit-learn>=0.22',
    'scipy>=1.6.1'
]

# test requirements
tests_require = [
    'pytest',
    'coverage'
]


# Read version file
version_info = {}
with open("ropwr/_version.py") as f:
    exec(f.read(), version_info)


setup(
    name="ropwr",
    version=version_info['__version__'],
    description="RoPWR: Robust Piecewise Regression",
    long_description=long_description,
    author="Guillermo Navas-Palencia",
    author_email="g.navas.palencia@gmail.com",
    packages=find_packages(),
    platforms="any",
    include_package_data=True,
    license="Apache Licence 2.0",
    url="https://github.com/guillermo-navas-palencia/ropwr",
    cmdclass={'clean': CleanCommand, 'test': PyTest},
    python_requires='>=3.7',
    install_requires=install_requires,
    tests_require=tests_require,
    classifiers=[
        'Topic :: Scientific/Engineering :: Mathematics',
        'Topic :: Software Development :: Libraries',
        'Topic :: Software Development :: Libraries :: Python Modules',
        'Intended Audience :: Developers',
        'Intended Audience :: Education',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: Apache Software License',
        'Programming Language :: Python :: 3']
    )
