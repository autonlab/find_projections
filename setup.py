import os
import sys
from distutils.core import Extension

from setuptools import setup

home_folder = os.path.expanduser("~")
user_site_packages_folder = "{0}/.local/lib/python{1}.{2}/site-packages".format(home_folder, sys.version_info[0],
                                                                                sys.version_info[1])
if user_site_packages_folder not in sys.path:
    sys.path.append(user_site_packages_folder)

import numpy as np

NAME = 'find_projections'
VERSION = '2.3.1'

REQUIRES = ['numpy >= 1.13']

find_projections_module = Extension('libfind_projections',
                                    sources=['find_projections/binary_tree.cpp', 'find_projections/projection.cpp',
                                             'find_projections/search.cpp', 'find_projections/helper.cpp',
                                             'find_projections/numeric_binary_tree.cpp',
                                             'find_projections/discrete_binary_tree.cpp', 'find_projections/datset.cpp',
                                             'find_projections/pyfind_projections.cpp'],
                                    include_dirs=[np.get_include()],
                                    extra_compile_args=['-pthread', '-std=c++14'],
                                    extra_link_args=['-shared', '-pthread', '-lboost_python3']
                                    )

setup(
    name=NAME,
    version=VERSION,
    url='http://autonlab.org',
    author='Saswati Ray',
    author_email='sray@cs.cmu.edu',
    description='Search for 2-d projection boxes separating out classes/quantiles of output',
    keywords='d3m_primitive',
    license='MIT',
    ext_modules=[find_projections_module],
    packages=['find_projections'],
    entry_points={
        'd3m.primitives': [
            'classification.search.Find_projections = find_projections:Search',
            'regression.search_numeric.Find_projections = find_projections:SearchNumeric',
            'classification.search_hybrid.Find_projections = find_projections:SearchHybrid',
            'regression.search_hybrid_numeric.Find_projections = find_projections:SearchHybridNumeric',
        ]
    }
)
