#!/usr/bin/env python3

import os
from setuptools import setup, find_packages
setup(
    name='Bandit-vnnlib-fuzz',
    version='0.1',
    description='A Bandit vnnlib fuzzer',
    author='Guanting Pan',
    author_email='g6pan@uwaterloo.ca',
    url='https://github.com/MapleDNNSAT/Bandit-VNN-LIB-Fuzz',
    scripts=['bin/random_fuzzer'],
    packages=find_packages(),
    package_dir={
        'vnnlib_fuzzer': 'vnnlib_fuzzer',
    },
)
