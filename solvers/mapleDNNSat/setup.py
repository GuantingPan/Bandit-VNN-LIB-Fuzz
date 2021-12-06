#!/usr/bin/env python3

import os
from setuptools import setup, find_packages
setup(
    name='mapleDNNsat',
    version='0.1',
    description='A vnnlib sat solver',
    author='Joe Scott',
    author_email='joseph.scott@uwaterloo.ca',
    url='https://github.com/j29scott/bookish-enigma',
    scripts=[
        'bin/mapleDNNsat',
    ],
    packages=find_packages(),
    package_dir={
        'mapleDNNsat': 'mapleDNNsat',
    },
    install_requires=[
        'scikit-learn',
        'func_timeout',
    ],
)
