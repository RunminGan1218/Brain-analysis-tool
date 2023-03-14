#!/usr/bin/env python
from __future__ import print_function
from setuptools import setup, find_packages
import sys
 
setup(
    name="brainanalysistool",
    version="0.0.3",
    author="Runmin Gan",
    author_email="354613146@qq.com",
    description="Ecog analysis tool",
    long_description=open("README.md").read(),
    license="MIT",
    url="https://github.com/RunminGan1218/Brain-analysis-tool",
    packages=['brainanalysistool'],
    install_requires=[
        "matplotlib>=3.6.2",
        "numpy>=1.23.5",
        "PyWavelets>=1.4.1",
        "scipy>=1.9.3",
        "pactools>= 0.3.1",
        "pycwt>=0.3.0a22"
        ],
    classifiers=[
        "Intended Audience :: Developers",
        "Programming Language :: Python :: 3",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
    ],
)