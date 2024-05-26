#!/usr/bin/env python

#from pip.download import PipSession
#from pip.req import parse_requirements
from setuptools import setup, find_packages
import os

# Load requirements
#install_reqs = parse_requirements('requirements.txt', session=PipSession())
#required = [str(ir.req) for ir in install_reqs]
#print(required)

with open('requirements.txt') as f:
    required = f.read().splitlines()

# Load scripts
scripts = [os.path.join('metatoolkit', f) for f in os.listdir('metatoolkit')
         if (os.path.isfile(os.path.join('metatoolkit', f))) & (f[0] != '_')]

setup(
    name='metatoolkit',
    version='0.1',
    packages=find_packages(),
    scripts=scripts,
    install_requires=required,
)
