#!/usr/bin/env python
from setuptools import setup, find_packages
import pathlib

# Read requirements
with open('requirements.txt') as f:
    install_requires = f.read().splitlines()

# Read long description from README if available
here = pathlib.Path(__file__).parent
long_description = (here / "README.md").read_text(encoding="utf-8") if (here / "README.md").exists() else ""

setup(
    name='metatoolkit',
    version='0.3.0',
    author='Theo Portlock',
    author_email='theo@portlocklab.com',
    description='A set of scripts for biological data analysis',
    long_description=long_description,
    long_description_content_type='text/markdown',
    url='https://github.com/theoportlock/metatoolkit',
    packages=find_packages(),
    python_requires='>=3.7',
    install_requires=install_requires,
    include_package_data=True,
    classifiers=[
        'Development Status :: 3 - Alpha',
        'Intended Audience :: Science/Research',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'License :: OSI Approved :: MIT License',
        'Operating System :: OS Independent',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Programming Language :: Python :: 3.11',
        'Programming Language :: Python :: 3.12',
    ],
    project_urls={
        'Source': 'https://github.com/theoportlock/metatoolkit',
        'Bug Tracker': 'https://github.com/theoportlock/metatoolkit/issues',
    },
)
