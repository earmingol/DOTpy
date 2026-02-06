"""Setup script for DOTpy"""

import ast
import re
from setuptools import setup, find_packages

_version_re = re.compile(r'__version__\s+=\s+(.*)')

with open('dotpy/__init__.py', 'rb') as f:
    hit = _version_re.search(f.read().decode('utf-8')).group(1)
    version = str(ast.literal_eval(hit))

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="DOTpy",
    version=version,
    author="Erick Armingol",
    author_email="erickarmingol@gmail.com",
    description="Python implementation of DOT for spatial transcriptomics deconvolution",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/earmingol/DOTpy",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "torch>=1.10.0",
        "scanpy>=1.9.0",
        "anndata>=0.8.0",
        "matplotlib>=3.5.0",
        "scikit-learn>=1.0.0",
        "scipy>=1.7.0",
    ],
    extras_require={
        "gpu": [
            "rapids-singlecell>=0.10.0",
            "cupy",
        ],
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
        ],
    },
)