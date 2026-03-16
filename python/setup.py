#!/usr/bin/env python3
"""
Setup script for CML Python bindings.

Installation:
    python setup.py install

Development mode:
    pip install -e .

Requirements:
    - cffi
    - C-ML library (must be built first)
"""

from setuptools import setup, find_packages
from setuptools.command.build_py import build_py
import subprocess
import sys
import os

# Get version from __init__.py
version = "0.0.3"

# Read long description
with open("README.md", "r", encoding="utf-8") as f:
    long_description = f.read()


class BuildCFFI(build_py):
    """Custom build command that builds CFFI bindings."""

    def run(self):
        # Build CFFI bindings
        print("Building CFFI bindings...")
        try:
            subprocess.check_call([sys.executable, "-m", "cml.build_cffi"])
        except subprocess.CalledProcessError as e:
            print(f"Error building CFFI bindings: {e}")
            print("\nMake sure CML library is built first:\n" "  cd ..\n" "  make\n")
            raise

        # Continue with normal build
        super().run()


setup(
    name="cml",
    version=version,
    author="C-ML Contributors",
    author_email="",
    description="Python bindings for C-ML (C Machine Learning Library)",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/jaywyawhare/C-ML",
    packages=find_packages(),
    classifiers=[
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "License :: OSI Approved :: MIT License",
        "Operating System :: OS Independent",
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Developers",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Artificial Intelligence",
        "Topic :: Scientific/Engineering :: Information Analysis",
    ],
    python_requires=">=3.8",
    install_requires=[
        "cffi>=1.14.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0",
            "pytest-cov",
            "black",
            "flake8",
            "mypy",
        ],
    },
    cmdclass={
        "build_py": BuildCFFI,
    },
    include_package_data=True,
    package_data={
        "cml": ["*.py"],
    },
)
