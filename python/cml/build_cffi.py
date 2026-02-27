#!/usr/bin/env python3
"""
Build script for CML CFFI bindings.

This script builds the CFFI bindings for the C-ML library.
Run this after building the C-ML library to generate the Python bindings.

Usage:
    python3 build_cffi.py

Requirements:
    - cffi package: pip install cffi
    - C-ML library must be built first (make or cmake)
"""

import os
import sys
import subprocess
from pathlib import Path


def find_cml_lib():
    """Find the compiled CML library."""
    cml_root = Path(__file__).parent.parent.parent
    lib_paths = [
        cml_root / "lib",
        cml_root / "lib" / "libcml.a",
        cml_root / "lib" / "libcml.so",
        cml_root / "build" / "lib",
        cml_root / "build" / "lib" / "libcml.a",
        cml_root / "build" / "lib" / "libcml.so",
    ]

    for path in lib_paths:
        if path.exists():
            # Return the parent directory if it's a file
            if path.is_file():
                return path.parent
            return path

    raise RuntimeError(
        "CML library not found. Please build CML first:\n"
        "  cd /path/to/C-ML\n"
        "  make\n"
        "or\n"
        "  mkdir build && cd build\n"
        "  cmake -DBUILD_SHARED_LIBS=ON ..\n"
        "  make"
    )


def check_dependencies():
    """Check if required dependencies are installed."""
    try:
        import cffi

        print(f"cffi {cffi.__version__} found")
    except ImportError:
        print("cffi not found. Install it with: pip install cffi")
        sys.exit(1)


def build_bindings():
    """Build the CFFI bindings."""
    print("Building CML CFFI bindings...")

    try:
        # Import CFFI builder
        from _cml_cffi import ffi
    except ImportError:
        print("Error: Could not import _cml_cffi")
        sys.exit(1)

    try:
        # Compile the bindings
        print("Compiling CFFI module...")
        ffi.compile(verbose=True)
        print("CFFI bindings compiled successfully")

    except Exception as e:
        print(f"Error compiling CFFI bindings: {e}")
        sys.exit(1)


def verify_installation():
    """Verify the bindings work."""
    print("\nVerifying installation...")

    try:
        from cml import init, cleanup, seed

        # Test basic initialization
        init()
        seed(42)
        cleanup()

        print("Bindings verified successfully")
        return True

    except Exception as e:
        print(f"Verification failed: {e}")
        import traceback

        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("CML Python Bindings Builder")
    print("=" * 40)

    # Check dependencies
    print("\nChecking dependencies...")
    check_dependencies()

    # Find CML library
    print("\nLocating CML library...")
    try:
        lib_path = find_cml_lib()
        print(f"CML library found at: {lib_path}")
    except RuntimeError as e:
        print(f"{e}")
        sys.exit(1)

    # Build bindings
    print()
    try:
        build_bindings()
    except Exception as e:
        print(f"Build failed: {e}")
        import traceback

        traceback.print_exc()
        sys.exit(1)

    # Verify
    if verify_installation():
        print("\n" + "=" * 40)
        print("Build successful! You can now use CML from Python:")
        print("\n  import cml")
        print("  cml.init()")
        print("  # ... use CML ...")
        print("  cml.cleanup()")
    else:
        sys.exit(1)
