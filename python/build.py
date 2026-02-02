#!/usr/bin/env python3
"""
Build script for C-ML Python package.

This script:
1. Builds the C library (libcml)
2. Generates CFFI bindings
3. Builds the visualization UI
4. Copies all artifacts to the package

Usage:
    python build.py          # Build everything
    python build.py --cffi   # Only rebuild CFFI bindings
    python build.py --ui     # Only rebuild UI
    python build.py --lib    # Only rebuild C library
"""

import argparse
import shutil
import subprocess
import sys
from pathlib import Path

# Paths
ROOT_DIR = Path(__file__).parent.parent
PYTHON_DIR = Path(__file__).parent
PACKAGE_DIR = PYTHON_DIR / "cml"
VIZ_UI_DIR = ROOT_DIR / "viz-ui"
BUILD_DIR = ROOT_DIR / "build"


def run(cmd: list[str], cwd: Path = ROOT_DIR, check: bool = True):
    """Run a command and print output."""
    print(f"  > {' '.join(cmd)}")
    result = subprocess.run(cmd, cwd=cwd, capture_output=False)
    if check and result.returncode != 0:
        print(f"Command failed with code {result.returncode}")
        sys.exit(1)
    return result


def build_c_library():
    """Build the C library using CMake."""
    print("\n[1/4] Building C library...")

    BUILD_DIR.mkdir(exist_ok=True)

    # Configure
    run(
        [
            "cmake",
            "-B",
            "build",
            "-DCMAKE_BUILD_TYPE=Release",
            "-DBUILD_SHARED_LIBS=ON",
        ],
        cwd=ROOT_DIR,
    )

    # Build
    run(["cmake", "--build", "build", "--config", "Release", "-j"], cwd=ROOT_DIR)

    # Copy shared library to package
    lib_dir = PACKAGE_DIR / "_lib"
    lib_dir.mkdir(exist_ok=True)

    # Find and copy the shared library
    for pattern in ["libcml.so", "libcml.dylib", "cml.dll", "libcml_shared.so"]:
        for lib_file in BUILD_DIR.rglob(pattern):
            dest = lib_dir / lib_file.name
            print(f"  Copying {lib_file} -> {dest}")
            shutil.copy2(lib_file, dest)
            break

    print("  C library built successfully")


def build_cffi_bindings():
    """Generate CFFI bindings."""
    print("\n[2/4] Building CFFI bindings...")

    cffi_script = PACKAGE_DIR / "build_cffi.py"
    if cffi_script.exists():
        run([sys.executable, str(cffi_script)], cwd=PYTHON_DIR)
        print("  CFFI bindings generated successfully")
    else:
        print(f"  Warning: {cffi_script} not found, skipping CFFI build")


def build_ui():
    """Build the visualization UI."""
    print("\n[3/4] Building visualization UI...")

    if not VIZ_UI_DIR.exists():
        print(f"  Warning: {VIZ_UI_DIR} not found, skipping UI build")
        return

    # Check for node/npm
    if shutil.which("npm") is None:
        print("  Warning: npm not found, skipping UI build")
        return

    # Install dependencies if needed
    if not (VIZ_UI_DIR / "node_modules").exists():
        print("  Installing npm dependencies...")
        run(["npm", "ci"], cwd=VIZ_UI_DIR)

    # Build
    run(["npm", "run", "build"], cwd=VIZ_UI_DIR)

    print("  UI built successfully")


def copy_ui_to_package():
    """Copy built UI files to package static directory."""
    print("\n[4/4] Copying UI to package...")

    dist_dir = VIZ_UI_DIR / "dist"
    static_dir = PACKAGE_DIR / "viz" / "static"

    if not dist_dir.exists():
        print(f"  Warning: {dist_dir} not found, skipping copy")
        return

    # Clear existing static files
    if static_dir.exists():
        shutil.rmtree(static_dir)

    # Copy dist to static
    shutil.copytree(dist_dir, static_dir)

    # Count files
    file_count = sum(1 for _ in static_dir.rglob("*") if _.is_file())
    print(f"  Copied {file_count} files to {static_dir}")


def main():
    parser = argparse.ArgumentParser(description="Build C-ML Python package")
    parser.add_argument("--lib", action="store_true", help="Only build C library")
    parser.add_argument("--cffi", action="store_true", help="Only build CFFI bindings")
    parser.add_argument("--ui", action="store_true", help="Only build UI")
    parser.add_argument(
        "--copy-ui", action="store_true", help="Only copy UI to package"
    )
    args = parser.parse_args()

    # If no specific target, build everything
    build_all = not (args.lib or args.cffi or args.ui or args.copy_ui)

    print("=" * 50)
    print("C-ML Python Package Build")
    print("=" * 50)

    if build_all or args.lib:
        build_c_library()

    if build_all or args.cffi:
        build_cffi_bindings()

    if build_all or args.ui:
        build_ui()

    if build_all or args.copy_ui or args.ui:
        copy_ui_to_package()

    print("\n" + "=" * 50)
    print("Build complete!")
    print("=" * 50)
    print("\nTo install the package:")
    print("  pip install -e .[viz]")
    print("\nTo run the visualizer:")
    print("  python -m cml.viz")
    print("  # or")
    print("  cml-viz")


if __name__ == "__main__":
    main()
