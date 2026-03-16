#!/usr/bin/env bash
# C-ML Build Script for Linux and macOS
# Usage: ./build.sh [command]
# Commands: all, lib, frontend, install, clean, help

set -e

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Detect OS
OS="unknown"
if [[ "$OSTYPE" == "linux-gnu"* ]]; then
    OS="linux"
elif [[ "$OSTYPE" == "darwin"* ]]; then
    OS="macos"
fi

echo -e "${BLUE}C-ML Build Script${NC}"
echo -e "${BLUE}==================${NC}"
echo -e "Platform: ${GREEN}$OS${NC}\n"

# Helper functions
print_success() {
    echo -e "${GREEN}✓${NC} $1"
}

print_error() {
    echo -e "${RED}✗${NC} $1"
}

print_info() {
    echo -e "${BLUE}ℹ${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}⚠${NC} $1"
}

# Check dependencies
check_dependencies() {
    print_info "Checking dependencies..."

    local missing_deps=()

    # Check for C compiler
    if ! command -v gcc &> /dev/null && ! command -v clang &> /dev/null; then
        missing_deps+=("gcc or clang")
    fi

    # Check for make
    if ! command -v make &> /dev/null; then
        missing_deps+=("make")
    fi

    # Check for Python (for viz server)
    if ! command -v python3 &> /dev/null; then
        print_warning "python3 not found - visualization server will not work"
    fi

    # Check for Node.js (for frontend)
    if ! command -v npm &> /dev/null; then
        print_warning "npm not found - frontend build will be skipped"
    fi

    if [ ${#missing_deps[@]} -ne 0 ]; then
        print_error "Missing required dependencies: ${missing_deps[*]}"
        echo ""
        echo "Install instructions:"
        if [ "$OS" = "linux" ]; then
            echo "  Ubuntu/Debian: sudo apt-get install build-essential"
            echo "  Fedora/RHEL:   sudo dnf install gcc make"
            echo "  Arch:          sudo pacman -S base-devel"
        elif [ "$OS" = "macos" ]; then
            echo "  Install Xcode Command Line Tools:"
            echo "  xcode-select --install"
        fi
        exit 1
    fi

    print_success "All required dependencies found"
}

# Build the C library
build_lib() {
    print_info "Building C library..."
    mkdir -p build
    cd build
    cmake .. -DCMAKE_BUILD_TYPE=Release
    make -j$(nproc 2>/dev/null || sysctl -n hw.ncpu 2>/dev/null || echo 4)
    cd ..
    print_success "Library built successfully"
}

# Build the frontend
build_frontend() {
    if ! command -v npm &> /dev/null; then
        print_warning "npm not found - skipping frontend build"
        return
    fi

    print_info "Building frontend..."
    cd viz-ui

    if [ ! -d "node_modules" ]; then
        print_info "Installing npm dependencies..."
        npm install
    fi

    npm run build
    cd ..
    print_success "Frontend built successfully"
}

# Install the library
install_lib() {
    print_info "Installing C-ML library..."

    if [ ! -d "build" ]; then
        print_info "Build directory not found, building first..."
        build_lib
    fi

    cd build
    if [ "$EUID" -ne 0 ]; then
        print_warning "Installation requires root privileges"
        sudo cmake --install .
    else
        cmake --install .
    fi
    cd ..

    print_success "Library installed successfully"

    # Update library cache on Linux
    if [ "$OS" = "linux" ]; then
        print_info "Updating library cache..."
        sudo ldconfig 2>/dev/null || true
    fi

    echo ""
    echo -e "${GREEN}Installation complete!${NC}"
    echo ""
    echo "To use the library:"
    echo "  #include <cml/cml.h>"
    echo "  gcc your_code.c -lcml -lm -o your_program"
    echo ""
    echo "To enable visualization:"
    echo "  VIZ=1 ./your_program"
}

# Clean build artifacts
clean_build() {
    print_info "Cleaning build artifacts..."

    if [ -d "build" ]; then
        rm -rf build
    fi

    if [ -d "viz-ui/dist" ]; then
        rm -rf viz-ui/dist
    fi

    if [ -d "viz-ui/node_modules" ]; then
        print_info "Removing node_modules..."
        rm -rf viz-ui/node_modules
    fi

    print_success "Clean complete"
}

# Run tests
run_tests() {
    print_info "Running tests..."
    if [ ! -d "build" ]; then
        print_info "Build directory not found, building first..."
        build_lib
    fi
    cd build
    ctest --output-on-failure
    cd ..
    print_success "All tests passed"
}

# Run example with visualization
run_example() {
    local example="${1:-training_loop_example}"

    if [ ! -d "build" ]; then
        print_info "Build directory not found, building first..."
        build_lib
    fi

    local binary="build/bin/${example}"
    if [ ! -f "$binary" ]; then
        # Try without bin subdirectory
        binary="build/${example}"
    fi

    if [ ! -f "$binary" ]; then
        print_error "Example '${example}' not found"
        echo "Available examples:"
        ls build/bin/ 2>/dev/null || ls build/*.out 2>/dev/null || echo "  No examples found in build/"
        exit 1
    fi

    print_info "Running ${example} with visualization enabled..."
    echo ""
    VIZ=1 "$binary"

    echo ""
    print_success "Example completed!"

    # Check what files were generated
    if [ -f "training.json" ] || [ -f "graph.json" ] || [ -f "model_architecture.json" ]; then
        print_info "Generated visualization data:"
        [ -f "training.json" ] && echo "  - training.json (training metrics)"
        [ -f "graph.json" ] && echo "  - graph.json (computational graph)"
        [ -f "model_architecture.json" ] && echo "  - model_architecture.json (model structure)"
        [ -f "kernels.json" ] && echo "  - kernels.json (kernel analysis)"
        echo ""
        echo "To view the visualization, run:"
        echo "  ./build.sh viz"
    fi
}

# Start visualization server
start_viz() {
    if ! command -v python3 &> /dev/null; then
        print_error "python3 not found - cannot start visualization server"
        exit 1
    fi

    print_info "Starting visualization server..."
    echo ""
    echo "Dashboard: http://localhost:8001"
    echo "Press Ctrl+C to stop"
    echo ""

    python3 viz/serve.py
}

# Show help
show_help() {
    cat << EOF
C-ML Build Script

Usage: ./build.sh [command] [args]

Commands:
  all         Build everything (library + frontend)
  lib         Build only the C library
  frontend    Build only the frontend
  install     Install the library system-wide (requires sudo)
  test        Run tests
  run [name]  Run an example with visualization (default: training_loop_example)
  viz         Start the visualization server (http://localhost:8001)
  clean       Clean build artifacts
  help        Show this help message

Examples:
  ./build.sh all                          # Build library and frontend
  ./build.sh run                          # Run training example with viz
  ./build.sh run simple_xor               # Run specific example
  ./build.sh viz                          # Start visualization dashboard
  ./build.sh install                      # Install to /usr/local
  ./build.sh clean                        # Clean all build files

Visualization Workflow:
  1. ./build.sh all                       # Build everything
  2. ./build.sh run training_loop_example # Run training with data export
  3. ./build.sh viz                        # Open dashboard at localhost:8001

Environment Variables:
  PREFIX      Installation prefix (default: /usr/local)
  CC          C compiler to use (default: gcc)
  VIZ=1       Enable visualization data export (auto-set by 'run' command)

EOF
}

# Main command handler
main() {
    local cmd="${1:-all}"

    case "$cmd" in
        all)
            check_dependencies
            build_lib
            build_frontend
            print_success "Build complete!"
            echo ""
            echo "Next steps:"
            echo "  1. Run 'sudo ./build.sh install' to install system-wide"
            echo "  2. Or link against build/lib/libcml.a directly"
            echo ""
            echo "Run examples with visualization:"
            echo "  ./build.sh run                    # Run training example"
            echo "  ./build.sh viz                    # Start dashboard"
            ;;
        lib)
            check_dependencies
            build_lib
            ;;
        frontend)
            build_frontend
            ;;
        install)
            install_lib
            ;;
        test)
            run_tests
            ;;
        run)
            run_example "${2:-training_loop_example}"
            ;;
        viz|server)
            start_viz
            ;;
        clean)
            clean_build
            ;;
        help|--help|-h)
            show_help
            ;;
        *)
            print_error "Unknown command: $cmd"
            echo ""
            show_help
            exit 1
            ;;
    esac
}

main "$@"
