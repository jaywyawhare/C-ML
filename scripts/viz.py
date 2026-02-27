#!/usr/bin/env python3
"""
C-ML Visualization Launcher

A CLI tool that launches the visualization server and runs a C-ML executable
with visualization enabled.

Usage:
    python viz.py <executable> [args...]
    ./viz.py ./build/bin/cml
"""

import argparse
import os
import signal
import socket
import subprocess
import sys
import time
import webbrowser
from pathlib import Path

# Configuration
HERE = Path(__file__).resolve().parent
VIZ_SERVER_SCRIPT = HERE / "fastapi_server.py"
DEFAULT_PORT = 8001
SERVER_TIMEOUT = 10  # seconds


def is_port_open(port: int, host: str = "localhost") -> bool:
    """Check if a port is open and accepting connections."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
            sock.settimeout(1)
            return sock.connect_ex((host, port)) == 0
    except (socket.error, OSError):
        return False


def wait_for_server(port: int, timeout: float = SERVER_TIMEOUT) -> bool:
    """Wait for the server to become available."""
    start = time.time()
    while time.time() - start < timeout:
        if is_port_open(port):
            return True
        time.sleep(0.25)
    return False


def start_viz_server(port: int = DEFAULT_PORT) -> tuple:
    """
    Start the visualization HTTP server.

    Returns:
        tuple: (process, url) or (None, None) on failure
    """
    print(f"Starting C-ML visualization server on port {port}...", file=sys.stderr)

    env = os.environ.copy()
    env["PORT"] = str(port)

    try:
        process = subprocess.Popen(
            [sys.executable, str(VIZ_SERVER_SCRIPT)],
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )
    except Exception as e:
        print(f"Failed to start server: {e}", file=sys.stderr)
        return None, None

    # Wait for server to be ready
    if wait_for_server(port):
        url = f"http://localhost:{port}"
        print(f"Server ready at {url}", file=sys.stderr)
        return process, url

    # Server failed to start
    print("Server failed to start (timeout)", file=sys.stderr)

    # Print any output from the server
    if process.poll() is not None:
        output = process.stdout.read()
        if output:
            print("Server output:", file=sys.stderr)
            for line in output.splitlines():
                print(f"  {line}", file=sys.stderr)

    try:
        process.terminate()
        process.wait(timeout=2)
    except Exception:
        process.kill()

    return None, None


def open_browser(url: str) -> bool:
    """Open URL in default browser."""
    try:
        webbrowser.open(url)
        print(f"Opened browser at {url}", file=sys.stderr)
        return True
    except Exception as e:
        print(f"Could not open browser: {e}", file=sys.stderr)
        print(f"Please open manually: {url}", file=sys.stderr)
        return False


def run_target(args: list, cwd: str = None) -> int:
    """
    Run the target executable with visualization environment variables.

    Returns:
        int: Exit code from the executable
    """
    env = os.environ.copy()
    env["CML_VIZ"] = "1"

    try:
        return subprocess.call(args, env=env, cwd=cwd)
    except FileNotFoundError:
        print(f"Error: Executable not found: {args[0]}", file=sys.stderr)
        return 127
    except PermissionError:
        print(f"Error: Permission denied: {args[0]}", file=sys.stderr)
        return 126


def cleanup_server(process: subprocess.Popen) -> None:
    """Gracefully stop the server process."""
    if process is None:
        return

    print("Stopping server...", file=sys.stderr)
    try:
        process.terminate()
        process.wait(timeout=3)
    except subprocess.TimeoutExpired:
        print("Force killing server...", file=sys.stderr)
        process.kill()
        process.wait()
    except Exception:
        pass


def main() -> int:
    parser = argparse.ArgumentParser(
        description="C-ML Visualization Launcher",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    %(prog)s ./build/bin/cml
    %(prog)s ./autograd_example --epochs 100
    %(prog)s python train.py
        """,
    )
    parser.add_argument(
        "--port",
        type=int,
        default=DEFAULT_PORT,
        help=f"Port for visualization server (default: {DEFAULT_PORT})",
    )
    parser.add_argument(
        "--no-browser",
        action="store_true",
        help="Don't open browser automatically",
    )
    parser.add_argument(
        "target",
        nargs=argparse.REMAINDER,
        help="Executable (and args) to run",
    )

    args = parser.parse_args()

    if not args.target:
        parser.print_help()
        print("\nError: No executable specified", file=sys.stderr)
        return 2

    print("C-ML Visualization Launcher", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    # Start visualization server
    server_process, viz_url = start_viz_server(args.port)
    if not server_process:
        return 1

    exit_code = 0
    try:
        # Open browser
        if not args.no_browser:
            open_browser(viz_url)

        # Run the target executable
        print(f"Running: {' '.join(args.target)}", file=sys.stderr)
        print("-" * 50, file=sys.stderr)

        exit_code = run_target(args.target)

        print("-" * 50, file=sys.stderr)

        if exit_code == 0:
            print("\nExecutable completed successfully", file=sys.stderr)
            print("Server is running. Press Ctrl+C to stop.", file=sys.stderr)

            # Keep server running until interrupted
            try:
                while True:
                    time.sleep(1)
                    # Check if server is still running
                    if server_process.poll() is not None:
                        print("Server stopped unexpectedly", file=sys.stderr)
                        break
            except KeyboardInterrupt:
                print("\nShutting down...", file=sys.stderr)
        else:
            print(f"\nExecutable exited with code {exit_code}", file=sys.stderr)

    finally:
        cleanup_server(server_process)

    return exit_code


if __name__ == "__main__":
    sys.exit(main())
