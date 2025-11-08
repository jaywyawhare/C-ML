#!/usr/bin/env python3
import argparse, os, subprocess, sys, time, webbrowser
import socket
import threading

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.abspath(os.path.join(HERE, ".."))
UI_DIR = os.path.join(ROOT, "viz-ui")
VIZ_SERVER_SCRIPT = os.path.join(HERE, "fastapi_server.py")


def ensure_ui_setup():
    node_modules = os.path.join(UI_DIR, "node_modules")
    if not os.path.isdir(node_modules):
        print("Installing npm dependencies...", file=sys.stderr)
        subprocess.check_call(["npm", "install"], cwd=UI_DIR)


def is_port_open(port):
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1)
        result = sock.connect_ex(("localhost", port))
        sock.close()
        return result == 0
    except:
        return False


def start_viz_server():
    """Start the C-ML viz HTTP server."""
    print("Starting C-ML viz HTTP server...", file=sys.stderr)
    env = os.environ.copy()
    # Force server to bind to 8001 so readiness check matches
    env["PORT"] = "8001"
    p = subprocess.Popen(
        [sys.executable, VIZ_SERVER_SCRIPT],
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
    )

    # Wait for server to be ready (max 10 seconds)
    url = "http://localhost:8001"
    for i in range(20):
        time.sleep(0.5)
        if is_port_open(8001):
            print(f"✓ Viz server ready at {url}", file=sys.stderr)
            return p, url
        if p.poll() is not None:
            print("✗ Viz server process exited", file=sys.stderr)
            return None, None

    print("✗ Viz server failed to start (timeout)", file=sys.stderr)
    try:
        p.terminate()
        p.wait(timeout=2)
    except:
        pass
    return None, None


def start_ui():
    print("Starting Vite dev server...", file=sys.stderr)
    env = os.environ.copy()
    # Use unbuffered output to see errors immediately
    p = subprocess.Popen(
        ["npm", "run", "dev"],
        cwd=UI_DIR,
        env=env,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=0,
    )

    # Read output in background
    import threading
    import queue

    output_queue = queue.Queue()

    def read_output():
        try:
            for line in iter(p.stdout.readline, ""):
                if line:
                    output_queue.put(line)
                    if "Local:" in line or "ready" in line.lower() or "VITE" in line:
                        break
        except:
            pass

    thread = threading.Thread(target=read_output, daemon=True)
    thread.start()

    # Wait for server to be ready (max 15 seconds)
    url = "http://localhost:5173"
    output_lines = []
    for i in range(30):
        # Collect output
        try:
            while True:
                output_lines.append(output_queue.get_nowait())
        except queue.Empty:
            pass

        time.sleep(0.5)
        if is_port_open(5173):
            print(f"✓ Server ready at {url}", file=sys.stderr)
            if output_lines:
                print("Server output:", file=sys.stderr)
                for line in output_lines[-3:]:
                    print(f"  {line}", file=sys.stderr, end="")
            return p, url
        if p.poll() is not None:
            # Process died, show output
            print("✗ Server process exited. Output:", file=sys.stderr)
            for line in output_lines[-10:]:
                print(f"  {line}", file=sys.stderr, end="")
            return None, None

    print("✗ Server failed to start (timeout)", file=sys.stderr)
    print("Last output:", file=sys.stderr)
    for line in output_lines[-5:]:
        print(f"  {line}", file=sys.stderr, end="")
    try:
        p.terminate()
        p.wait(timeout=2)
    except:
        pass
    return None, None


def open_browser(url):
    try:
        webbrowser.open(url)
        print(f"Opened browser at {url}", file=sys.stderr)
    except Exception as e:
        print(f"Could not open browser: {e}", file=sys.stderr)
        print(f"Please open manually: {url}", file=sys.stderr)


def run_target(exe_and_args):
    env = os.environ.copy()
    env["CML_VIZ"] = "1"
    public_dir = os.path.join(UI_DIR, "public")
    os.makedirs(public_dir, exist_ok=True)
    return subprocess.call(exe_and_args, env=env)


def main():
    ap = argparse.ArgumentParser(description="C-ML viz launcher")
    ap.add_argument(
        "target", nargs=argparse.REMAINDER, help="Executable (and args) to run"
    )
    args = ap.parse_args()
    if not args.target:
        print("Usage: viz <executable> [args...]", file=sys.stderr)
        return 2

    print("C-ML Viz Launcher", file=sys.stderr)
    print("=" * 50, file=sys.stderr)

    ensure_ui_setup()

    # Start viz HTTP server first
    viz_server_proc, viz_url = start_viz_server()
    if not viz_server_proc:
        print("Failed to start viz HTTP server", file=sys.stderr)
        return 1

    # Start Vite dev server
    ui_proc, ui_url = start_ui()
    if not ui_proc:
        print("Failed to start UI server", file=sys.stderr)
        try:
            viz_server_proc.terminate()
            viz_server_proc.wait(timeout=2)
        except:
            pass
        return 1

    try:
        if ui_url:
            print(f"Opening browser at {ui_url}...", file=sys.stderr)
            open_browser(ui_url)
            print(f"\nUI URL: {ui_url}", file=sys.stderr)
            print(f"Viz Server URL: {viz_url}\n", file=sys.stderr)

        exe_path = args.target[0]
        print(f'Running: {" ".join(args.target)}', file=sys.stderr)
        print("-" * 50, file=sys.stderr)
        rc = run_target(args.target)
        print("-" * 50, file=sys.stderr)

        if rc == 0:
            print("\n✓ Model ran successfully", file=sys.stderr)
            graph_path = os.path.join(UI_DIR, "public", "graph.json")
            if os.path.exists(graph_path):
                print(f"✓ Graph exported to {graph_path}", file=sys.stderr)
            else:
                print(f"⚠ Graph file not found at {graph_path}", file=sys.stderr)
            print("\nServers are running. Press Ctrl-C to stop.", file=sys.stderr)
            try:
                while True:
                    time.sleep(1)
            except KeyboardInterrupt:
                print("\nShutting down...", file=sys.stderr)
        else:
            print(f"\n✗ Target exited with code {rc}", file=sys.stderr)
        return rc
    finally:
        try:
            print("Stopping servers...", file=sys.stderr)
            ui_proc.terminate()
            ui_proc.wait(timeout=2)
            viz_server_proc.terminate()
            viz_server_proc.wait(timeout=2)
        except:
            pass


if __name__ == "__main__":
    sys.exit(main())
