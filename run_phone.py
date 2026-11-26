"""Find an IP Webcam server on the phone hotspot and launch Smart surveillance system for Real-Time Multi-Threat Detection (Fall, Fight, and Fire)."""
import argparse
import re
import socket
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

PORT = 8080
COMMON = ["192.168.43.1", "192.168.137.1", "172.20.10.1", "192.168.49.1", "192.168.42.129", "192.168.44.1"]


def my_ip():
    try:
        s = socket.socket(socket.AF_INET, socket.SOCK_DGRAM)
        s.connect(("8.8.8.8", 80))
        ip = s.getsockname()[0]
        s.close()
        return ip
    except Exception:
        return None


def gateways_from_route(text):
    """Read the default gateway from Windows route output."""
    return re.findall(r"^\s*0\.0\.0\.0\s+0\.0\.0\.0\s+(\d+\.\d+\.\d+\.\d+)", text, re.M)


def gateways():
    try:
        out = subprocess.run(["route", "print", "-4"], capture_output=True, text=True, timeout=10).stdout
        return gateways_from_route(out)
    except Exception:
        return []


def reachable(ip):
    try:
        socket.create_connection((ip, PORT), timeout=1.5).close()
        return ip
    except Exception:
        return None


def is_ip_webcam(ip, port=PORT, timeout=2.0):
    """Confirm the host serves IP Webcam's JPEG snapshot, not just an open port."""
    url = f"http://{ip}:{port}/shot.jpg"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            if response.status != 200:
                return False
            content_type = response.headers.get_content_type()
            return content_type == "image/jpeg" or response.read(3).startswith(b"\xff\xd8\xff")
    except Exception:
        return False


def candidates():
    c = list(gateways())
    me = my_ip()
    if me:
        c.append(".".join(me.split(".")[:3]) + ".1")
    c += COMMON
    seen, out = set(), []
    for x in c:
        if x not in seen:
            seen.add(x)
            out.append(x)
    return me, out


def discover_phone(port=PORT, retry_s=3.0):
    """Keep looking until the phone's IP Webcam server becomes available."""
    while True:
        me, cands = candidates()
        print(f"Laptop IP address: {me}")
        print("Searching these hotspot addresses for IP Webcam:", ", ".join(cands))
        with ThreadPoolExecutor(max_workers=max(1, min(16, len(cands)))) as ex:
            reachable_hosts = [ip for ip in ex.map(reachable, cands) if ip]
        with ThreadPoolExecutor(max_workers=max(1, min(16, len(reachable_hosts)))) as ex:
            found = [ip for ip in ex.map(lambda host: host if is_ip_webcam(host, port) else None,
                                         reachable_hosts) if ip]
        if found:
            return found[0]
        print(f"Phone not found. Start IP Webcam and keep the laptop connected to its hotspot; retrying in {retry_s:g}s.")
        time.sleep(retry_s)


def launch_app(url, app_args):
    app = Path(__file__).resolve().with_name("app.py")
    if not any(arg == "--rotate" or arg.startswith("--rotate=") for arg in app_args):
        app_args = ["--rotate", "90", *app_args]
    if "--flip-horizontal" not in app_args:
        app_args = ["--flip-horizontal", *app_args]
    print(f"\nPhone source: {url}\nPress q in the video window to quit.\n")
    return subprocess.call([sys.executable, str(app), "--source", url, *app_args], cwd=app.parent)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--ip", help="phone IP address; skips auto-discovery")
    parser.add_argument("--url", help="full IP Webcam video URL; skips auto-discovery and probing")
    parser.add_argument("--port", type=int, default=PORT)
    args, app_args = parser.parse_known_args(argv)

    if args.url:
        return launch_app(args.url, app_args)
    if args.ip:
        while not is_ip_webcam(args.ip, args.port):
            print(f"IP Webcam is not available at {args.ip}:{args.port}; retrying in 3s.")
            time.sleep(3.0)
        return launch_app(f"http://{args.ip}:{args.port}/video", app_args)

    while True:
        ip = discover_phone(args.port)
        result = launch_app(f"http://{ip}:{args.port}/video", app_args)
        if result == 0:
            return 0
        print("Smart surveillance system for Real-Time Multi-Threat Detection (Fall, Fight, and Fire) could not start the phone stream. Searching for IP Webcam again in 3s.")
        time.sleep(3.0)


if __name__ == "__main__":
    raise SystemExit(main())
