#!/usr/bin/env python3
"""Probe relocated dependencies with network and development paths denied by macOS sandbox."""
import argparse
import json
from pathlib import Path
import subprocess
import tempfile

parser = argparse.ArgumentParser()
parser.add_argument("runtime", type=Path)
parser.add_argument("report", type=Path)
parser.add_argument("--deny-root", type=Path, required=True)
args = parser.parse_args()
runtime = args.runtime.resolve()
denied = [args.deny_root.resolve(), Path("/opt/homebrew"), Path("/Users/zmu/.local/share/uv"), Path("/Library/Developer")]
profile = "(version 1) (allow default) (deny network*) (deny file-read* " + " ".join(
    "(subpath " + json.dumps(str(path)) + ")" for path in denied) + ")"
code = """import sys,json,torch,numpy,cv2,xatlas,trimesh,scipy,psutil,safetensors
from pxr import Usd
print(json.dumps({'prefix':sys.prefix,'executable':sys.executable,'torch':torch.__version__,
'mps':torch.backends.mps.is_available(),'opencv':cv2.__version__,'usd':Usd.GetVersion(),
'modules':{m.__name__:m.__file__ for m in [torch,numpy,cv2,xatlas,trimesh,scipy,psutil,safetensors]}},indent=2))
"""
result = subprocess.run(["/usr/bin/sandbox-exec", "-p", profile, str(runtime / "python/bin/python3.12"), "-I", "-B", "-c", code],
    cwd=runtime, env={"PATH": "/usr/bin:/bin", "TMPDIR": tempfile.gettempdir()}, text=True, capture_output=True)
record = {"relocatedRuntime": str(runtime), "sandboxProfile": profile, "exitCode": result.returncode,
          "stdout": result.stdout, "stderr": result.stderr,
          "scope": "Dependency import and Metal support only; not full reconstruction or independent-user UI acceptance"}
args.report.write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
raise SystemExit(result.returncode)
