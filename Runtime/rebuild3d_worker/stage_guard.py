"""Keep a heavy stage tied to its supervisor even if that supervisor is killed."""
import os
import signal
import subprocess
import sys
import time


def main():
    owner = int(sys.argv[1])
    if os.getppid() != owner:
        return 130
    # Supervisor starts us as a new session/process-group leader. The algorithm
    # inherits that group so both ordinary cancellation and owner death stop it.
    if os.getpgrp() != os.getpid():
        raise RuntimeError("Stage guard must own its process group")
    stopping = False

    def stop(*_):
        nonlocal stopping
        stopping = True

    signal.signal(signal.SIGTERM, stop)
    signal.signal(signal.SIGINT, stop)
    child = subprocess.Popen(sys.argv[2:])
    while child.poll() is None:
        if stopping or os.getppid() != owner:
            os.killpg(os.getpgrp(), signal.SIGTERM)
            try:
                child.wait(timeout=8)
            except subprocess.TimeoutExpired:
                # Also kills this guard; no detached stage may outlive its owner.
                os.killpg(os.getpgrp(), signal.SIGKILL)
            return 130
        time.sleep(0.25)
    return child.returncode if child.returncode >= 0 else 128 - child.returncode


if __name__ == "__main__":
    sys.exit(main())
