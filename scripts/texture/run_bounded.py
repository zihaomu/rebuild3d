"""Monitor in a separate process: native UV libraries may hold Python's GIL."""
import argparse
import json
from pathlib import Path
import subprocess
import time
import psutil

p=argparse.ArgumentParser();p.add_argument('output',type=Path);p.add_argument('--pid',type=int)
p.add_argument('--seconds',type=float,default=1200);p.add_argument('--gib',type=float,default=11)
p.add_argument('command',nargs=argparse.REMAINDER)
a=p.parse_args();a.output.mkdir(parents=True,exist_ok=True)
cmd=a.command[1:] if a.command[:1]==['--'] else a.command
if a.pid:
    child=None;process=psutil.Process(a.pid)
else:
    if not cmd:p.error('A command or --pid is required')
    log=(a.output/'process.log').open('w')
    child=subprocess.Popen(cmd,stdout=log,stderr=subprocess.STDOUT);process=psutil.Process(child.pid)
start=time.monotonic();peak=0;reason=None
with (a.output/'external-resources.jsonl').open('w') as f:
    while True:
        if child and child.poll() is not None:break
        try:
            if not process.is_running() or process.status()==psutil.STATUS_ZOMBIE:break
            rss=process.memory_info().rss+sum(x.memory_info().rss for x in process.children(recursive=True) if x.is_running())
        except psutil.NoSuchProcess:break
        peak=max(peak,rss);elapsed=time.monotonic()-start
        f.write(json.dumps({'secondsSinceMonitorStart':elapsed,'rssBytesIncludingChildren':rss})+'\n');f.flush()
        if elapsed>a.seconds or rss>a.gib*1024**3:
            reason='resource-limit'
            descendants=process.children(recursive=True)
            for x in descendants:x.terminate()
            process.terminate()
            _,alive=psutil.wait_procs([process,*descendants],timeout=5)
            for x in alive:x.kill()
            break
        time.sleep(1)
code=child.wait() if child else None
report={'elapsedSeconds':time.monotonic()-start,'sampledPeakRSSBytesIncludingChildren':peak,
        'exitCode':code,'terminationReason':reason,'monitorAttachedAfterStart':bool(a.pid),'command':cmd}
(a.output/'process-report.json').write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report))
raise SystemExit(2 if reason else (code or 0))
