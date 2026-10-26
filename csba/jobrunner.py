import subprocess, sys, time, os
jobs = [l.strip() for l in open(sys.argv[1]) if l.strip() and not l.startswith('#')]
P = int(sys.argv[2]) if len(sys.argv) > 2 else 5
log = open(sys.argv[1] + '.status', 'a')
while subprocess.run("pgrep -f '[c]03_train.py --variant main --fold'", shell=True, capture_output=True).stdout.strip():
    time.sleep(30)
running, k = [], 0
while jobs or running:
    running = [(p, c) for p, c in running if p.poll() is None or (log.write(f'{time.strftime("%H:%M")} exit {p.returncode} {c}\n') and log.flush() and False)]
    while jobs and len(running) < P:
        c = jobs.pop(0).replace('{gpu}', str(k % 2)); k += 1
        running.append((subprocess.Popen(c, shell=True, cwd=os.environ.get('CSBA_ROOT', 'work')), c))
        log.write(f'{time.strftime("%H:%M")} start {c}\n'); log.flush()
    time.sleep(20)
log.write('ALL DONE\n'); log.flush()
