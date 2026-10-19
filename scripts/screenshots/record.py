#!/usr/bin/env python3
"""Record a TUI's output from a real pty with a real window size.

`script -q /dev/null <cmd>` does hand the child a tty, but that pty's winsize is 0x0, so ratatui
draws nothing and the capture is silently empty. openpty + TIOCSWINSZ is the missing piece.

Writes newline-delimited JSON: {"at": ms, "data": "<base64>"} per chunk.
"""
import base64, errno, fcntl, json, os, pty, select, signal, struct, sys, termios, time

def main():
    cfg = json.loads(sys.argv[1])
    cols, rows = cfg.get("cols", 150), cfg.get("rows", 42)
    out_path = cfg["out"]
    plan = cfg["plan"]                      # [{waitMs, keys?}]
    env = dict(os.environ)
    env.update(cfg.get("env", {}))
    env.update({"COLUMNS": str(cols), "LINES": str(rows), "TERM": "xterm-256color"})

    master, slave = pty.openpty()
    fcntl.ioctl(slave, termios.TIOCSWINSZ, struct.pack("HHHH", rows, cols, 0, 0))

    pid = os.fork()
    if pid == 0:
        os.setsid()
        fcntl.ioctl(slave, termios.TIOCSCTTY, 0)
        os.dup2(slave, 0); os.dup2(slave, 1); os.dup2(slave, 2)
        os.close(master); os.close(slave)
        os.execvpe(cfg["bin"], [cfg["bin"]], env)
        os._exit(127)

    os.close(slave)
    t0 = time.time()
    chunks = []
    deadline_idx = 0
    next_action_at = t0 + plan[0]["waitMs"] / 1000 if plan else None

    def drain(timeout):
        r, _, _ = select.select([master], [], [], timeout)
        if master in r:
            try:
                data = os.read(master, 65536)
            except OSError as e:
                if e.errno in (errno.EIO, errno.EBADF):
                    return False
                raise
            if not data:
                return False
            chunks.append({"at": int((time.time() - t0) * 1000),
                           "data": base64.b64encode(data).decode("ascii")})
        return True

    alive = True
    while alive and deadline_idx < len(plan):
        now = time.time()
        if now >= next_action_at:
            step = plan[deadline_idx]
            if step.get("keys"):
                os.write(master, step["keys"].encode())
            deadline_idx += 1
            if deadline_idx < len(plan):
                next_action_at = time.time() + plan[deadline_idx]["waitMs"] / 1000
        alive = drain(0.05)

    os.write(master, b"q")
    end = time.time() + 0.6
    while time.time() < end and alive:
        alive = drain(0.05)
    try:
        os.kill(pid, signal.SIGTERM)
    except ProcessLookupError:
        pass
    os.waitpid(pid, os.WNOHANG)

    with open(out_path, "w") as fh:
        for c in chunks:
            fh.write(json.dumps(c) + "\n")
    total = sum(len(base64.b64decode(c["data"])) for c in chunks)
    print(json.dumps({"chunks": len(chunks), "bytes": total, "out": out_path}))
    # A recorder that captured nothing must say so, not leave a caller to render blank frames.
    if total == 0:
        sys.exit(3)

main()
