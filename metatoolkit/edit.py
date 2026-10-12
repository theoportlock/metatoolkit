#!/usr/bin/env python3
"""Open a file in vim in a new Windows Terminal window (from WSL).

Assumes the calling shell already has the project environment loaded
(e.g. the workforce wrapper: `cd ~/nipper && . env.sh && ...`). That
environment and working directory are snapshotted and replayed in the new
window, so this script never needs to know where env.sh lives.

A small temp script is used instead of `bash -c "<string>"` because wt.exe
treats ';' as a subcommand separator and mangles inline command strings.
"""
import os
import re
import shlex
import subprocess
import sys
import tempfile

VALID_NAME = re.compile(r"^[A-Za-z_][A-Za-z0-9_]*$")

# Per-session / per-process variables that shouldn't be copied into the new shell.
# TERM & co. matter: the wrapper may run without a tty, and a stale TERM breaks vim.
SKIP_VARS = {
    "_", "PWD", "OLDPWD", "SHLVL", "PPID", "UID", "EUID",
    "BASHOPTS", "SHELLOPTS", "TERM", "COLORTERM", "LINES", "COLUMNS",
}
SKIP_PREFIXES = ("WSL",)  # WSL_INTEROP etc. are set fresh by each WSL session


def build_script(abs_path, cwd, environ=None):
    """Return bash script text that recreates the environment, then runs vim."""
    environ = os.environ if environ is None else environ
    lines = [
        "#!/usr/bin/env bash",
        'rm -f -- "$0"  # self-delete: this file holds a copy of the environment',
    ]
    for name, value in environ.items():
        if VALID_NAME.match(name) and name not in SKIP_VARS and not name.startswith(SKIP_PREFIXES):
            lines.append(f"export {name}={shlex.quote(value)}")
    lines += [
        # env.sh puts *relative* entries on PATH ("code/", ...), so the cwd must match
        f"cd {shlex.quote(cwd)}",
        f"vim {shlex.quote(abs_path)}",
        "exec bash  # keep the window open after vim exits",
    ]
    return "\n".join(lines) + "\n"


def open_vim_in_new_window(target_file):
    abs_path = os.path.abspath(target_file)

    fd, script_path = tempfile.mkstemp(prefix="edit-", suffix=".sh")  # created 0600
    with os.fdopen(fd, "w") as f:
        f.write(build_script(abs_path, os.getcwd()))

    wsl_cmd = ["wsl.exe"]
    distro = os.environ.get("WSL_DISTRO_NAME")
    if distro:
        wsl_cmd += ["-d", distro]
    wsl_cmd += ["bash", script_path]

    # Windows Terminal first; fall back to a plain cmd window
    for launcher in (["wt.exe"], ["cmd.exe", "/c", "start"]):
        try:
            subprocess.Popen(
                launcher + wsl_cmd,
                stdin=subprocess.DEVNULL,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                start_new_session=True,
            )
        except FileNotFoundError:
            continue
        print(f"✔ Sent {os.path.basename(target_file)} to a new window with environment loaded.")
        return 0

    os.remove(script_path)
    print("Error: neither wt.exe nor cmd.exe could be found on PATH", file=sys.stderr)
    return 1


def main():
    if len(sys.argv) < 2:
        print("Usage: edit.py <filename>", file=sys.stderr)
        return 2
    return open_vim_in_new_window(sys.argv[1])


if __name__ == "__main__":
    sys.exit(main())
