#!/usr/bin/env python

"""mtk <script> [args...] -- run any executable file in this package by its full filename."""
import os
import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent


def main():
    args = sys.argv[1:]
    if not args or args[0] in {"-h", "--help"}:
        print("usage: mtk <script> [args...]\n")
        print("\n".join(sorted(
            p.name for p in HERE.iterdir()
            if p.is_file() and os.access(p, os.X_OK)
            and p.name != "cli.py" and not p.name.startswith("_")
        )))
        return
    path = HERE / Path(args[0]).name
    if not path.is_file():
        sys.exit(f"mtk: no script called {path.name!r} (try `mtk --help`)")
    # make `#!/usr/bin/env python3` resolve to this venv's interpreter
    os.environ["PATH"] = os.path.dirname(sys.executable) + os.pathsep + os.environ.get("PATH", "")
    try:
        os.execv(path, [str(path), *args[1:]])
    except OSError as e:
        sys.exit(f"mtk: cannot run {path.name}: {e} (missing +x or shebang?)")
