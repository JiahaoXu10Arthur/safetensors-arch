"""CLI: ``python -m safetensors_arch <path> [path ...]``

Walks directories. Prints one line per file: kind, then the reason. The reason
is printed because a classifier you cannot argue with is a classifier you
cannot trust.
"""

import sys
from pathlib import Path

from . import detect


def main(argv=None):
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in ("-h", "--help"):
        print(__doc__)
        return 0
    paths = []
    for a in argv:
        p = Path(a)
        if p.is_dir():
            paths += sorted(p.rglob("*.safetensors"))
        else:
            paths.append(p)
    if not paths:
        print("no .safetensors files found", file=sys.stderr)
        return 1
    width = max(len(p.name) for p in paths)
    for p in paths:
        kind, why = detect(p)
        print("%-*s  %-18s  %s" % (min(width, 60), p.name[:60], kind, why))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
