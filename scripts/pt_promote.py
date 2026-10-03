#!/usr/bin/env python3
"""Compatibility entrypoint for `pt promote`; canonical artifacts only."""

import sys

from boldt_posttrain.cli import main as _main


def main(argv=None):
    return _main(["promote"] + list(sys.argv[1:] if argv is None else argv))


if __name__ == "__main__":
    raise SystemExit(main())
