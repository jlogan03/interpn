#!/usr/bin/env python3
"""Run the project's pytest suite as a PGO workload."""

from __future__ import annotations

import sys


def main() -> None:
    import pytest  # Local import to avoid mandatory test dependency for runtime users

    argv = ["-q"]
    raise SystemExit(pytest.main(argv))


if __name__ == "__main__":
    main()
