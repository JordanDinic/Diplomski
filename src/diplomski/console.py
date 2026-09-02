from __future__ import annotations

import sys


def configure_console_output() -> None:
    """Use UTF-8 output in Windows terminals when Python allows reconfiguration."""

    for stream in (sys.stdout, sys.stderr):
        if hasattr(stream, "reconfigure"):
            stream.reconfigure(encoding="utf-8", errors="replace")
