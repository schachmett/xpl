#!/usr/bin/env python3
"""Starts the program."""

import signal
import sys

from gi.repository import GLib

from xpl import __config__
from xpl.xpl import XPL


def main():
    """Runs app from xpl.xpl module."""
    app = XPL()
    GLib.unix_signal_add(GLib.PRIORITY_DEFAULT, signal.SIGINT, app.on_quit)
    exit_status = app.run(sys.argv)
    sys.exit(exit_status)


if __name__ == "__main__":
    main()

# TODO fitting is broken
# TODO get code directory for assets and glade files
# TODO make ID names meaningful
# TODO: diagram and disentangling of components
