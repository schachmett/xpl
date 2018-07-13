"""Loads the configuration ini and sets app name and version."""
# pylint: disable=invalid-name

import os
import configparser
import datetime
import sys
import logging
import logging.config
from collections import OrderedDict

__appname__ = "XPL"
__version__ = "0.3"
__authors__ = ["Simon Fischer <sfischer@ifp.uni-bremen.de>"]

BASEDIR = os.path.dirname(os.path.realpath(__file__))
CONFDIR = os.path.join(os.environ["HOME"], ".xpl")
CFG_NAME = os.path.join(CONFDIR, "config.ini")
LOGFNAME = os.path.join(CONFDIR, "xpl.log")

__config__ = configparser.ConfigParser()

COLORS = {
    "tv-highlight-bg": "#F08763",
    "axisticks": "#AAAAAA",
    "spectrum": "#B5C689",
    "region-vlines": "#F58B4C",
    "region-background": "#F58B4C",
    "rsf-annotation": "#AAAAAA",
    "rsf-vlines": ["#468CDE", "#52D273", "#E94F64", "#E57254", "#E8C454"]
}


def onload():
    """Gets called when xpl is imported."""
    make_logger()
    load_config(__config__)


def load_config(config):
    """Loads the config from file or creates a new one if that file is
    missing."""
    if not os.path.isfile(CFG_NAME):
        if not os.path.isdir(os.path.dirname(CFG_NAME)):
            os.mkdir(os.path.dirname(CFG_NAME))
        config.add_section("general")
        config.set("general", "basedir", BASEDIR)
        config.set("general", "xydir", os.path.join(CONFDIR, "xy_temp/"))
        config.set("general", "conf_filename", CFG_NAME)
        config.add_section("window")
        config.set("window", "xsize", "1200")
        config.set("window", "ysize", "700")
        config.set("window", "xpos", "200")
        config.set("window", "ypos", "200")
        config.add_section("io")
        config.set("io", "project_file", "None")
        config.set("io", "project_dir", os.environ["HOME"])
        config.set("io", "data_dir", os.environ["HOME"])
        with open(CFG_NAME, "w") as cfg_file:
            config.write(cfg_file)
    else:
        config.read(CFG_NAME)


def make_logger():
    """Configures the root logger for this application."""
    logger_conf = {
        "version": 1,
        "disable_existing_loggers": True,
        "formatters": {
            "verbose": {
                "format": "{levelname:8s} {asctime} "
                          "{lineno:5d}:{name:30s} {message}",
                "style": "{",
                },
            "brief": {
                "format": "{levelname:8s}: {message}",
                "style": "{",
                }
            },
        "handlers": {
            "console": {
                "class":"logging.StreamHandler",
                "level":"WARNING",
                "formatter": "brief",
                "stream": sys.stderr,
                },
            "file": {
                "class": "logging.handlers.RotatingFileHandler",
                "level": "DEBUG",
                "filename": LOGFNAME,
                "formatter": "verbose",
                "maxBytes": 2000000,
                "backupCount": 3,
                },
            },
        "root": {
            "handlers": ["console", "file"],
            "level":"DEBUG",
            }
        }
    logging.config.dictConfig(logger_conf)
    logging.getLogger(__name__)
    logging.getLogger("matplotlib").setLevel(logging.WARNING)


def introspect(gobject):
    """Print all Property values of a GObject.GObject."""
    for param in gobject.list_properties():
        try:
            print("{}: {}".format(
                param.name, gobject.get_property(param.name)))
        except TypeError:
            print("{} not readable".format(param.name))


onload()
