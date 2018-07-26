"""Setup script."""
# pylint: disable=invalid-name

from setuptools import setup, find_packages
from xpl import __version__, __appname__, __website__


assert __appname__ == "XPL"

with open("README.md", "r") as rmf:
    long_description = rmf.read()

setup(
    name="xpl",
    version=__version__,
    author="Simon Fischer",
    description="XPS spectrum analysis",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url=__website__,
    packages=find_packages(exclude=["doc", "tests", ".git"]),
    classifiers=(
        "Programming Language :: Python :: 3",
        "Environment :: X11 Applications :: GTK",
        "License :: OSI Approved :: GNU General Public License v3 (GPLv3)",
        "Topic :: Scientific/Engineering :: Physics"
    ),
    keywords="physics XPS x-ray photoelectron spectroscopy chemistry",
    install_requires=[
        "matplotlib",
        "numpy",
        "cairocffi",
        "lmfit"
    ],
    python_requires="~=3.5",
    package_data={
        "assets": ["atom_lib.png", "logo.svg", "logo48.png", "pan.png"],
        "xpl": ["menubar.ui", "rsf.db", "xpl.glade", "xpl_catalog.xml"]
    }
)
