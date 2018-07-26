# XPL

XPL is a tool for plotting and analyzing X-ray photoelectron spectroscopy (XPS) data. It can fit peaks using Pseudo Voigt profiles (more models to come) while enabling Area, Position and FWHM restrainment of the individual peaks to physically sensible values (or expressions).


## Installation
### Ubuntu

If you don't already have it installed, install python and pip as well as libffi6 and pythongi through apt.

```shell
$ sudo apt install python3 pip3
$ sudo apt install libffi6 python3-gi
$ pip3 install xpl
```

To start xpl, run it as a module. On starting, XPL creates a `~/.config/xpl` folder where configuration files and converted spectrum files will be stored.

```shell
$ python3 -m xpl
```

## Usage

You can import spectra by clicking the plus icon in the main toolbar. So far, only exportet `.txt` files from the EIS Omicron software can be parsed.

When the "Spectra" tab is selected, you can choose the spectra to view by selecting them, right clicking and clicking "Plot selected spectra". The rightmost icon in the top toolbar lets you select elements whose peak positions should be displayed (see screenshot).

![Screenshot](doc/demo_atomlib.png "Matching peaks")

In the fitting tab, you can first add one or more regions by clicking "+" next to "Regions", then dragging across the Plot View. 

When a region is selected, you can add peaks by clicking "+" next to "Peaks" and drawing them by dragging from the peak maximum downwards inside the selected region. After this, you can constrain the peak values in the bottommost three entries: They accept input either like `< min > max` where `min` and `max` are minimum and maximum values for the corresponding parameter, or like `= expression` where `expression` can be a simple arithmetic expression. A relation to the same parameter of another peak can be expressed by using their label (see screenshot below).

![Screenshot](doc/demo_fitting.png "Fitting Ag3d peaks")

Exporting the data or the plot is not yet supported.