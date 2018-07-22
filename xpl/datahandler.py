"""This module handles spectrum data through the DataHandler class that
provides methods for reading in and manipulating spectra as well as regions
defined in these spectra and peak models for fitting in these regions."""
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=logging-format-interpolation

import weakref
import copy
import logging

import numpy as np

from xpl.processing import (x_at_maximum, calibrate, normalize, smoothen,
                            calculate_background, getspan)
from xpl.fitter import RegionFitModelIface


logger = logging.getLogger(__name__)


class BaseDataHandler(object):
    """Provides basic signaling and data storage for Spectrum, Region, Peak
    objects."""
    signals = (
        "added-spectrum",
        "removed-spectrum",
        "amended-spectrum",
        "cleared-spectra",
        "changed-spectrum",
        "added-region",
        "removed-region",
        "cleared-regions",
        "changed-region",
        "added-peak",
        "removed-peak",
        "cleared-peaks",
        "changed-peak",
    )

    def __init__(self):
        self._spectra = []
        self._idbook = XPLContainer.idbook
        self._observers = dict((signal, []) for signal in self.signals)

    def children(self, ID):
        """Returns ID list of children of ID."""
        IDs = []
        cont = self._idbook[ID]
        if cont.type == "spectrum":
            IDs.extend([region.ID for region in cont.regions])
            # for region in cont.regions:
            #     IDs.append(region.ID)
            #     IDs.extend([peak.ID for peak in region.peaks])
        elif cont.type == "region":
            IDs.extend([peak.ID for peak in cont.peaks])
        return IDs

    def parent(self, ID):
        """Returns parent ID."""
        cont = self._idbook[ID]
        if cont.type == "region":
            return cont.spectrum.ID
        if cont.type == "peak":
            return cont.region.ID
        return None

    def ID_exists(self, ID):
        """Returns True if ID exists."""
        return ID in self._idbook

    def connect(self, signal, func):
        """Connects function func to be executed whenever signal is emitted."""
        try:
            self._observers[signal].append(func)
        except KeyError:
            logger.error("Signal {} does not exist".format(signal))
            raise

    def disconnect(self, signal, func):
        """Disconnects function func from signal."""
        try:
            self._observers[signal].remove(func)
        except ValueError:
            pass

    def _emit(self, signal, *data):
        """Emits a signal with additional arguments data."""
        for func in self._observers[signal]:
            func(*data)

    def isempty(self):
        """Returns True if self._spectra is empty."""
        return not self._spectra


class DataHandler(BaseDataHandler):
    """A DataHandler internally manages Spectrum, Region and Peak objects
    and provides an interface for them to an application."""
    def __init__(self):
        super().__init__()
        self.altered = False

    def get(self, ID, attr=None):
        """Gets attribute attr of object with ID."""
        cont = self._idbook[ID]
        if attr is None:
            return str(cont)
        return cont.get(attr)

    def get_multiple(self, ID, *attrs):
        """Gets attributes attrs of object with ID."""
        cont = self._idbook[ID]
        return cont.get_multiple(attrs)

    def manipulate_region(self, ID, **newdict):
        """Sets the attribute attr of container with ID."""
        region = self._idbook[ID]
        for attr, value in newdict.items():
            if attr not in ("emin", "emax", "bgtype"):
                logger.warning("attribute {} can not be changed through"
                               "DataHandler.manipulate_region()".format(attr))
                continue
            if not self.get(ID, attr) == value:
                region.set(attr, value)
                self._emit("changed-region", ID)
                self.altered = True

    def manipulate_spectrum(self, ID, **newdict):
        """Manipulates data that leads to recalculation of other data or
        direct need for replotting."""
        spectrum = self._idbook[ID]
        for attr, value in newdict.items():
            if attr not in ("calibration", "norm", "smoothness"):
                logger.warning("attribute {} can not be changed through"
                               "DataHandler.manipulate_spectrum()"
                               "".format(attr))
                continue
            if not self.get(ID, attr) == value:
                spectrum.set(attr, value)
                self.altered = True
                self._emit("changed-spectrum", ID, attr)

    def add_spectrum(self, specdict):
        """Adds a spectrum from a dictionary defining its attributes."""
        spectrum = Spectrum(specdict)
        self._spectra.append(spectrum)
        self.altered = True
        self._emit("added-spectrum", spectrum.ID)
        return spectrum.ID

    def remove_spectrum(self, ID):
        """Removes spectrum that is identified by ID."""
        if self.get(ID, "type") != "spectrum":
            logger.error("object with ID {} is no spectrum".format(ID))
            raise TypeError
        self._emit("removed-spectrum", ID)
        self.altered = True
        self._spectra.remove(self._idbook[ID])

    def amend_spectrum(self, ID, newdict):
        """Changes values of a spectrum."""
        if self.get(ID, "type") != "spectrum":
            logger.error("object with ID {} is no spectrum".format(ID))
            raise TypeError
        if "raw_sweeps" in newdict or "raw_dwelltime" in newdict:
            newdict["int_time"] = str(
                int(newdict.get("raw_sweeps", self.get(ID, "raw_sweeps")))
                * float(newdict.get("raw_dwelltime",
                                    self.get(ID, "raw_dwelltime")))
            )

        self._emit("amended-spectrum", ID, newdict)

        spectrum = self._idbook[ID]
        for attr, value in newdict.items():
            if attr == "ID":
                continue
            if attr in ("int_time", "raw_dwelltime", "pass_energy"):
                value = float(value)
            elif attr in ("raw_sweeps", "eis_region"):
                value = int(value)
            if not self.get(ID, attr) == value:
                spectrum.set(attr, value)
                self.altered = True
                if attr in ("int_time",):
                    self._emit("changed-spectrum", ID)

    def clear_spectra(self):
        """Removes all spectra."""
        self.altered = False
        self._spectra.clear()
        self._emit("cleared-spectra")

    def add_region(self, ID, **regiondict):
        """Adds a region to spectrum with ID."""
        if self.get(ID, "type") != "spectrum":
            logger.error("object with ID {} is no spectrum".format(ID))
            raise TypeError
        spectrum = self._idbook[ID]
        regiondict["spectrum"] = spectrum
        # region = Region(regiondict)
        regionID = spectrum.add_region(regiondict)
        self._emit("added-region", regionID)
        self.altered = True
        return regionID

    def remove_region(self, ID):
        """Removes a region."""
        if self.get(ID, "type") != "region":
            logger.error("object with ID {} is no region".format(ID))
            raise TypeError
        region = self._idbook[ID]
        region.spectrum.regions.remove(region)
        del region
        self._emit("removed-region", ID)
        self.altered = True

    def clear_regions(self, ID):
        """Clears regions of spectrum with ID."""
        if self.get(ID, "type") != "spectrum":
            logger.error("object with ID {} is no spectrum".format(ID))
            raise TypeError
        self._idbook[ID].regions.clear()
        self._emit("cleared-regions", ID)
        self.altered = True

    def add_peak(self, ID, **peakdict):
        """Adds a peak to region with ID."""
        if self.get(ID, "type") != "region":
            logger.error("object with ID {} is no region".format(ID))
            raise TypeError
        region = self._idbook[ID]
        peakdict["region"] = region
        peakID = region.add_peak(peakdict)
        self._emit("added-peak", peakID)
        self.altered = True
        return peakID

    def remove_peak(self, ID):
        """Removes a peak."""
        if self.get(ID, "type") != "peak":
            logger.error("object with ID {} is no peak".format(ID))
            raise TypeError
        peak = self._idbook[ID]
        peak.region.remove_peak(peak)
        del peak
        self._emit("removed-peak", ID)
        self.altered = True

    def clear_peaks(self, ID):
        """Clears regions of spectrum with ID."""
        if self.get(ID, "type") != "region":
            logger.error("object with ID {} is no region".format(ID))
            raise TypeError
        self._idbook[ID].clear_peaks()
        self._emit("cleared-peaks", ID)
        self.altered = True

    def fit_region(self, regionID):
        """Fits peaks in a given region to the cps."""
        if self.get(regionID, "type") != "region":
            logger.error("object with ID {} is no region".format(regionID))
            raise TypeError
        self._idbook[regionID].fit()
        self._emit("changed-peak", regionID)
        self.altered = True

    def load(self, spectra):
        """Loads spectra after deleting all contents."""
        self.clear_spectra()
        for spectrum in spectra:
            spectrum.new_ID()
            self._spectra.append(spectrum)
            self._emit("added-spectrum", spectrum.ID)
        self.altered = False

    def save(self):
        """Returns all contents for pickling."""
        self.altered = False
        return self._spectra

    # def by_attr(self, **kwargs):
    #     """Returns list of ContainerIDs by searching through attributes."""
    #     IDs = []
    #     for ID, cont in self._idbook.items():
    #         for attr, value in kwargs.items():
    #             if cont.get(attr) != value:
    #                 break
    #         else:
    #             IDs.append(ID)
    #     return IDs


class XPLContainer(object):
    """A generic container that stores spectra, regions or peaks. Should be
    subclassed."""
    _required = ()
    _defaults = {}
    _newID = 0
    idbook = weakref.WeakValueDictionary({})

    def __init__(self, attrdict):
        for attr in self._required:
            if attr not in attrdict:
                raise ValueError("Attribute '{}' missing".format(attr))
        self.new_ID()

        self.type = "notype"
        for (attr, default) in self._defaults.items():
            setattr(self, attr, attrdict.get(attr, default))

        self.specdict_additional = dict([
            (attr, default) for (attr, default) in attrdict.items()
            if attr not in self._defaults
            and attr not in self._required])

    def new_ID(self):
        """Fetches a new ID."""
        self.ID = XPLContainer._newID
        XPLContainer._newID += 1
        XPLContainer.idbook[self.ID] = self

    def set(self, attr, value):
        """Sets an attribute."""
        try:
            setattr(self, attr, value)
        except AttributeError:
            try:
                self.specdict_additional[attr] = value
            except KeyError:
                logger.error("Object {} has no attribute {}"
                             "".format(self.type, attr))
                raise

    def get(self, attr):
        """Gets attribute value. If the attribute does not exist, None is
        returned."""
        try:
            return getattr(self, attr)
        except AttributeError:
            try:
                return self.specdict_additional[attr]
            except KeyError:
                raise

    def get_multiple(self, attrs):
        """Gets attribute value. If the attribute does not exist, None is
        returned."""
        values = []
        for attr in attrs:
            try:
                values.append(getattr(self, attr))
            except AttributeError:
                try:
                    values.append(self.specdict_additional[attr])
                except KeyError:
                    values.append(None)
        return values

    def __eq__(self, other):
        """Tests equality."""
        if isinstance(other, XPLContainer):
            return self.ID == other.ID
        return False

    def __repr__(self):
        return "{}({})".format(self.__class__, self.__dict__)

    def __str__(self):
        strlist = []
        strlist.append("{}\n".format(self.__class__))
        def pretty(d, indent=1):
            """Returns a pretty string of a nested dictionary."""
            for key, value in d.items():
                strlist.append("\t" * indent + "{}:\n".format(str(key)))
                if key in ("regions", "peaks"):
                    IDlist = [child.ID for child in value]
                    strlist.append("\t" * (indent + 1) + str(IDlist) + "\n")
                    continue
                elif key in ("spectrum", "region"):
                    ID = value.ID
                    strlist.append("\t" * (indent + 1) + str(ID) + "\n")
                    continue
                if isinstance(value, dict):
                    pretty(value, indent=indent+1)
                elif len(str(value)) > 80:
                    strlist.append(
                        "\t" * (indent + 1) + str(value)[:80] + "\n")
                else:
                    strlist.append("\t" * (indent + 1) + str(value) + "\n")
        pretty(self.__dict__)
        return "".join(strlist)


# pylint: disable=no-member

class Spectrum(XPLContainer):
    """A spectrum container."""
    _required = ("energy", "cps")
    _defaults = {
        "name": "",
        "notes": "",
        "filename": "",
        "int_time": 0,
        "pass_energy": 0
    }

    def __init__(self, specdict):
        specdict["parent"] = None
        super().__init__(specdict)
        self.type = "spectrum"

        self._raw_energy = specdict["energy"]
        self.energy_c = copy.deepcopy(self._raw_energy)
        self.energy = copy.deepcopy(self._raw_energy)
        self._raw_cps = specdict["cps"]
        self.cps_c = copy.deepcopy(self._raw_cps)
        self.cps = copy.deepcopy(self._raw_cps)

        self.regions = []
        self.region_number = 1
        self.calibration = 0
        self.norm = False
        self.smoothness = 0

    def set(self, attr, value):
        """Overload set for some special attributes."""
        if attr in ("int_time",):
            self._raw_cps = self._raw_cps / self.int_time * value
        super().set(attr, value)
        if attr in ("smoothness", "calibration", "norm", "int_time"):
            self.calculate_cps()

    def add_region(self, regiondict):
        """Adds Region region to this spectrum."""
        region = Region(regiondict)
        self.regions.append(region)
        self.region_number += 1
        return region.ID

    def remove_region(self, region):
        """Removes Region region from this spectrum."""
        self.regions.remove(region)

    def clear_regions(self):
        """Removes all Regions from this spectrum."""
        self.regions.clear()

    def get_energy_at_maximum(self, span):
        """Returns energy value at the maximum intensity point in span, which
        is a tuple containing min and max energy value."""
        maxen = x_at_maximum(self.energy_c, self.cps_c, span)
        return maxen

    def calculate_cps(self):
        """Smoothes the displayed intensity by moving average, normalizes
        highest cps to 1 and shifts energy axis by calibration. The raw
        intensity values are kept."""
        if self.calibration:
            self.energy_c = calibrate(self._raw_energy, self.calibration)
        else:
            self.energy_c = self._raw_energy
        if self.norm:
            self.cps_c = normalize(self._raw_cps, self.norm)
        else:
            self.cps_c = self._raw_cps
        if self.smoothness:
            self.cps = smoothen(self.cps_c, self.smoothness)
        else:
            self.cps = self.cps_c
        self.energy = self.energy_c


class Region(XPLContainer):
    """A region container."""
    bgtypes = ("none", "shirley", "linear")
    _required = ("spectrum", "emin", "emax")
    _defaults = {
        "name": ""
    }

    def __init__(self, regiondict):
        super().__init__(regiondict)
        self.type = "region"

        self.spectrum = regiondict["spectrum"]
        self.eminmax = (regiondict["emin"], regiondict["emax"])

        self.energy, self.cps = getspan(
            self.spectrum.energy_c,
            self.spectrum.cps_c,
            self.eminmax)

        self.bgtype = "shirley"
        self.background = None
        self.calculate_bg(self.bgtype)

        self.peaks = []
        self.peak_number = 0
        self.model = RegionFitModelIface(self)
        self.fit_all = None

        if "name" not in regiondict:
            self.name = "Region {}".format(self.spectrum.region_number)

    def set(self, attr, value):
        """Overload set for some special attributes."""
        if attr == "emin":
            attr = "eminmax"
            value = (value, self.eminmax[1])
        elif attr == "emax":
            attr = "eminmax"
            value = (self.eminmax[0], value)
        super().set(attr, value)
        if attr == "bgtype":
            self.calculate_bg(value)
        if attr in ("spectrum", "eminmax"):
            self.energy, self.cps = getspan(
                self.spectrum.energy_c,
                self.spectrum.cps_c,
                self.eminmax)
            self.calculate_bg(self.bgtype)

    def get(self, attr):
        """Overload get for emin/emax."""
        if attr == "emin":
            return self.eminmax[0]
        if attr == "emax":
            return self.eminmax[1]
        return super().get(attr)

    def calculate_bg(self, bgtype):
        """Calculates a self.background."""
        if bgtype not in self.bgtypes:
            raise ValueError("background type {} doesn't exist".format(bgtype))
        self.background = calculate_background(bgtype, self.energy, self.cps)

    def fit(self):
        """Fits the sum of peaks to self._cps - self.background, then
        stores fit results."""
        self.model.fit()

    @property
    def fit_cps(self):
        """Fetches the evaluation of the total model from ModelIface."""
        return self.model.get_cps()

    def add_peak(self, peakdict):
        """Adds Peak peak to this region."""
        if "totalheight" in peakdict:
            bg_at_center = self.background[
                np.abs(self.energy - peakdict["center"]).argmin()]
            height = peakdict.pop("totalheight") - bg_at_center
            peakdict["height"] = height
        if "angle" in peakdict:
            fwhm = np.tan(peakdict.pop("angle")) * peakdict["height"]
            peakdict["fwhm"] = fwhm

        peak = Peak(peakdict)
        self.peaks.append(peak)
        self.model.add_peak(peak)
        self.peak_number += 1
        return peak.ID

    def remove_peak(self, peak):
        """Removes Peak peak from this region."""
        self.peaks.remove(peak)
        self.model.remove_peak(peak)

    def clear_peaks(self):
        """Removes all Peaks from this region."""
        for peak in self.peaks:
            self.model.remove_peak(peak)
        self.peaks.clear()


class Peak(XPLContainer):
    """A peak container."""
    # pylint: disable=attribute-defined-outside-init
    peaknames = "ABCDEFGHIJKLMNOPQRSTUVWXYZ"
    _required = ("region",)
    _defaults = {
        "name": "",
        "model_name": "PseudoVoigt",
        "_area": None,
        "_height": None,
        "fwhm": None,
        "center": None
    }

    def __init__(self, peakdict):
        if "height" in peakdict:
            height = peakdict.pop("height")
            peakdict["_height"] = height
        if "area" in peakdict:
            area = peakdict.pop("area")
            peakdict["_area"] = area
        super().__init__(peakdict)
        self.type = "peak"

        self.region = peakdict["region"]

        self._constraints = []
        self.prefix = "p{}_".format(self.ID)

        if "name" not in peakdict:
            self.name = "Peak {}".format(
                self.peaknames[self.region.peak_number])

        self.model = self.region.model
        if None in (self.height, self.area, self.fwhm, self.center):
            self.model.guess_params(self)
        else:
            self.model.init_params(
                self,
                fwhm=self.fwhm,
                area=self.area,
                center=self.center
            )

    @property
    def area(self):
        """Returns self._area, important job is in the setter."""
        if not self._area and self._height:
            area = (
                self._height
                * (self.fwhm * np.sqrt(np.pi / np.log(2)))
                / (1 + np.sqrt(1 / (np.pi * np.log(2))))
            )
            self._area = area
        return self._area

    @area.setter
    def area(self, value):
        """Sets self._area and unsets self._height."""
        self._area = value
        height = (
            self._area
            / (self.fwhm * np.sqrt(np.pi / np.log(2)))
            * (1 + np.sqrt(1 / (np.pi * np.log(2))))
        )
        self._height = height

    @property
    def height(self):
        """Returns self._height, important job is in the setter."""
        if not self._height and self._area:
            height = (
                self._area
                / (self.fwhm * np.sqrt(np.pi / np.log(2)))
                * (1 + np.sqrt(1 / (np.pi * np.log(2))))
            )
            self._height = height
        return self._height

    @height.setter
    def height(self, value):
        """Sets self._height and self._area."""
        self._height = value
        area = (
            self._height
            * (self.fwhm * np.sqrt(np.pi / np.log(2)))
            / (1 + np.sqrt(1 / (np.pi * np.log(2))))
        )
        self._area = area

    @property
    def fit_cps(self):
        """Fetches peak cps from the ModelIface."""
        return self.model.get_peak_cps(self)

    def set_params_from_model(self, **kwargs):
        """Setter for peak paramaters to use from the fitting interface."""
        for key, value in kwargs.items():
            if key not in ("height", "area", "center", "fwhm"):
                raise AttributeError("model can only change peak parameters")
            setattr(self, key, value)

    def set(self, attr, value):
        """Overload set for some special attributes. (setter
        for DataHandler)"""
        super().set(attr, value)
        if attr == "model_name":
            pass
        if attr in ("fwhm", "area", "center"):
            pass
            # self.model.init_params(
            #     self,
            #     fwhm=self.fwhm,
            #     area=self.area,
            #     center=self.center
            # )

    def add_constraints(self, constraints):
        """Adds the constraints written in the tuple constraints."""
        pass
