"""This module handles spectrum data through the DataHandler class that
provides methods for reading in and manipulating spectra as well as regions
defined in these spectra and peak models for fitting in these regions."""
# pylint: disable=invalid-name
# pylint: disable=too-many-instance-attributes
# pylint: disable=logging-format-interpolation

import weakref
import copy
import logging
from string import ascii_uppercase

import numpy as np

from xpl.processing import (x_at_maximum, calibrate, normalize, smoothen,
                            calculate_background)
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
        "fit-region",
        "added-peak",
        "removed-peak",
        "cleared-peaks",
        "changed-peak",
        "altered"
    )

    def __init__(self):
        self._spectra = []
        self._idbook = XPLContainer.idbook
        self._observers = dict((signal, []) for signal in self.signals)
        self.altered = False
        self.connect("altered", self.alter)

    def alter(self, isaltered):
        """Changes altered value when the signal is emitted."""
        self.altered = isaltered

    def emit_init_ready(self):
        """Emits cleared-spectra so that views refresh the first time
        at application startup."""
        self._emit("cleared-spectra", None)

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

    def children(self, ID):
        """Returns ID list of children of ID."""
        if ID is None:
            return [spectrum.ID for spectrum in self._spectra]
        cont = self._idbook[ID]
        if cont.type == "spectrum":
            return [region.ID for region in cont.regions]
        elif cont.type == "region":
            return [peak.ID for peak in cont.peaks]
        elif cont.type == "peak":
            return []
        raise TypeError("ID {} not valid".format(ID))

    def parent(self, ID):
        """Returns parent ID."""
        cont = self._idbook[ID]
        if cont.type == "region":
            return cont.spectrum.ID
        if cont.type == "peak":
            return cont.region.ID
        return None

    def isspectrum(self, ID):
        """Tests whether ID belongs to spectrum."""
        return self.get(ID, "type") == "spectrum"

    def isregion(self, ID):
        """Tests whether ID belongs to region."""
        return self.get(ID, "type") == "region"

    def ispeak(self, ID):
        """Tests whether ID belongs to spectrum."""
        return self.get(ID, "type") == "peak"

    def exists(self, ID):
        """Returns True if ID exists."""
        return ID in self._idbook

    def connect(self, signal, func):
        """Connects function func to be executed whenever signal is emitted."""
        try:
            self._observers[signal].append(func)
        except KeyError:
            raise

    def disconnect(self, signal, func):
        """Disconnects function func from signal."""
        try:
            self._observers[signal].remove(func)
        except ValueError:
            pass

    def _emit(self, signal, *data, **kwdata):
        """Emits a signal with additional arguments data."""
        for func in self._observers[signal]:
            func(*data, **kwdata)

    def isempty(self):
        """Returns True if self._spectra is empty."""
        return not self._spectra

    def by_attr(self, **kwargs):
        """Returns list of ContainerIDs by searching through attributes."""
        IDs = []
        for ID, cont in self._idbook.items():
            for attr, value in kwargs.items():
                try:
                    if cont.get(attr) != value:
                        break
                except AttributeError:
                    break
            else:
                IDs.append(ID)
        return IDs


class DataHandler(BaseDataHandler):
    """A DataHandler internally manages Spectrum, Region and Peak objects
    and provides an interface for them to an application."""
    def fit_region(self, regionID):
        """Fits peaks in a given region to the cps."""
        assert self.isregion(regionID)
        self._idbook[regionID].fit()
        self._emit("fit-region", regionID)
        self._emit("altered", True)

    def manipulate_spectrum(self, spectrumID, **newdict):
        """Manipulates data that leads to recalculation of other data or
        direct need for replotting."""
        assert self.isspectrum(spectrumID)
        spectrum = self._idbook[spectrumID]
        for attr, value in newdict.items():
            if attr not in ("calibration", "norm", "smoothness"):
                logger.warning("attribute {} can not be changed through"
                               "DataHandler.manipulate_spectrum()"
                               "".format(attr))
                continue
            if not self.get(spectrumID, attr) == value:
                spectrum.set(attr, value)
                self._emit("altered", True)
                self._emit("changed-spectrum", spectrumID, attr)

    def manipulate_region(self, regionID, **newdict):
        """Sets the attribute attr of container with ID."""
        assert self.isregion(regionID)
        region = self._idbook[regionID]
        for attr, value in newdict.items():
            if attr not in ("emin", "emax", "bgtype"):
                logger.warning("attribute {} can not be changed through"
                               "DataHandler.manipulate_region()".format(attr))
                continue
            if not self.get(regionID, attr) == value:
                region.set(attr, value)
                self._emit("changed-region", regionID, attr)
                self._emit("altered", True)

    def constrain_peak(self, peakID, attr, **constraints):
        """Sets constrains for a peak."""
        assert self.ispeak(peakID)
        peak = self._idbook[peakID]
        isvalid = peak.set_constraints(attr, **constraints)
        self._emit("changed-peak", peakID, attr, isvalid)
        self._emit("altered", True)

    def get_peak_constraints(self, peakID, attr):
        """Gets constraints from a peak."""
        assert self.ispeak(peakID)
        peak = self._idbook[peakID]
        return peak.get_constraints(attr)

    def load(self, spectra):
        """Loads spectra after deleting all contents."""
        self.clear_spectra()
        for spectrum in spectra:
            spectrum.new_ID()
            self._spectra.append(spectrum)
            for region in spectrum.regions:
                region.new_ID()
                for peak in region.peaks:
                    peak.new_ID()
            self._emit("added-spectrum", spectrum.ID)
        self._emit("altered", False)

    def save(self):
        """Returns all contents for pickling."""
        self._emit("altered", False)
        return self._spectra

    def add_spectrum(self, **specdict):
        """Adds a spectrum from a dictionary defining its attributes."""
        spectrum = Spectrum(**specdict)
        self._spectra.append(spectrum)
        self._emit("added-spectrum", spectrum.ID)
        self._emit("altered", True)
        return spectrum.ID

    def add_averaged_spectrum(self, spectrumIDs):
        """Adds a new spectrum which is the average of the given ones."""
        for spectrumID in spectrumIDs:
            self.isspectrum(spectrumID)
        spectra = [self._idbook[spectrumID] for spectrumID in spectrumIDs]
        first_energy = spectra[0].get("raw_energy")
        for spectrum in spectra:
            if any(spectrum.get("raw_energy") != first_energy):
                logger.warning("spectra {} averaging failed: energies"
                               "don't match".format(spectrumIDs))
                return None
        energy = first_energy
        intensity = np.sum(
            [s.get("cps") * s.get("int_time") for s in spectra], axis=0)
        sweeps = np.sum([s.get("sweeps") for s in spectra])
        int_time = np.sum([s.get("int_time") for s in spectra])
        pass_energies = [s.get("pass_energy") for s in spectra]
        pass_energy = pass_energies[0] if len(set(pass_energies)) == 1 else 0
        notes = ", ".join([s.get("name") for s in spectra])
        specdict = {
            "filename": "created_by_XPL",
            "energy": energy,
            "intensity": intensity,
            "sweeps": sweeps,
            "int_time": int_time,
            "dwelltime": int_time / sweeps,
            "pass_energy": pass_energy,
            "notes": notes,
        }
        spectrum = Spectrum(**specdict)
        spectrum.name = "AVG{}".format(spectrum.ID)
        self._spectra.append(spectrum)
        self._emit("added-spectrum", spectrum.ID)
        self._emit("altered", True)
        return spectrum.ID

    def remove_spectrum(self, spectrumID):
        """Removes spectrum that is identified by ID."""
        assert self.isspectrum(spectrumID)
        spectrum = self._idbook.pop(spectrumID)
        for region in spectrum.regions:
            self._idbook.pop(region.ID)
        spectrum.clear_regions()
        self._spectra.remove(spectrum)
        self._emit("removed-spectrum", spectrumID)
        self._emit("altered", True)

    def amend_spectrum(self, spectrumID, **newdict):
        """Changes values of a spectrum."""
        assert self.isspectrum(spectrumID)

        if "sweeps" in newdict or "dwelltime" in newdict:
            newdict["int_time"] = str(
                int(newdict.get("sweeps", self.get(spectrumID, "sweeps")))
                * float(newdict.get("dwelltime",
                                    self.get(spectrumID, "dwelltime")))
            )
        spectrum = self._idbook[spectrumID]
        for attr, value in newdict.items():
            if attr == "ID":
                continue
            if attr in ("calibration", "norm", "smoothness"):
                logger.error("attribute {} can not be changed through"
                             "DataHandler.amend_spectrum()".format(attr))
                continue
            elif attr in ("int_time", "dwelltime", "pass_energy"):
                value = float(value)
            elif attr in ("sweeps", "eis_region"):
                value = int(value)
            if not self.get(spectrumID, attr) == value:
                spectrum.set(attr, value)
                self._emit("changed-spectrum", spectrumID, attr)
                self._emit("altered", True)

    def clear_spectra(self):
        """Removes all spectra."""
        for spectrum in self._spectra:
            self._idbook.pop(spectrum.ID)
            for region in spectrum.regions:
                self._idbook.pop(region.ID)
                for peak in region.peaks:
                    self._idbook.pop(peak.ID)
        self._spectra.clear()
        self._emit("cleared-spectra", None)
        self._emit("altered", False)

    def add_region(self, spectrumID, **regiondict):
        """Adds a region to spectrum with ID."""
        assert self.isspectrum(spectrumID)
        spectrum = self._idbook[spectrumID]
        regiondict["spectrum"] = spectrum
        regionID = spectrum.add_region(**regiondict)
        self._emit("added-region", regionID)
        self._emit("altered", True)
        return regionID

    def remove_region(self, regionID):
        """Removes a region."""
        assert self.isregion(regionID)
        region = self._idbook.pop(regionID)
        for peak in region.peaks:
            self._idbook.pop(peak.ID)
        region.clear_peaks()
        region.spectrum.remove_region(region)
        self._emit("removed-region", regionID)
        self._emit("altered", True)

    def clear_regions(self, spectrumID):
        """Clears regions of spectrum with ID."""
        assert self.isspectrum(spectrumID)
        spectrum = self._idbook[spectrumID]
        for region in spectrum.regions:
            self._idbook.pop(region.ID)
            for peak in region.peaks:
                self._idbook.pop(peak.ID)
        spectrum.clear_regions()
        self._emit("cleared-regions", spectrumID)
        self._emit("altered", True)

    def add_peak(self, regionID, **peakdict):
        """Adds a peak to region with ID."""
        assert self.isregion(regionID)
        region = self._idbook[regionID]
        peakdict["region"] = region
        peakID = region.add_peak(**peakdict)
        self._emit("added-peak", peakID)
        self._emit("altered", True)
        return peakID

    def remove_peak(self, peakID):
        """Removes a peak."""
        assert self.ispeak(peakID)
        peak = self._idbook.pop(peakID)
        peak.region.remove_peak(peak)
        self._emit("removed-peak", peakID)
        self._emit("altered", True)

    def amend_peak(self, peakID, **newdict):
        """Changes peak attributes that don't affect the plot."""
        assert self.ispeak(peakID)
        for attr, value in newdict.items():
            if attr not in ("name", ):
                continue
            if not self.get(peakID, attr) == value:
                peak = self._idbook[peakID]
                peak.set(attr, value)
                self._emit("changed-peak", peakID, attr, True)
                self._emit("altered", True)

    def clear_peaks(self, regionID):
        """Clears regions of spectrum with ID."""
        assert self.isregion(regionID)
        region = self._idbook[regionID]
        for peak in region.peaks:
            self._idbook.pop(peak.ID)
        region.clear_peaks()
        self._emit("cleared-peaks", regionID)
        self._emit("altered", True)


class XPLContainer(object):
    """A generic container that stores spectra, regions or peaks. Should be
    subclassed."""
    _required = ()
    _defaults = {}
    _newID = 0
    idbook = weakref.WeakValueDictionary({})

    def __init__(self, **attrdict):
        for attr in self._required:
            if attr not in attrdict:
                raise ValueError("Attribute '{}' missing".format(attr))
        self.new_ID()

        self.type = "notype"
        for (attr, default) in self._defaults.items():
            setattr(self, attr, attrdict.pop(attr, default))

    def new_ID(self):
        """Fetches a new ID."""
        self.ID = XPLContainer._newID
        XPLContainer._newID += 1
        XPLContainer.idbook[self.ID] = self

    def set(self, attr, value):
        """Sets an attribute."""
        setattr(self, attr, value)

    def get(self, attr):
        """Gets attribute value. If the attribute does not exist, None is
        returned."""
        return getattr(self, attr)

    def get_multiple(self, attrs):
        """Gets attribute value. If the attribute does not exist, None is
        returned."""
        values = []
        for attr in attrs:
            values.append(getattr(self, attr))
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


class Spectrum(XPLContainer):
    """A spectrum container."""
    # pylint: disable=no-member
    # pylint: disable=access-member-before-definition
    _required = ("energy", "intensity")
    _defaults = {
        "name": "",
        "notes": "",
        "filename": "",
        "int_time": 1.0,
        "sweeps": 1,
        "dwelltime": 1.0,
        "pass_energy": 0.0,
        "eis_region": 0,
    }

    def __init__(self, **specdict):
        super().__init__(**specdict)

        self.type = "spectrum"
        if not self.name:
            if self.eis_region:
                self.name = "EIS Spectrum {}".format(self.eis_region)
            else:
                self.name = "SPEC {}".format(self.ID)

        self.int_time = self.dwelltime * self.sweeps

        issorted = all(np.diff(specdict["energy"] >= 0))
        self.raw_energy = np.array(sorted(specdict.pop("energy")))
        self.energy_c = copy.deepcopy(self.raw_energy)
        self.energy = copy.deepcopy(self.raw_energy)
        self.intensity = np.array(specdict.pop("intensity"))
        if not issorted:
            self.intensity = self.intensity[::-1]
        self.raw_cps = self.intensity / self.int_time
        self.cps_c = copy.deepcopy(self.raw_cps)
        self.cps = copy.deepcopy(self.raw_cps)

        self.regions = []
        self.region_number = 1
        self.calibration = 0
        self.norm = False
        self.smoothness = 0

        assert not [attr for attr in specdict if attr not in self._defaults]

    def set(self, attr, value):
        """Overload set for some special attributes."""
        if attr in ("int_time",):
            self.raw_cps = self.raw_cps / self.int_time * value
        super().set(attr, value)
        if attr in ("smoothness", "calibration", "norm", "int_time"):
            self.calculate_cps()

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
            self.energy_c = calibrate(self.raw_energy, self.calibration)
            for region in self.regions:
                region.set("emin", region.emin)
                region.set("emax", region.emax)
        else:
            self.energy_c = self.raw_energy
        if self.norm:
            self.cps_c = normalize(self.raw_cps, self.norm)
        else:
            self.cps_c = self.raw_cps
        if self.smoothness:
            self.cps = smoothen(self.cps_c, self.smoothness)
        else:
            self.cps = self.cps_c
        self.energy = self.energy_c

    def add_region(self, **regiondict):
        """Adds Region region to this spectrum."""
        region = Region(**regiondict)
        self.regions.append(region)
        self.region_number += 1
        return region.ID

    def remove_region(self, region):
        """Removes Region region from this spectrum."""
        self.regions.remove(region)

    def clear_regions(self):
        """Removes all Regions from this spectrum."""
        self.regions.clear()


class Region(XPLContainer):
    """A region container."""
    # pylint: disable=no-member
    # pylint: access-member-before-definition
    bgtypes = ("none", "shirley", "linear")
    _required = ("spectrum", "emin", "emax")
    _defaults = {
        "name": ""
    }

    def __init__(self, **regiondict):
        super().__init__(**regiondict)
        self.type = "region"

        self.spectrum = regiondict.pop("spectrum")
        self.emin = float(regiondict.pop("emin"))
        self.emax = float(regiondict.pop("emax"))

        self.bgtype = "shirley"
        self.background = None
        self.calculate_bg(self.bgtype)

        self.peaks = []
        self.peak_number = 1
        self.model = RegionFitModelIface(self)
        self.fit_all = None

        if not self.name:
            self.name = "Region {}".format(self.spectrum.region_number)

        assert not [attr for attr in regiondict if attr not in self._defaults]

    def set(self, attr, value):
        """Overload set for some special attributes."""
        if attr == "emin":
            if value < min(self.spectrum.energy):
                value = min(self.spectrum.energy)
        elif attr == "emax":
            if value > max(self.spectrum.energy):
                value = max(self.spectrum.energy)
        super().set(attr, value)
        if attr == "bgtype":
            self.calculate_bg(value)
        if attr in ("spectrum", "emin", "emax"):
            self.calculate_bg(self.bgtype)

    def get_peak_by_label(self, peaklabel):
        """Returns the peak corresponding to peaklabel."""
        for peak in self.peaks:
            if peak.label == peaklabel:
                return peak
        return None

    def get_peak_by_prefix(self, peakprefix):
        """Returns the peak corresponding to peak.prefix."""
        for peak in self.peaks:
            if peak.prefix == peakprefix:
                return peak
        return None

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
    def energy(self):
        """Energy slice from the spectrum."""
        idx1, idx2 = sorted([
            np.searchsorted(self.spectrum.energy_c, self.emin),
            np.searchsorted(self.spectrum.energy_c, self.emax)
        ])
        return self.spectrum.energy_c[idx1:idx2]

    @property
    def cps(self):
        """CPS slice from the spectrum."""
        idx1, idx2 = sorted([
            np.searchsorted(self.spectrum.energy_c, self.emin),
            np.searchsorted(self.spectrum.energy_c, self.emax)
        ])
        return self.spectrum.cps_c[idx1:idx2]

    @property
    def fit_cps(self):
        """Fetches the evaluation of the total model from ModelIface."""
        return self.model.get_cps(self.energy)

    @property
    def fit_cps_fullrange(self):
        """Fetches the evaluation of the total model from ModelIFace
        over the full energy range of the spectrum.
        """
        return self.model.get_cps(self.spectrum.energy_c)

    def add_peak(self, **peakdict):
        """Adds Peak peak to this region."""
        if "totalheight" in peakdict:
            bg_at_center = self.background[
                np.abs(self.energy - peakdict["center"]).argmin()]
            height = peakdict.pop("totalheight") - bg_at_center
            peakdict["height"] = height
        if "angle" in peakdict:
            fwhm = np.tan(peakdict.pop("angle")) * peakdict["height"]
            peakdict["fwhm"] = fwhm

        peak = Peak(**peakdict)
        self.peaks.append(peak)
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
    # pylint: disable=no-member
    # pylint: disable=attribute-defined-outside-init
    # pylint: disable=access-member-before-definition
    _required = ("region",)
    _defaults = {
        "name": "",
        "model_name": "PseudoVoigt"
    }

    def __init__(self, **peakdict):
        super().__init__(**peakdict)
        self.region = peakdict.pop("region")
        self.model = self.region.model
        self.prefix = "p{}_".format(self.ID)
        self.label = "P{}".format(self.region.peak_number)
        self.type = "peak"

        if peakdict.pop("guess", None):
            self.model.guess_params(self)
        else:
            # Order matters
            self.fwhm = peakdict.pop("fwhm")
            self.center = peakdict.pop("center")
            if "area" in peakdict:
                self.area = peakdict.pop("area")
            elif "height" in peakdict:
                self.height = peakdict.pop("height")

        if not self.name:
            self.name = "Peak {}".format(
                ascii_uppercase[self.region.peak_number])

        assert not [attr for attr in peakdict if attr not in self._defaults]

    def set_constraints(self, param, **constraints):
        """Adds the constraints written in the dict constraints.
        """
        constraints = {k: v for k, v in constraints.items() if v is not None}
        value = constraints.get("value", None)
        vary = constraints.get("vary", True)
        min_ = constraints.get("min_", 0)
        max_ = constraints.get("max_", np.inf)
        expr = constraints.get("expr", "")
        logger.info("set peak {} '{}' constraints: min={}, max={}, expr={}"
                    "".format(self.ID, param, min_, max_, expr))
        return self.model.set_constraints(
            self, param, value, vary, min_, max_, expr
        )

    def get_constraints(self, attr):
        """Returns a dictionary containing the constraints.
        """
        return self.model.get_constraints(self, attr)

    @property
    def area(self):
        """Returns value of model area parameter.
        """
        return self.model.get_value(self, "area")

    @area.setter
    def area(self, value):
        """Sets the value of model area parameter in the model.
        """
        self.model.set_value(self, "area", value)

    @property
    def height(self):
        """Returns self._height, important job is in the setter."""
        return self.model.get_value(self, "height")

    @height.setter
    def height(self, value):
        """Sets self._height and self._area."""
        self.model.set_value(self, "height", value)

    @property
    def fwhm(self):
        """Gets fwhm inside model.
        """
        return self.model.get_value(self, "fwhm")

    @fwhm.setter
    def fwhm(self, value):
        """Sets fwhm inside model.
        """
        self.model.set_value(self, "fwhm", value)

    @property
    def center(self):
        """Gets center from model.
        """
        return self.model.get_value(self, "center")

    @center.setter
    def center(self, value):
        """Sets center in model.
        """
        self.model.set_value(self, "center", value)

    @property
    def fit_cps(self):
        """Fetches peak cps from the ModelIface.
        """
        return self.model.get_peak_cps(self, self.region.energy)

    @property
    def fit_cps_fullrange(self):
        """Fetches peak cps over the full length of the spectrum.
        """
        return self.model.get_peak_cps(self, self.region.spectrum.energy_c)

    @property
    def energy(self):
        """Returns energy vector from region.
        """
        return self.region.energy

    @property
    def background(self):
        """Returns background vector from region.
        """
        return self.region.background
