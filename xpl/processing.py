"""Provides functions for data processing."""
# pylint: disable=invalid-name

import logging

import numpy as np
from lmfit import Parameters
from lmfit.models import PseudoVoigtModel


logger = logging.getLogger(__name__)

def calculate_background(bgtype, energy, intensity):
    """Returns background subtracted intensity."""
    # pylint: disable=unsubscriptable-object
    if bgtype == "linear":
        background = np.linspace(intensity[0], intensity[-1], len(energy))
    elif bgtype == "shirley":
        background = shirley(energy, intensity)
    else:
        background = None
    return background


def shirley(energy, intensity, tol=1e-5, maxit=20):
    """Calculates shirley background."""
    if energy[0] < energy[-1]:
        is_reversed = True
        energy = energy[::-1]
        intensity = intensity[::-1]
    else:
        is_reversed = False

    background = np.ones(energy.shape) * intensity[-1]
    integral = np.zeros(energy.shape)
    spacing = (energy[-1] - energy[0]) / (len(energy) - 1)

    subtracted = intensity - background
    ysum = subtracted.sum() - np.cumsum(subtracted)
    for i in range(len(energy)):
        integral[i] = spacing * (ysum[i] - 0.5
                                 * (subtracted[i] + subtracted[-1]))

    iteration = 0
    while iteration < maxit:
        subtracted = intensity - background
        integral = spacing * (subtracted.sum() - np.cumsum(subtracted))
        bnew = ((intensity[0] - intensity[-1])
                * integral / integral[0] + intensity[-1])
        if np.linalg.norm((bnew - background) / intensity[0]) < tol:
            background = bnew.copy()
            break
        else:
            background = bnew.copy()
        iteration += 1
    if iteration >= maxit:
        logger.warning("shirley: Max iterations exceeded before convergence.")

    if is_reversed:
        return background[::-1]
    return background

def smoothen(intensity, interval=20):
    """Smoothed intensity."""
    odd = int(interval / 2) * 2 + 1
    even = int(interval / 2) * 2
    cumsum = np.cumsum(np.insert(intensity, 0, 0))
    avged = (cumsum[odd:] - cumsum[:-odd]) / odd
    for _ in range(int(even / 2)):
        avged = np.insert(avged, 0, avged[0])
        avged = np.insert(avged, -1, avged[-1])
    return avged

def calibrate(energy, shift):
    """Calibrate energy scale."""
    return energy + shift

def x_at_maximum(energy, intensity, span):
    """Calibrate energy axis."""
    emin, emax = span
    idx1, idx2 = sorted([np.searchsorted(energy, emin),
                         np.searchsorted(energy, emax)])
    maxidx = np.argmax(intensity[idx1:idx2]) + idx1
    maxen = energy[maxidx]
    return maxen

def normalize(intensity, norm):
    """Normalize intensity."""
    if not norm:
        return intensity
    if isinstance(norm, (int, float)) and norm != 1:
        normto = norm
    else:
        normto = max(intensity)
    return intensity / normto

# def moving_average():
#     pass
#
# def get_energy_at_maximum():
#     pass

def getspan(energy, intensity, eminmax):
    """Get a slice of (energy_s, intensity_s) from energy interval."""
    idx1, idx2 = sorted([
        np.searchsorted(energy, eminmax[0]),
        np.searchsorted(energy, eminmax[1])])
    return energy[idx1:idx2], intensity[idx1:idx2]


class RegionFitModelIface(object):
    """This manages the Peak models and does the fitting."""
    # pylint: disable=invalid-name
    def __init__(self, region):
        self.single_models = {}
        self.params = Parameters()
        self.region = region

    @property
    def total_model(self):
        """Returns the sum of all models."""
        if not self.single_models:
            return None
        model_list = list(self.single_models.values())
        total = model_list[0]
        for i in range(1, len(model_list)):
            total += model_list[i]
        return total

    def add_peak(self, peak):
        """Adds a new Peak to the Model list."""
        if peak.region is not self.region:
            raise ValueError("Peak does not belong to this Region")

        if peak.model_name == "PseudoVoigt":
            model = PseudoVoigtModel(prefix=peak._prefix)
            model.set_param_hint("sigma", value=2, min=1e-5, max=5)
            model.set_param_hint("amplitude", value=2000, min=0)
            model.set_param_hint("fraction", vary=False)
        self.single_models[peak._prefix] = model

    def remove_peak(self, peak):
        """Removes a Peak from the model and instantiates a new
        CompositeModel."""
        del self.single_models[peak._prefix]
        for parname, param in self.params.items():
            if peak._prefix in parname:
                del param

    def guess_params(self, peak):
        """Guesses parameters for a new peak."""
        model = self.single_models[peak._prefix]
        y = (peak.region.cps
             - peak.region.background
             - self.get_intensity())
        params = model.guess(y, x=self.region.energy)
        self.params += params

    def init_params(self, peak, **kwargs):
        """Sets initial values chosen by user."""
        for parname in ("area", "fwhm", "center"):
            if parname not in kwargs:
                raise TypeError("Missing keyword argument {}".format(parname))

        model = self.single_models[peak._prefix]
        if peak.model_name == "PseudoVoigt":
            sigma = kwargs["fwhm"] / 2
            model.set_param_hint("sigma", value=sigma)
            model.set_param_hint("amplitude", value=kwargs["area"])
            model.set_param_hint("center", value=kwargs["center"])
            params = model.make_params()
            self.params += params

    def fit(self):
        """Returns the fitted intensity values."""
        if not self.single_models:
            return
        y = self.region.cps - self.region.background
        result = self.total_model.fit(y, self.params, x=self.region.energy)
        self.params = result.params

        for peak in self.region.peaks:
            if peak.model_name == "PseudoVoigt":
                amp = self.params["{}amplitude".format(peak._prefix)].value
                sigma = self.params["{}sigma".format(peak._prefix)].value
                center = self.params["{}center".format(peak._prefix)].value
                peak.set("fwhm", sigma * 2)
                peak.set("area", amp)
                peak.set("center", center)

        # print(result.fit_report())

    def get_peak_intensity(self, peak):
        """Returns the model evaluation value for a given Peak."""
        model = self.single_models[peak._prefix]
        return model.eval(params=self.params, x=self.region.energy)

    def get_intensity(self):
        """Returns overall fit result."""
        if not self.total_model:
            return None
        return self.total_model.eval(params=self.params, x=self.region.energy)

    def add_constraint(self, peak, attr, **kwargs):
        """Adds a constraint to a Peak parameter."""
        minval = kwargs.get("min", None)
        if minval:
            minval = float(minval)
        maxval = kwargs.get("max", None)
        if maxval:
            maxval = float(maxval)
        vary = kwargs.get("vary", None)
        expr = kwargs.get("expr", None)
        value = kwargs.get("value", None)
        if peak.model_name == "PseudoVoigt":
            relations = {
                "area": "amplitude",
                "fwhm": "sigma",
                "center": "center"}
            if attr == "fwhm":
                if minval:
                    minval /= 2
                if maxval:
                    maxval /= 2
                if value:
                    value /= 2
                if expr:
                    expr += "/ 2"
        else:
            relations = {}
        self.params["{}{}".format(peak._prefix, relations[attr])].set(
            min=minval, max=maxval, vary=vary, expr=expr, value=value)

    def get_constraint(self, peak, attr, argname):
        """Returns a string containing min/max or expr."""
        if peak.model_name == "PseudoVoigt":
            relations = {
                "area": "amplitude",
                "fwhm": "sigma",
                "center": "center"}
        else:
            relations = {}
        param = self.params["{}{}".format(peak._prefix, relations[attr])]
        minval = param.min
        maxval = param.max
        _vary = param.vary
        expr = param.expr
        if attr == "fwhm":
            minval *= 2
            maxval *= 2
            if expr:
                if expr[-3:] == "/ 2":
                    expr = expr[:-3]
        if argname == "min":
            return minval
        if argname == "max":
            return maxval
        if argname == "expr":
            if expr is None:
                return ""
            return expr
        return None
