"""Provides functions for data processing."""
# pylint: disable=invalid-name
# pylint: disable=logging-format-interpolation
# pylint: disable=too-many-arguments

import logging
import re

import numpy as np
from scipy.special import erf
from lmfit import Parameters
from lmfit.models import (
    PseudoVoigtModel, guess_from_peak, update_param_vals#, SkewedVoigtModel
    )
from lmfit.lineshapes import pvoigt, gaussian
from lmfit.model import Model


logger = logging.getLogger(__name__)

s2 = np.sqrt(2)
ln2 = 1 * np.log(2)
sqrtln2 = np.sqrt(ln2)

def realdoniach(x, amplitude=1.0, center=0, sigma=1.0, gamma=0.0):
    """Lineshape for a x-axis reversed doniach, see lmfit."""
    arg = (x-center)/sigma
    arg = -arg          # this is the reversal
    gm1 = (1.0 - gamma)
    scale = amplitude/(sigma**gm1)
    ds = (
        scale
        * np.cos(np.pi * gamma / 2 + gm1 * np.arctan(arg))
        / (1 + arg ** 2) ** (gm1 / 2)
    )
    # g = gaussian(x, amplitude, center, sigma/2)
    # value = np.convolve(ds, g, mode="same")
    # original: scale*cos(pi*gamma/2 + gm1*arctan(arg))/(1 + arg**2)**(gm1/2)
    # return value
    return ds

def convolved_ds(x, amplitude=1.0, center=0, sigma=1.0, conv=0.5, gamma=0.0):
    """convolute DS*GAUSS"""
    ds = realdoniach(x, amplitude, center, sigma, gamma)
    g = gaussian(x, amplitude, center, conv)
    return np.convolve(ds, g, mode="same") / 5000


def skewedPV(x, amplitude=1.0, center=0.0, sigma=1.0, fraction=0.2, skew=0.0):
    """Skewed PseudoVoigt"""
    beta = skew / (s2 * sigma)
    asym = 1 + erf(beta * (x - center))  # NOT MINUS (x-center)
    return asym * pvoigt(x, amplitude, center, sigma, fraction=fraction)


def mygaussian(x, amplitude=1.0, center=0.0, sigma=1.0):
    """G"""
    E = center
    F = sigma
    return np.exp(-4 * ln2 * (x - E) ** 2 / (F ** 2))

def my_GL(x, amplitude=1.0, center=0.0, sigma=1.0, alpha=0.5):
    """GL"""
    E = center
    F = sigma
    m = alpha
    gl = (
        np.exp(-4 * ln2 * 2 * (1 - m) * (x - E) ** 2 / (F ** 2))
        / (1 + 4 * m * (x - E) ** 2 / (F ** 2))
    )
    return gl


def donnyPV(x, amplitude=1.0, center=0.0, sigma=1.0, gamma=0.5): #, gamma2=0.5):
    """skewedPV mocking the realdoniach signature."""
    # return skewedPV(x, amplitude, center, sigma, skew=gamma)
    E = center
    F = sigma
    g = 1 - gamma
    # a = 0.35
    # b = gamma
    val = my_GL(x, 1, E, F, alpha=1)

    val += (1 - val) * np.exp(-g * (x - E) / F) * np.heaviside(x - E, 1)
    # w = b * (0.7 + 0.3 / (a + 0.01))
    # aw = np.exp(
    #     -(
    #         (2 * sqrtln2 * (x - E))
    #         / (F - a * 2 * sqrtln2 * (x - E))
    #     ) ** 2
    # )
    # val += w * (aw - mygaussian(x, amplitude, E, F)) * np.heaviside(x - E, 1)
    return amplitude * val


class MyDoniach(Model):
    """x-axis reversed Doniach model from lmfit."""
    # pylint: disable=dangerous-default-value
    # pylint: disable=arguments-differ
    # pylint: disable=abstract-method
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        # super().__init__(donnyPV, **kwargs)
        super().__init__(realdoniach, **kwargs)
        # super().__init__(convolved_ds, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        fmt = ("{prefix:s}amplitude/({prefix:s}sigma**(1-{prefix:s}gamma))"
               "*cos(pi*{prefix:s}gamma/2)")
        self.set_param_hint('height', expr=fmt.format(prefix=self.prefix))

    def guess(self, data, x=None, negative=False, **kwargs):
        """Guess the pars."""
        pars = guess_from_peak(self, data, x, negative, ampscale=0.5)
        return update_param_vals(pars, self.prefix, **kwargs)


class ConvolvedDoniach(Model):
    """x-axis reversed Doniach model from lmfit."""
    # pylint: disable=dangerous-default-value
    # pylint: disable=arguments-differ
    # pylint: disable=abstract-method
    def __init__(self, independent_vars=['x'], prefix='', nan_policy='raise',
                 **kwargs):
        kwargs.update({'prefix': prefix, 'nan_policy': nan_policy,
                       'independent_vars': independent_vars})
        # super().__init__(donnyPV, **kwargs)
        # super().__init__(realdoniach, **kwargs)
        super().__init__(convolved_ds, **kwargs)
        self._set_paramhints_prefix()

    def _set_paramhints_prefix(self):
        fmt = ("{prefix:s}amplitude/({prefix:s}sigma**(1-{prefix:s}gamma))"
               "*cos(pi*{prefix:s}gamma/2)")
        self.set_param_hint('height', expr=fmt.format(prefix=self.prefix))

    def guess(self, data, x=None, negative=False, **kwargs):
        """Guess the pars."""
        pars = guess_from_peak(self, data, x, negative, ampscale=0.5)
        return update_param_vals(pars, self.prefix, **kwargs)


class RegionFitModelIface(object):
    """This manages the Peak models and does the fitting."""
    # pylint: disable=invalid-name
    def __init__(self, region):
        self._single_models = {}
        self._params = Parameters()
        self._region = region

    @property
    def _total_model(self):
        """Returns the sum of all models."""
        if not self._single_models:
            return None
        model_list = list(self._single_models.values())
        model_sum = model_list[0]
        for i in range(1, len(model_list)):
            model_sum += model_list[i]
        return model_sum

    def add_peak(self, peak):
        """Adds a new Peak to the Model list."""
        if peak.region is not self._region:
            logger.error("peak with ID {} does not belong to region ID {}"
                         "".format(peak.ID, self._region.ID))
            raise ValueError("Peak does not belong to Region")
        if peak.model_name == "PseudoVoigt":
            model = PseudoVoigtModel(prefix=peak.prefix)
            # model.set_param_hint("fraction", vary=False, value=0.2)
            params = model.make_params()
            self._params += params
            fwname = "{}fwhm".format(peak.prefix)
            sigmaname = "{}sigma".format(peak.prefix)
            ampname = "{}amplitude".format(peak.prefix)
            centername = "{}center".format(peak.prefix)
            # alphaname = "{}fraction".format(peak.prefix)
            params[fwname].set(value=params[fwname].value, vary=True, min=0)
            params[sigmaname].set(expr="{}/2".format(fwname))
            params[ampname].set(min=0)
            params[centername].set(min=0)
        elif peak.model_name == "Doniach":
            model = MyDoniach(prefix=peak.prefix)
            params = model.make_params()
            self._params += params
            sigmaname = "{}sigma".format(peak.prefix)
            ampname = "{}amplitude".format(peak.prefix)
            centername = "{}center".format(peak.prefix)
            gammaname = "{}gamma".format(peak.prefix)
            params[sigmaname].set(min=0)
            params[ampname].set(min=0)
            params[centername].set(min=0)
            params[gammaname].set(vary=True)
        elif peak.model_name == "ConvDoniach":
            model = ConvolvedDoniach(prefix=peak.prefix)
            params = model.make_params()
            self._params += params
            sigmaname = "{}sigma".format(peak.prefix)
            ampname = "{}amplitude".format(peak.prefix)
            centername = "{}center".format(peak.prefix)
            gammaname = "{}gamma".format(peak.prefix)
            convname = "{}conv".format(peak.prefix)
            params[sigmaname].set(min=0)
            params[ampname].set(min=0)
            params[centername].set(min=0)
            params[gammaname].set(vary=True)
            params[convname].set(min=0)
        else:
            raise NotImplementedError("Only PseudoVoigt models supported")
        self._single_models[peak.prefix] = model

    def remove_peak(self, peak):
        """Removes a Peak from the model and instantiates a new
        CompositeModel."""
        if peak.prefix not in self._single_models:
            logger.error("peak {} not in model of region {}"
                         "".format(peak.ID, self._region.ID))
            raise AttributeError("Peak not in model")
        self._single_models.pop(peak.prefix)
        pars_to_del = [par for par in self._params if peak.prefix in par]
        for par in pars_to_del:
            self._params.pop(par)

    def guess_params(self, peak):
        """Guesses parameters for a new peak and adds it. See add_peak()."""
        if peak.region is not self._region:
            logger.error("peak with ID {} does not belong to region ID {}"
                         "".format(peak.ID, self._region.ID))
            raise ValueError("Peak does not belong to Region")
        if peak.model_name == "PseudoVoigt":
            model = PseudoVoigtModel(prefix=peak.prefix)
            # model.set_param_hint("fraction", vary=False)
        else:
            raise NotImplementedError("Only PseudoVoigt models supported")
        other_models_cps = [0] * len(self._region.energy)
        for other_peak in self._region.peaks:
            if other_peak == peak:
                continue
            other_models_cps += self.get_peak_cps(other_peak, peak.energy)
        y = self._region.cps - self._region.background - other_models_cps
        params = model.guess(y, x=peak.energy)
        self._params += params
        fwhmname = "{}fwhm".format(peak.prefix)
        sigmaname = "{}sigma".format(peak.prefix)
        ampname = "{}amplitude".format(peak.prefix)
        centername = "{}center".format(peak.prefix)
        params[fwhmname].set(value=params[fwhmname].value, vary=True, min=0)
        params[sigmaname].set(expr="{}/2".format(fwhmname))
        params[ampname].set(min=0)
        params[centername].set(min=0)
        self._single_models[peak.prefix] = model

    def fit(self):
        """Returns the fitted cps values."""
        if not self._single_models:
            return
        y = (self._region.cps - self._region.background)
        result = self._total_model.fit(y, self._params, x=self._region.energy)
        self._params = result.params

    def get_peak_cps(self, peak, energy):
        """Returns the model evaluation value for a given Peak."""
        if peak.prefix not in self._single_models:
            logger.error("peak {} not in model of region {}"
                         "".format(peak.ID, self._region.ID))
            raise AttributeError("Peak not in model")
        model = self._single_models[peak.prefix]
        results = model.eval(params=self._params, x=energy)
        return results

    def get_cps(self, energy):
        """Returns overall fit result."""
        if not self._total_model:
            return [0] * len(energy)
        results = self._total_model.eval(
            params=self._params,
            x=energy
        )
        return results

    def get_value(self, peak, attr):
        """Returns the current value of the paramater corresponding to attr.
        """
        if peak.model_name == "PseudoVoigt":
            names = {
                "area": "amplitude",
                "fwhm": "fwhm",
                "center": "center",
                "alpha": "fraction"
            }
            if attr == "height":
                area = self.get_value(peak, "area")
                fwhm = self.get_value(peak, "fwhm")
                height = (
                    area
                    / (fwhm * np.sqrt(np.pi / np.log(2)))
                    * (1 + np.sqrt(1 / (np.pi * np.log(2))))
                )
                return height
        elif peak.model_name == "Doniach":
            names = {
                "area": "amplitude",
                "sigma": "sigma",
                "center": "center",
                "alpha": "gamma",
                "height": "height"
            }
            if attr == "fwhm":
                return self.get_value(peak, "sigma") * 2
        elif peak.model_name == "Doniach":
            names = {
                "area": "amplitude",
                "sigma": "sigma",
                "center": "center",
                "alpha": "gamma",
                "height": "height"
            }
            if attr == "fwhm":
                return self.get_value(peak, "sigma") * 2
        elif peak.model_name == "ConvDoniach":
            names = {
                "area": "amplitude",
                "sigma": "sigma",
                "center": "center",
                "alpha": "gamma",
                "height": "height",
                "conv": "conv"
            }
            if attr == "fwhm":
                return self.get_value(peak, "sigma") * 2
        else:
            raise NotImplementedError("Only PseudoVoigt models supported")

        paramname = "{}{}".format(peak.prefix, names[attr])
        return self._params[paramname].value

    def set_value(self, peak, attr, value):
        """
        Sets the current value of the parameter corresponding to attr.
        """
        if peak.prefix not in self._single_models:
            self.add_peak(peak)

        if peak.model_name == "PseudoVoigt":
            names = {
                "area": "amplitude",
                "fwhm": "fwhm",
                "center": "center",
                "alpha": "fraction"
            }
            if attr == "height":
                fwhm = self.get_value(peak, "fwhm")
                height = value
                area = (
                    height
                    * (fwhm * np.sqrt(np.pi / np.log(2)))
                    / (1 + np.sqrt(1 / (np.pi * np.log(2))))
                )
                attr = "area"
                value = area
        elif peak.model_name == "Doniach":
            names = {
                "area": "amplitude",
                "fwhm": "sigma",
                "center": "center",
                "alpha": "gamma"
            }
            if attr == "height":
                sigma = self.get_value(peak, "sigma")
                gamma = self.get_value(peak, "alpha")
                attr = "area"
                value = (
                    value
                    * (sigma ** (1 - gamma))
                    / np.cos(np.pi * gamma / 2)
                )
            if attr == "fwhm":
                value /= 2
        elif peak.model_name == "ConvDoniach":
            names = {
                "area": "amplitude",
                "fwhm": "sigma",
                "center": "center",
                "alpha": "gamma",
                "conv": "conv"
            }
            if attr == "height":
                sigma = self.get_value(peak, "sigma")
                gamma = self.get_value(peak, "alpha")
                attr = "area"
                value = (
                    value
                    * (sigma ** (1 - gamma))
                    / np.cos(np.pi * gamma / 2)
                )
            if attr == "fwhm":
                value /= 2
        else:
            raise NotImplementedError("Only PseudoVoigt models supported")
        paramname = "{}{}".format(peak.prefix, names[attr])
        self._params[paramname].set(value=value)

    # pylint: disable=too-many-arguments, too-many-locals
    def set_constraints(self, peak, attr, value, vary, min_, max_, expr):
        """Adds a constraint to a Peak parameter."""
        if peak.model_name == "PseudoVoigt":
            names = {
                "area": "amplitude",
                "fwhm": "fwhm",
                "center": "center",
                "alpha": "fraction"
            }
        elif peak.model_name == "Doniach":
            names = {
                "area": "amplitude",
                "fwhm": "sigma",
                "center": "center",
                "alpha": "gamma"
            }
        elif peak.model_name == "ConvDoniach":
            names = {
                "area": "amplitude",
                "fwhm": "sigma",
                "center": "center",
                "alpha": "gamma",
                "conv": "conv"
            }
        else:
            raise NotImplementedError("Only PseudoVoigt models supported")

        if peak.model_name in ("Doniach", "ConvDoniach"):
            if attr == "fwhm":
                if value:
                    value /= 2
                if min_ and np.isfinite(min_):
                    min_ /= 2
                if max_ and np.isfinite(max_):
                    max_ /= 2
                if expr:
                    expr += "/2"

        if expr:
            def peakrepl(matchobj):
                """Replaces peak.label by peak.prefix_param."""
                label = matchobj.group(0).title()
                other_peak = self._region.get_peak_by_label(label)
                if other_peak is None or other_peak == peak:
                    return "?"
                return "{}{}".format(other_peak.prefix, names[attr])
            expr = re.sub(r"(P|p)\d+", peakrepl, expr)

        paramname = "{}{}".format(peak.prefix, names[attr])
        param = self._params[paramname]

        # pylint: disable=protected-access
        evaluator = param._expr_eval
        # pylint: enable=protected-access
        evaluator.error.clear()
        isvalid = True
        try:
            evaluator.parse(expr)
            self._params.valuesdict()
            # evaluator(ast_expr, show_errors=False)
        except (SyntaxError, NameError):
            expr = ""
            isvalid = False
            logger.debug("invalid expression '{}'".format(expr))

        param.set(min=min_, max=max_, vary=vary, expr=expr, value=value)
        logger.debug("invalid expression '{}'".format(expr))
        return isvalid

    def get_constraints(self, peak, attr):
        """Returns a string containing min/max or expr."""
        if peak.model_name == "PseudoVoigt":
            names = {
                "area": "amplitude",
                "fwhm": "fwhm",
                "center": "center",
                "alpha": "fraction"
            }
        elif peak.model_name == "Doniach":
            names = {
                "area": "amplitude",
                "fwhm": "sigma",
                "center": "center",
                "alpha": "gamma"
            }
        elif peak.model_name == "ConvDoniach":
            names = {
                "area": "amplitude",
                "fwhm": "sigma",
                "center": "center",
                "alpha": "gamma",
                "conv": "conv"
            }
        else:
            raise NotImplementedError("Only PseudoVoigt models supported")

        constraints = {}
        paramname = "{}{}".format(peak.prefix, names[attr])
        constraints["min_"] = self._params[paramname].min
        constraints["max_"] = self._params[paramname].max
        constraints["vary"] = self._params[paramname].vary
        constraints["expr"] = self._params[paramname].expr

        if constraints["expr"]:
            def reverse_peakrepl(matchobj):
                """Replaces peak.label by peak.prefix_param."""
                prefix = matchobj.group(0).split("_")[0] + "_"
                other_peak = self._region.get_peak_by_prefix(prefix)
                if other_peak is None or other_peak == peak:
                    return ""
                return other_peak.label
            constraints["expr"] = re.sub(
                r"p\d+_[a-zA-Z_]+",
                reverse_peakrepl,
                constraints["expr"]
            )

        if peak.model_name in ("Doniach", "ConvDoniach"):
            if attr == "fwhm":
                if np.isfinite(constraints["min_"]):
                    constraints["min_"] *= 2
                if np.isfinite(constraints["min_"]):
                    constraints["max_"] *= 2
                expr = constraints["expr"]
                if expr and "/2" in expr:
                    constraints["expr"] = expr.replace("/2", "", 1)

        return constraints
