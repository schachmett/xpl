"""Provides functions for data processing."""
# pylint: disable=invalid-name
# pylint: disable=logging-format-interpolation

import logging
import re

import numpy as np
from lmfit import Parameters
from lmfit.models import PseudoVoigtModel


logger = logging.getLogger(__name__)


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
            model.set_param_hint("fraction", vary=False)
            params = model.make_params()
        else:
            raise NotImplementedError("Only PseudoVoigt models supported")
        self._params += params
        fwhmname = "{}fwhm".format(peak.prefix)
        sigmaname = "{}sigma".format(peak.prefix)
        params[fwhmname].set(vary=True)
        params[sigmaname].set(expr="{}/2".format(fwhmname))
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
        """Guesses parameters for a new peak."""
        if peak.prefix not in self._single_models:
            self.add_peak(peak)
        model = self._single_models[peak.prefix]
        other_models_cps = [0] * len(self._region.energy)
        for other_peak in self._region.peaks:
            if other_peak == peak:
                continue
            other_models_cps += self.get_peak_cps(other_peak)
        y = self._region.cps - self._region.background - other_models_cps
        params = model.guess(y, x=self._region.energy)
        self._params += params

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
                "center": "center"
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
                "center": "center"
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
                "center": "center"}
        else:
            raise NotImplementedError("Only PseudoVoigt models supported")

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
            ast_expr = evaluator.parse(expr)
            evaluator(ast_expr, show_errors=False)
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
                "center": "center"}
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
        return constraints
