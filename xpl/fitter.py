"""Provides functions for data processing."""
# pylint: disable=invalid-name
# pylint: disable=protected-access
# pylint: disable=logging-format-interpolation

import logging

# import numpy as np
from lmfit import Parameters
from lmfit.models import PseudoVoigtModel


logger = logging.getLogger(__name__)


class RegionFitModelIface(object):
    """This manages the Peak models and does the fitting."""
    # pylint: disable=invalid-name
    def __init__(self, regionID, datahandler):
        self._single_models = {}
        self._params = Parameters()
        self._dh = datahandler
        self._regionID = regionID

    @property
    def total_model(self):
        """Returns the sum of all models."""
        if not self._single_models:
            return None
        model_list = list(self._single_models.values())
        model_sum = model_list[0]
        for i in range(1, len(model_list)):
            model_sum += model_list[i]
        return model_sum

    def add_peak(self, peakID):
        """Adds a new Peak to the Model list."""
        if self._dh.get(peakID, "region") is not self._regionID:    #TODO
            raise ValueError("Peak does not belong to this Region")

        if self._dh.get(peakID, "model_name") == "PseudoVoigt":
            model = PseudoVoigtModel(prefix=self._dh.get(peakID, "prefix"))
            model.set_param_hint("sigma", value=2, min=1e-5, max=5)
            model.set_param_hint("amplitude", value=2000, min=0)
            model.set_param_hint("fraction", vary=False)
        else:
            raise NotImplementedError("Only PseudoVoigt models supported")
        self._single_models[self._dh.get(peakID, "prefix")] = model

    def remove_peak(self, peakID):
        """Removes a Peak from the model and instantiates a new
        CompositeModel."""
        del self._single_models[self._dh.get(peakID, "prefix")]
        for parname, param in self._params.items():
            if self._dh.get(peakID, "prefix") in parname:
                del param

    def guess_params(self, peakID):
        """Guesses parameters for a new peak."""
        model = self._single_models[self._dh.get(peakID, "prefix")]
        y = (self._dh.get(self._regionID, "cps")
             - self._dh.get(self._regionID, "background")
             - self.get_intensity())
        params = model.guess(y, x=self._dh.get(self._regionID, "energy"))
        self._params += params

    def init_params(self, peakID, **kwargs):
        """Sets initial values chosen by user."""
        for parname in ("area", "fwhm", "center"):
            if parname not in kwargs:
                logger.error("Missing parameter {}".format(parname))
                raise TypeError("Missing parameter {}".format(parname))

        model = self._single_models[self._dh.get(peakID, "prefix")]
        if self._dh.get(peakID, "model_name") == "PseudoVoigt":
            sigma = kwargs["fwhm"] / 2
            model.set_param_hint("sigma", value=sigma)
            model.set_param_hint("amplitude", value=kwargs["area"])
            model.set_param_hint("center", value=kwargs["center"])
            params = model.make_params()
            self._params += params
        else:
            raise NotImplementedError("Only PseudoVoigt models supported")

    def fit(self):
        """Returns the fitted intensity values."""
        if not self._single_models:
            return
        y = (
            self._dh.get(self._regionID, "cps")
            - self._dh.get(self._regionID, "background")
        )
        result = self.total_model.fit(
            y,
            self._params,
            x=self._dh.get(self._regionID, "energy")
        )
        self._params = result._params

        for peakID in self._dh.get(self._regionID, "peaks"):
            if self._dh.get(peakID, "model_name") == "PseudoVoigt":
                prefix = self._dh.get(peakID, "prefix")
                amp = self._params["{}amplitude".format(prefix)].value
                sigma = self._params["{}sigma".format(prefix)].value
                center = self._params["{}center".format(prefix)].value
                self._dh.modify_peak(fwhm=sigma * 2, area=amp, center=center)
            else:
                raise NotImplementedError("Only PseudoVoigt models supported")
        # print(result.fit_report())

    def get_peak_intensity(self, peakID):
        """Returns the model evaluation value for a given Peak."""
        model = self._single_models[self._dh.get(peakID, "prefix")]
        results = model.eval(
            params=self._params,
            x=self._dh.get(self._regionID, "energy")
        )
        return results

    def get_intensity(self):
        """Returns overall fit result."""
        if not self.total_model:
            return None
        results = self.total_model.eval(
            params=self._params,
            x=self._dh.get(self._regionID, "energy")
        )
        return results

    # pylint: disable=too-many-arguments
    def add_constraint(self, peakID, attr, min_=None, max_=None, vary=None,
                       expr=None, value=None):
        """Adds a constraint to a Peak parameter."""
        if self._dh.get(peakID, "model_name") == "PseudoVoigt":
            names = {
                "area": "amplitude",
                "fwhm": "sigma",
                "center": "center"}
            if attr == "fwhm":
                if min_:
                    min_ /= 2
                if max_:
                    max_ /= 2
                if value:
                    value /= 2
                if expr:
                    expr += "/ 2"
        else:
            raise NotImplementedError("Only PseudoVoigt models supported")
        paramname = "{}{}".format(self._dh.get(peakID, "prefix"), names[attr])
        self._params[paramname].set(
            min=min_,
            max=max_,
            vary=vary,
            expr=expr,
            value=value
        )

    def get_constraint(self, peakID, attr, argname):
        """Returns a string containing min/max or expr."""
        if self._dh.get(peakID, "model_name") == "PseudoVoigt":
            names = {
                "area": "amplitude",
                "fwhm": "sigma",
                "center": "center"}
        else:
            raise NotImplementedError("Only PseudoVoigt models supported")
        paramname = "{}{}".format(self._dh.get(peakID, "prefix"), names[attr])
        min_ = self._params[paramname].min
        max_ = self._params[paramname].max
        _vary = self._params[paramname].vary
        expr = self._params[paramname].expr
        if attr == "fwhm":
            min_ *= 2
            max_ *= 2
            if expr:
                if expr[-3:] == "/ 2":
                    expr = expr[:-3]
        if argname == "min":
            return min_
        if argname == "max":
            return max_
        if argname == "expr":
            if expr is None:
                return ""
            return expr
        return None
