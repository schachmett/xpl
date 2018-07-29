"""Manages database file and has import filters."""
# pylint: disable=invalid-name
# pylint: disable=logging-format-interpolation

import re
import pickle
import sqlite3
import logging
import copy

import numpy as np

from xpl import __config__


logger = logging.getLogger(__name__)


class FileParser():
    """Parses arbitrary spectrum files"""
    def parse_spectrum_file(self, fname):
        """Checks file extension and calls appropriate parsing method."""
        specdicts = []
        if fname.split(".")[-1] == "txt":
            with open(fname, "r") as f:
                firstline = f.readline()
            if "Region" in firstline:
                specdicts.extend(self.parse_eistxt(fname))
            elif re.fullmatch(r"\d+\.\d+,\d+\n", firstline):
                specdicts.append(self.parse_simple_xy(fname))
        elif fname.split(".")[-1] == "xy":
            logger.warning("parsing {} not yet implemented".format(fname))
        else:
            logger.warning("file {} not recognized".format(fname))
        return specdicts

    @staticmethod
    def parse_simple_xy(fname):
        """
        Parses the most simple x, y file with no header.
        """
        energy, intensity = np.genfromtxt(
            fname,
            delimiter=",",
            unpack=True
        )
        specdict = {
            "energy": energy,
            "intensity": intensity,
            "notes": "No notes in file"
        }
        return specdict

    @staticmethod
    def parse_eistxt(fname):
        """Splits Omicron EIS txt file."""
        splitregex = re.compile(r"^Region.*")
        skipregex = re.compile(r"^[0-9]*\s*False\s*0\).*")
        spectrumdata = []
        single_spectrumdata = []
        with open(fname, "r") as eisfile:
            for i, line in enumerate(eisfile):
                if re.match(splitregex, line):
                    if single_spectrumdata:
                        spectrumdata.append(single_spectrumdata)
                    single_spectrumdata = []
                elif re.match(skipregex, line):
                    continue
                elif i == 0:
                    raise TypeError("wrong file, not matching EIS format")
                single_spectrumdata.append(line)
        specdicts = []
        for data in spectrumdata:
            energy, intensity = np.genfromtxt(
                data,
                skip_header=5,
                comments="L",
                unpack=True
            )
            header = [line.split("\t") for line in data[:4]]
            specdict = {
                "energy": energy,
                "intensity": intensity,
                "eis_region": int(header[1][0]),
                "sweeps": int(header[1][6]),
                "dwelltime": float(header[1][7]),
                "pass_energy": float(header[1][9]),
                "notes": header[1][12],
            }
            if header[3][0] == "1":
                specdicts.append(specdict)
        return specdicts


def save_project(fname, datahandler):
    """Saves the current datahandler object as a binary file."""
    with open(fname, "wb") as file:
        pickle.dump(datahandler.save(), file, pickle.HIGHEST_PROTOCOL)


def load_project(fname, datahandler):
    """Loads a datahandler object from a binary file."""
    with open(fname, "rb") as file:
        datahandler.load(pickle.load(file))

def export_txt(dh, spectrumID, fname):
    """Export given spectra and everything that belongs to it as txt."""
    energy = dh.get(spectrumID, "energy")
    data = np.insert(data, 0, )
    cps = dh.get(spectrumID, "cps")
    background = copy.deepcopy(cps)
    for regionID in dh.children(spectrumID):
        emin = np.searchsorted(energy, dh.get(regionID, "emin"))
        emax = np.searchsorted(energy, dh.get(regionID, "emax"))
        single_bg = dh.get(regionID, "background")
        background[emin:emax] -= cps[emin:emax] - single_bg
        fit = dh.get(regionID, "fit_cps_fullrange")
        for peakID in dh.children(regionID):
            peak_intensity = dh.get(peakID, "fit_cps_fullrange")
    #TODO obviously


class RSFHandler():
    """Handles rsf library."""
    # pylint: disable=too-few-public-methods
    k_alpha = {
        "Al": 1486.3,
        "Mg": 1253.4
    }

    def __init__(self, filename):
        self.filename = str(filename)

    def get_element(self, element, source):
        """Gets binding energies, rsf and orbital name for specific
        element."""
        with sqlite3.connect(self.filename) as rsfbase:
            cursor = rsfbase.cursor()
            sql = """SELECT Fullname, IsAuger, BE, RSF FROM Peak
                     WHERE Element=? AND (Source=? OR Source=?)"""
            values = (element.title(), source, "Any")
            cursor.execute(sql, values)
            rsf_data = cursor.fetchall()
            rsf_dicts = []
            for dataset in rsf_data:
                if dataset[1] == 1.0:
                    energy = self.k_alpha[source] - dataset[2]
                else:
                    energy = dataset[2]
                rsf_dicts.append({"Fullname": dataset[0],
                                  "BE": energy,
                                  "RSF": dataset[3]})
            return rsf_dicts
