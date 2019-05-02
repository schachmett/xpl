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


def parse_spectrum_file(fname):
    """Checks file extension and calls appropriate parsing method."""
    specdicts = []
    if fname.split(".")[-1] == "txt":
        with open(fname, "r") as f:
            firstline = f.readline()
        if "Region" in firstline:
            specdicts.extend(parse_eistxt(fname))
        elif re.fullmatch(r"\d+\.\d+,\d+\n", firstline):
            specdicts.append(parse_simple_xy(fname, delimiter=","))
    elif fname.split(".")[-1] == "xy":
        specdicts.append(parse_simple_xy(fname))
    else:
        logger.warning("file {} not recognized".format(fname))
    return specdicts

def parse_simple_xy(fname, delimiter=None):
    """
    Parses the most simple x, y file with no header.
    """
    energy, intensity = np.genfromtxt(
        fname,
        delimiter=delimiter,
        unpack=True
    )
    specdict = {
        "filename": fname,
        "energy": energy,
        "intensity": intensity,
        "notes": "file {}".format(fname.split("/")[-1])
    }
    return specdict

def parse_eistxt(fname):
    """Splits Omicron EIS txt file."""
    splitregex = re.compile(r"^Region.*")
    skipregex = re.compile(r"^[0-9]*\s*False\s*0\).*")
    spectrumdata = []
    single_spectrumdata = []
    with open(fname, "br") as eisfile:
        for i, line in enumerate(eisfile):
            line = line.decode("utf-8", "backslashreplace")
            if re.match(splitregex, line):
                if single_spectrumdata:
                    spectrumdata.append(single_spectrumdata)
                single_spectrumdata = []
            elif re.match(skipregex, line):
                continue
            elif i == 0:
                raise TypeError("wrong file, not matching EIS format")
            single_spectrumdata.append(line)
        spectrumdata.append(single_spectrumdata)
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
            "filename": fname,
            "energy": energy,
            "intensity": intensity,
            "eis_region": int(header[1][0]),
            "name": "S {}".format(header[1][0]),
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

def merge_project(fname, datahandler):
    """Merges a datahandler object from a binary file."""
    with open(fname, "rb") as file:
        datahandler.merge(pickle.load(file))

def export_txt(dh, spectrumID, fname):
    """Export given spectra and everything that belongs to it as txt."""
    # pylint: disable=too-many-locals
    energy = dh.get(spectrumID, "energy")
    cps = dh.get(spectrumID, "cps")
    background = copy.deepcopy(cps)
    allfit = np.array([0.0] * len(energy))
    peaknames = []
    peaks = []
    for regionID in dh.children(spectrumID):
        emin = np.searchsorted(energy, dh.get(regionID, "emin"))
        emax = np.searchsorted(energy, dh.get(regionID, "emax"))
        single_bg = dh.get(regionID, "background")
        background[emin:emax] -= cps[emin:emax] - single_bg
        allfit[emin:emax] += dh.get(regionID, "fit_cps")
        for peakID in dh.children(regionID):
            peakname = dh.get(peakID, "name")
            peaknames.append("Peak {:19}".format(peakname.replace("Peak", "")))
            peaks.append(dh.get(peakID, "fit_cps_fullrange"))
    data = np.column_stack(
        (energy, cps, background, allfit, *[peak for peak in peaks])
    )
    header = """
        {:22}\t{:24}\t{:24}\t{:24}\t{}
    """.format(
        "Energy",
        "CPS",
        "Background",
        "Fit",
        "{}".format("\t".join(peaknames))
    ).strip()
    np.savetxt(fname, data, delimiter="\t", header=header)


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
