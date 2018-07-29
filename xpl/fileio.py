"""Manages database file and has import filters."""
# pylint: disable=invalid-name
# pylint: disable=logging-format-interpolation

import re
import pickle
import sqlite3
import logging

import numpy as np

from xpl import __config__


logger = logging.getLogger(__name__)


class FileParser():
    """Parses arbitrary spectrum files"""
    def parse_spectrum_file(self, fname):
        """Checks file extension and calls appropriate parsing method."""
        specdicts = []
        if fname.split(".")[-1] == "xym":
            specdicts.append(self.parse_xymfile(fname))
        elif fname.split(".")[-1] == "txt":
            specdicts.extend(self.parse_eistxt(fname))
        elif fname.split(".")[-1] == "xy":
            logger.warning("parsing {} not yet implemented".format(fname))
        else:
            logger.warning("file {} not recognized".format(fname))
        return specdicts

    @staticmethod
    def parse_xymfile(fname):
        """Parses Omicron EIS split txt file."""
        data = dict()
        data["filename"] = fname
        values = np.loadtxt(fname, delimiter="\t", comments="L",
                            skiprows=5, unpack=True)
        data["energy"] = values[0, ::-1]
        data["raw_intensity"] = values[1, ::-1]
        with open(fname, "r") as xyfile:
            header = [line.split("\t") for i, line in enumerate(xyfile)
                      if i in range(0, 4)]
        data["eis_region"] = int(header[1][0])
        data["raw_sweeps"] = int(header[1][6])
        data["raw_dwelltime"] = float(header[1][7])
        data["pass_energy"] = float(header[1][9])
        data["notes"] = header[1][12]
        data["name"] = ""
        data["int_time"] = data["raw_dwelltime"] * data["raw_sweeps"]
        data["cps"] = data["raw_intensity"] / data["int_time"]
        if header[3][0] != "1":
            return None
        return data

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
                "raw_intensity": intensity,
                "eis_region": int(header[1][0]),
                "raw_sweeps": int(header[1][6]),
                "raw_dwelltime": float(header[1][7]),
                "pass_energy": float(header[1][9]),
                "notes": header[1][12],
                "name": "",
                "int_time": float(header[1][7]) * float(header[1][6]),
                "cps": intensity / (float(header[1][7]) * float(header[1][6]))
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
