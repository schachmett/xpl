"""Manages database file and has import filters."""
# pylint: disable=invalid-name
# pylint: disable=logging-format-interpolation

import re
import os
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
        spectra = []
        if fname.split(".")[-1] == "xym":
            parsed = self.parse_xymfile(fname)
            spectra.append(parsed)
        elif fname.split(".")[-1] == "txt":
            xymfiles = self.unpack_eistxt(fname)
            for xymfile in xymfiles:
                parsed = self.parse_xymfile(xymfile)
                spectra.append(parsed)
        elif fname.split(".")[-1] == "xy":
            logger.warning("parsing {} not yet implemented".format(fname))
        else:
            logger.warning("file {} not recognized".format(fname))
        return spectra

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
    def unpack_eistxt(fname):
        """Splits Omicron EIS txt file."""
        xydir = __config__.get("io", "xydir")
        if not os.path.isdir(xydir):
            os.mkdir(xydir)
        splitregex = re.compile(r"^Region.*")
        skipregex = re.compile(r"^[0-9]*\s*False\s*0\).*")
        fnamelist = []
        splitcount = 0
        with open(fname, "r") as eisfile:
            for line in eisfile:
                if re.match(splitregex, line):
                    splitcount += 1
                    wname = (
                        "{0}/{1}-{2}.xym".format(
                            xydir,
                            os.path.basename(fname),
                            str(splitcount).zfill(2)
                        )
                    )
                    xyfile = open(wname, 'w')
                    fnamelist.append(wname)
                    skip = False
                elif re.match(skipregex, line):
                    skip = True
                elif splitcount == 0:
                    raise TypeError("wrong file, not matching EIS format")
                if not skip:
                    xyfile.write(line)
        return fnamelist

def save_project(fname, datahandler):
    """Saves the current datahandler object as a binary file."""
    with open(fname, "wb") as file:
        pickle.dump(datahandler.save(), file, pickle.HIGHEST_PROTOCOL)

def load_project(fname, datahandler):
    """Loads a datahandler object from a binary file."""
    with open(fname, "rb") as file:
        datahandler.load(pickle.load(file))


# class DBHandler():
#     """Handles database access, opening and saving projects
#     (i.e. SpectrumContainers)"""
#     spectrum_keys = ["Name", "Notes", "EISRegion", "Filename", "Sweeps",
#                      "DwellTime", "PassEnergy", "Visibility"]
#
#     def __init__(self, dbfilename="untitled.npl"):
#         self.dbfilename = dbfilename
#
#     def save(self, spectrum_container, fname):
#         """Saves SpectrumContainer to fname."""
#         self.change_dbfile(fname)
#         self.wipe_tables()
#         self.save_container(spectrum_container)
#
#     def load(self, fname):
#         """Loads SpectrumContainer from fname."""
#         self.change_dbfile(fname)
#         spectrum_container = self.get_container()
#         return spectrum_container
#
#     def change_dbfile(self, new_filename):
#         """Change db file name."""
#         self.dbfilename = new_filename
#
#     def remove_dbfile(self):
#         """Trashes db file."""
#         os.remove(self.dbfilename)
#         self.dbfilename = None
#
#     def create_tables(self):
#         """Creates tables if not already created."""
#         create_sql = ["""CREATE TABLE Spectrum
#                          (SpectrumID integer,
#                           Name text,
#                           Notes text,
#                           EISRegion integer,
#                           Filename text,
#                           Sweeps integer,
#                           DwellTime real,
#                           PassEnergy real,
#                           Visibility text,
#                           Energy blob,
#                           Intensity blob,
#                           Regions blob,
#                           PRIMARY KEY (SpectrumID))"""]
#         with sqlite3.connect(self.dbfilename) as database:
#             cursor = database.cursor()
#             for sql in create_sql:
#                 table_name = sql.split()[2]
#                 cursor.execute("SELECT name FROM sqlite_master WHERE name=?",
#                                (table_name, ))
#                 exists = cursor.fetchall()
#                 if not exists:
#                     cursor.execute(sql, ())
#                     database.commit()
#
#     def wipe_tables(self):
#         """Drops tables and creates new ones."""
#         with sqlite3.connect(self.dbfilename) as database:
#             sqls = ["DROP TABLE IF EXISTS Spectrum"]
#             cursor = database.cursor()
#             for sql in sqls:
#                 cursor.execute(sql, ())
#             database.commit()
#         self.create_tables()
#
#     def get_container(self):
#         """Loads project file and returns SpectrumContainer."""
#         with sqlite3.connect(self.dbfilename) as database:
#             cursor = database.cursor()
#             sql = """SELECT SpectrumID, Name, Notes, EISRegion, Filename,
#                      Sweeps, DwellTime, PassEnergy, Visibility, Energy,
#                      Intensity, Regions
#                      FROM Spectrum"""
#             cursor.execute(sql, ())
#             spectrum_container = SpectrumContainer()
#             spectra = cursor.fetchall()
#             for spectrum in spectra:
#                 specdict = {"sid": spectrum[0],
#                             "name": spectrum[1],
#                             "notes": spectrum[2],
#                             "eis_region": spectrum[3],
#                             "fname": spectrum[4],
#                             "sweeps": spectrum[5],
#                             "dwelltime": spectrum[6],
#                             "passenergy": spectrum[7],
#                             "visibility": "",
#                             "energy": pickle.loads(spectrum[9]),
#                             "intensity": pickle.loads(spectrum[10]),
#                             "regions": pickle.loads(spectrum[11])}
#                 spectrum_container.append(Spectrum(**specdict))
#         return spectrum_container
#
#     def save_container(self, spectrum_container):
#         """Dumps SpectrumContainer as project file."""
#         self.wipe_tables()
#         idlist = []
#         with sqlite3.connect(self.dbfilename) as database:
#             cursor = database.cursor()
#             for spectrum in spectrum_container:
#                 idlist.append(self.add_spectrum(spectrum, cursor))
#             database.commit()
#         return idlist
#
#     def add_spectrum(self, spectrum, cursor=None):
#         """Adds a spectrum to the project file."""
#         needs_closing = False
#         if cursor is None:
#             needs_closing = True
#             database = sqlite3.connect(self.dbfilename)
#             cursor = database.cursor()
#         sql = """INSERT INTO Spectrum(Name, Notes, EISRegion, Filename,
#                                       Sweeps, DwellTime, PassEnergy,
#                                       Visibility, Energy, Intensity, Regions)
#                  VALUES(?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)"""
#         values = (spectrum.name,
#                   spectrum.notes,
#                   spectrum.eis_region,
#                   spectrum.fname,
#                   spectrum.sweeps,
#                   spectrum.dwelltime,
#                   spectrum.passenergy,
#                   spectrum.visibility,
#                   pickle.dumps(spectrum.energy),
#                   pickle.dumps(spectrum.intensity),
#                   pickle.dumps(spectrum.regions))
#         cursor.execute(sql, values)
#         spectrum_id = cursor.lastrowid
#         if needs_closing:
#             database.commit()
#             database.close()
#         return spectrum_id
#
#     def remove_spectrum_by_sql_id(self, spectrum_id):
#         """Removes spectrum from the project file."""
#         with sqlite3.connect(self.dbfilename) as database:
#             cursor = database.cursor()
#             sql = "DELETE FROM Spectrum WHERE SpectrumID=?"
#             cursor.execute(sql, (spectrum_id, ))
#             sql = "DELETE FROM SpectrumData WHERE SpectrumID=?"
#             cursor.execute(sql, (spectrum_id, ))
#             database.commit()
#
#     def get_sql_id(self, spectrum):
#         """Searches for a spectrum and gives the sql ID."""
#         with sqlite3.connect(self.dbfilename) as database:
#             cursor = database.cursor()
#             sql = """SELECT SpectrumID FROM Spectrum
#                     WHERE Notes=? AND EISRegion=? AND Filename=? AND Sweeps=?
#                      AND DwellTime=? AND PassEnergy=?"""
#             values = tuple(spectrum[key] for key in self.spectrum_keys[1:-1])
#             cursor.execute(sql, values)
#             ids = cursor.fetchall()
#             if len(ids) < 1:
#                 print("Did not find {0}".format(spectrum))
#                 return None
#             if len(ids) == 1:
#                 return ids[0][0]
#             if len(ids) > 1:
#                 print("Found multiple IDs for spectrum {0}:\n"
#                       "{1}".format(spectrum, ids))
#                 return ids[0][0]


class RSFHandler():
    """Handles rsf library."""
    # pylint: disable=too-few-public-methods
    k_alpha = {"Al": 1486.3,
               "Mg": 1253.4}

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
