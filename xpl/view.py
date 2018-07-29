"""This module provides a class with GUI-friendly methods for controling
the DataHandler from xpl.datahandler."""
# pylint: disable=invalid-name
# pylint: disable=wrong-import-position
# pylint: disable=too-many-instance-attributes
# pylint: disable=logging-format-interpolation

from collections import OrderedDict
import re
import logging

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Pango
import numpy as np

from xpl.fileio import RSFHandler
from xpl.canvas_selection import DraggableAttributeLine
from xpl import __colors__, RSF_DB_PATH


logger = logging.getLogger(__name__)

SPECTRUM_TITLES = OrderedDict({
    "ID": "SpectrumID",
    "name": "Name",
    "notes": "Notes",
    "eis_region": "EIS Region",
    "filename": "File name",
    "sweeps": "Sweeps",
    "dwelltime": "Dwell Time [s]",
    "pass_energy": "Pass Energy [eV]",
    "int_time": "Total Integration Time [s]"
})
TV_TITLES = OrderedDict(
    (attr, SPECTRUM_TITLES[attr]) for attr in (
        "name",
        "notes"
    )
)
DEFAULT_COLNAME = "Notes"
EDIT_TITLES = OrderedDict(
    (attr, SPECTRUM_TITLES[attr]) for attr in (
        "name",
        "notes",
        "sweeps",
        "dwelltime",
        "pass_energy"
    )
)
EXCLUDING_KEY = " (multiple)"

PEAK_TITLES = OrderedDict({
    "ID": "PeakID",
    "label": "Label",
    "name": "Name",
    "model_name": "Model",
    "fwhm": "FWHM",
    "area": "Area",
    "center": "Position"
})
PEAK_TV_TITLES = OrderedDict(
    (attr, PEAK_TITLES[attr]) for attr in (
        "label",
        "name",
        "center",
        "area",
        "fwhm",
    )
)


class ActiveIDs():
    """Stores the active spectra, regions and peaks."""
    # pylint: disable=too-few-public-methods
    SPECTRA = []
    REGION = None
    PEAK = None


class XPLView():
    """This class provides methods for manipulating the view, accessing the
    DataHandler only to retrieve data, not setting them."""
    def __init__(self, builder, datahandler):
        self._builder = builder
        self._dh = datahandler

        # here, the order of instantiation is important: cviface has
        # to register its callbacks last so _cviface.plot() is always
        # called at the end of signal handling from datahandler
        self._tviface = XPLTreeViewInterface(builder, datahandler)
        self._fitiface = XPLFitInterface(builder, datahandler)
        self._cviface = XPLCanvasInterface(builder, datahandler)
        self.set_visible = self._cviface.set_visible
        self.get_visible = self._cviface.get_visible

        self.get_selected_spectra = self._tviface.get_selected_spectra
        self.get_active_spectra = lambda *args: ActiveIDs.SPECTRA

        self.get_active_region = lambda *args: ActiveIDs.REGION

        self.get_selected_peak = self._fitiface.get_selected_peak
        self.get_active_peak = lambda *args: ActiveIDs.PEAK

        self.filter_spectra = self._tviface.filter_treeview
        self.show_rsf = self._cviface.show_rsf

    def activate_spectra(self, spectrumIDs):
        """Plot only spectra with ID in spectrumIDs."""
        if not spectrumIDs:
            ActiveIDs.SPECTRA = []
        else:
            for spectrumID in spectrumIDs:
                assert self._dh.isspectrum(spectrumID)
            self._cviface.store_xylims()
            ActiveIDs.SPECTRA = spectrumIDs
            self._tviface.activate_spectra(spectrumIDs)
        self._fitiface.activate_spectra(spectrumIDs)
        self._cviface.plot_only(spectrumIDs)
        self._revert_adjustment_widgets()

    def activate_region(self, regionID):
        """Activate region."""
        self._fitiface.activate_region(regionID)
        # ActiveIDs.REGION = regionID
        self._cviface.plot()

    def activate_peak(self, peakID):
        """Activate peak."""
        self._fitiface.activate_peak(peakID)
        # ActiveIDs.PEAK = peakID
        self._cviface.plot()

    def set_region_boundary_setter(self, setter):
        """Sets a callback to be called when a DraggableAttributeLine
        belonging to a region is moved."""
        self._cviface.dragline_callback = setter

    def pop_spectrum_menu(self, event):
        """Pops up the spectrum menu and returns True if the event was
        in an already selected treeview row."""
        self._tviface.pop_spectrum_context_menu(event)

    def get_edit_spectra_dialog(self, spectrumIDs):
        """Customizes the dialog for editing spectra."""
        for spectrumID in spectrumIDs:
            assert self._dh.isspectrum(spectrumID)
        dialog = self._builder.get_object("edit_spectrum_dialog")
        dialog.flush()
        if len(spectrumIDs) == 1:
            dialog.populate_filename(self._dh.get(spectrumIDs[0], "filename"))
            for attr, title in EDIT_TITLES.items():
                value = str(self._dh.get(spectrumIDs[0], attr))
                dialog.add_attribute(attr, title, value)
        else:
            fnames = ""
            for spectrumID in spectrumIDs:
                fnames += self._dh.get(spectrumID, "filename") + "\n"
            fnames = fnames.strip()
            dialog.populate_filename(fnames)
            for attr, title in EDIT_TITLES.items():
                valueset = set([
                    str(self._dh.get(spectrumID, attr))
                    for spectrumID in spectrumIDs
                ])
                value = " | ".join(valueset) + EXCLUDING_KEY
                dialog.add_attribute(attr, title, value)
        return dialog

    def _revert_adjustment_widgets(self):
        """If something different is selected, all adjustment widgets that
        affect spectra have to be reset."""
        cautions = (
            self._builder.get_object("adj_caution_image1"),
            self._builder.get_object("adj_caution_image2"),
            self._builder.get_object("adj_caution_image3")
        )
        smooth = self._builder.get_object("smoothing_scale_adjustment")
        cal = self._builder.get_object("calibration_spinbutton_adjustment")
        norm = self._builder.get_object("normalization_switch")

        spectrumIDs = self._cviface.get_active_spectra()
        if len(spectrumIDs) != 1:
            for caution in cautions:
                caution.set_visible(True)
            # smooth.set_value(self._dh.get(IDs[0], "smoothness"))
            # cal.set_value(self._dh.get(IDs[0], "calibration"))
        else:
            for caution in cautions:
                caution.set_visible(False)
            smooth.set_value(self._dh.get(spectrumIDs[0], "smoothness"))
            cal.set_value(self._dh.get(spectrumIDs[0], "calibration"))
            norm.set_active(self._dh.get(spectrumIDs[0], "norm"))


class XPLTreeViewInterface():
    """This class provides methods for manipulating the view, accessing the
    DataHandler only to retrieve data."""
    def __init__(self, builder, datahandler):
        self._dh = datahandler

        # register callbacks into the data handler to reflect changes in
        # the meta data values or added/deleted spectra
        handlers = {
            "added-spectrum": self._on_spectrum_added,
            "removed-spectrum": self._on_spectrum_removed,
            "changed-spectrum": self._on_spectrum_changed,
            "cleared-spectra": self._on_spectra_cleared
        }
        for signal, handler in handlers.items():
            self._dh.connect(signal, handler)

        # get all the gui elements defined in the xml file
        # self._treeview is special as it is a xpl.gui.XPLSpectrumTreeStore
        self._treeview = builder.get_object("spectrum_view")
        self._treemodel = builder.get_object("spectrum_treestore")
        self._treemodelfilter = builder.get_object("spectrum_filter_treestore")
        self._filtercombo = builder.get_object("spectrum_view_search_combo")
        self._treemodelsort = builder.get_object("spectrum_sort_treestore")
        self._selection = builder.get_object("spectrum_selection")
        self._tvmenu = builder.get_object("spectrum_view_context_menu")
        self._tvmenu.attach_to_widget(self._treeview, None)

        self._make_columns()
        # tv_filter = (attr, regex) for filtering the column belonging to
        # attr with the given regex
        self.tv_filter = (None, None)
        self._setup_filter()

    def get_selected_spectra(self):
        """Returns list of currently selected Spectrum IDs."""
        treemodelsort, pathlist = self._selection.get_selected_rows()
        iters = [treemodelsort.get_iter(path) for path in pathlist]
        spectrumIDs = [int(treemodelsort.get(iter_, 0)[0]) for iter_ in iters]
        return spectrumIDs

    def activate_spectra(self, spectrumIDs):
        """Mark only the spectra with ID in spectrumIDs as active."""
        for spectrumID in spectrumIDs:
            assert self._dh.isspectrum(spectrumID)
        col_index_active = self._treemodel.get_col_index("active")
        for row in self._treemodel:
            if int(row[0]) in spectrumIDs:
                self._treemodel.set(row.iter, col_index_active, True)
            else:
                self._treemodel.set(row.iter, col_index_active, False)
        logger.debug("spectrumview: marked spectra {}".format(spectrumIDs))

    def pop_spectrum_context_menu(self, event):
        """Makes the context menu for spectra pop up."""
        self._tvmenu.popup(None, None, None, None, event.button, event.time)

    def filter_treeview(self, attr_or_title, search_term):
        """Filters the TreeView, only showing rows which attribute matches
        search_term, which should be a string containing a regular
        expression."""
        if attr_or_title not in SPECTRUM_TITLES.keys():
            for attr, title in SPECTRUM_TITLES.items():
                if title == attr_or_title:
                    search_attr = attr
                    break
            else:
                search_attr = None
        else:
            search_attr = attr
        self.tv_filter = (search_attr, search_term)
        self._treemodelfilter.refilter()
        logger.debug("spectrumview: searched attr '{}' with regex '{}'"
                     "".format(search_attr, search_term))

    def _make_columns(self):
        """Initializes columns. Must therefore be called in __init__."""
        # this column is True when the row is activated
        col_index_active = self._treemodel.get_col_index("active")
        # render function for making plotted spectra bold and light blue
        bgcolor = self._treeview.style_get_property("even-row-color")
        def render_isplotted(_col, renderer, model, iter_, *_data):
            """Renders the cell light blue if this spectrum is plotted."""
            if model.get_value(iter_, col_index_active):
                renderer.set_property(
                    "cell-background",
                    __colors__.get("treeview", "tv-highlight-bg")
                )
                renderer.set_property("weight", Pango.Weight.BOLD)
            else:
                renderer.set_property("cell-background", bgcolor)
                renderer.set_property("weight", Pango.Weight.NORMAL)
        # the other columns are simple, just apply the render_isplotted func
        for attr, title in TV_TITLES.items():
            renderer = Gtk.CellRendererText(xalign=0)
            col_index = self._treemodel.get_col_index(attr)
            column = Gtk.TreeViewColumn(title, renderer, text=col_index)
            column.set_cell_data_func(renderer, render_isplotted)
            column.set_sort_column_id(col_index)
            column.set_resizable(True)
            column.set_reorderable(True)
            self._treeview.append_column(column)

    def _setup_filter(self):
        """Initializes the TreeModelFilter and its filtering function.
        Must therefore be called in __init__."""
        # filling the combobox that determines self.tv_filter[0]
        for i, title in enumerate(TV_TITLES.values()):
            self._filtercombo.append_text(title)
            if title == DEFAULT_COLNAME:
                self._filtercombo.set_active(i)
        # this function looks into self.tv_filter and executes the regex
        # matching, returning True if the row should be visible
        def filter_func(treemodel, iter_, *_data):
            """Returns True only for rows whose values for the attr
            from self.tv_filter matches the regex from self.tv_filter."""
            attr, search_term = self.tv_filter
            regex = re.compile(r".*{}.*".format(search_term), re.IGNORECASE)
            if not attr or not search_term:
                return True
            col_index = treemodel.get_col_index(attr)
            return re.match(regex, treemodel.get(iter_, col_index)[0])
        self._treemodelfilter.set_visible_func(filter_func)

    def _on_spectrum_added(self, spectrumID):
        """Adds spectrum to the the TreeView. It must already be present
        in the DataHandler."""
        specdict = dict(
            (attr, self._dh.get(spectrumID, attr))
            for attr in self._treemodel.titles
        )
        specdict["active"] = False
        self._treemodel.append(specdict)
        logger.debug("spectrumview: added spectrum {}".format(spectrumID))

    def _on_spectrum_removed(self, spectrumID):
        """Removes spectrum from the TreeView. It does not matter if it is
        still in the DataHandler."""
        for row in self._treemodel:
            if int(row[0]) == spectrumID:
                self._treemodel.remove(row.iter)
        logger.debug("spectrumview: removed spectrum {}".format(spectrumID))

    def _on_spectrum_changed(self, spectrumID, attr):
        """Changes spectrum data by applying newdict which contains all
        (attr: newvalue) pairs that should be updated."""
        if attr not in TV_TITLES:
            return
        col_index = self._treemodel.get_col_index(attr)
        value = self._dh.get(spectrumID, attr)
        for row in self._treemodel:
            if int(row[0]) == spectrumID:
                self._treemodel.set(row.iter, col_index, value)
        logger.debug("spectrumview: updated spectrum {}, attr '{}' is now '{}'"
                     "".format(spectrumID, attr, value))

    def _on_spectra_cleared(self, _none):
        """Removes all spectra from the model."""
        self._treemodel.clear()
        logger.debug("spectrumview: cleared all spectra")


class XPLCanvasInterface():
    """Methods for showing stuff on the canvas."""
    def __init__(self, builder, datahandler):
        self._dh = datahandler

        # register callbacks into the data handler to reflect changes in
        # the meta data values or added/deleted spectra
        handlers = {
            "removed-spectrum": self._on_dh_data_changed,
            "cleared-spectra": self._on_dh_data_changed,
            "changed-spectrum": self._on_dh_data_changed,
            "added-region": self._on_dh_data_changed,
            "removed-region": self._on_dh_data_changed,
            "cleared-regions": self._on_dh_data_changed,
            "changed-region": self._on_dh_data_changed,
            "fit-region": self._on_dh_data_changed,
            "added-peak": self._on_dh_data_changed,
            "removed-peak": self._on_dh_data_changed,
            "changed-peak": self._on_dh_data_changed,
            "cleared-peaks": self._on_dh_data_changed,
        }
        for signal, handler in handlers.items():
            self._dh.connect(signal, handler)

        # get all the gui elements defined in the xml file
        # _canvas, _fig and _navbar special as they are defined in xpl.gui
        self._canvas = builder.get_object("main_canvas")
        self._fig = self._canvas.figure
        self._ax = self._fig.ax
        self._navbar = builder.get_object("plot_toolbar")

        # get the rsfhandler needed to fetch rsf data that can be plotted
        self.rsfhandler = RSFHandler(RSF_DB_PATH)
        # set up which rsf data should be plotted
        # by rsf_filter = (elements, source)
        self.rsf_filter = ([], "")

        # dragline_callback is evoked when a DraggableVLine is moved
        # all of those that are avtive are stored in self._draglines
        self.dragline_callback = lambda ID, **kwargs: None
        self._draglines = []

        # contains all IDs for objects that should be plotted
        self._plotIDs = []

        # which elements should be plotted?
        self._doplot = {
            "spectrum": True,
            "region-boundaries": True,
            "region-background": True,
            "region-fit": True,
            "peak": True,
            "rsfs": True
        }

    def get_active_spectra(self):
        """Returns which spectra are on the canvas right now."""
        spectrumIDs = [ID for ID in self._plotIDs if self._dh.isspectrum(ID)]
        return spectrumIDs

    def set_visible(self, keyword, yesno):
        """Sets plot things that belong to keyword to visible/invisible."""
        if keyword not in self._doplot:
            raise TypeError("canvas: keyword '{}' unknown".format(keyword))
        self._doplot[keyword] = bool(yesno)
        self.plot()

    def get_visible(self, keyword):
        """Returns if things that belong to keyword are visible."""
        if keyword not in self._doplot:
            raise TypeError("canvas: keyword '{}' unknown".format(keyword))
        return self._doplot[keyword]

    def plot_only(self, IDs):
        """Plot only the given IDs."""
        self._plotIDs.clear()
        self._plotIDs.extend(IDs)
        self.plot()

    def show_rsf(self, elements, source):
        """Add elements to the rsfs to be shown."""
        self.rsf_filter = (elements, source)
        self.plot()

    def _update_active(self):
        """Update self._plotIDs with child and grandchild IDs."""
        new_IDs = []
        for ID in self._plotIDs:
            if not self._dh.exists(ID):
                continue
            new_IDs.append(ID)
            for childID in self._dh.children(ID):
                new_IDs.append(childID)
                for grandchildID in self._dh.children(childID):
                    new_IDs.append(grandchildID)
        self._plotIDs = list(set(new_IDs))

    def store_xylims(self):
        """Stores the current xylims for the current view."""
        xy_keyword = str(ActiveIDs.SPECTRA)
        self._fig.store_xylims(keyword=xy_keyword)

    def plot(self, keepaxes=True):
        """Plots every ID in self._plotIDs after checking for child and
        grandchild IDs. Also plots RSF data"""
        self._update_active()
        # check if there actually is something to plot
        if not self._plotIDs:
            self._draglines.clear()
            self._ax.cla()
            self._fig.restore_xylims()
            self._canvas.draw_idle()
            self._navbar.disable_tools()
            return

        xy_keyword = str(ActiveIDs.SPECTRA)
        if not self._fig.isstored(xy_keyword):
            keepaxes = False

        self._draglines.clear()
        self._fig.reset_xy_centerlims()
        self._ax.cla()

        # call the individual plot jobs
        for ID in self._plotIDs:
            if self._dh.isspectrum(ID):
                self._plot_spectrum(ID)
            elif self._dh.isregion(ID):
                self._plot_region(ID)
            elif self._dh.ispeak(ID):
                self._plot_peak(ID)
            else:
                raise TypeError("ID {} can not be plotted".format(ID))
        if self.rsf_filter[0]:
            self._plot_rsfs(self.rsf_filter[0], self.rsf_filter[1])
            logger.debug("canvas: plot rsfs {}".format(self.rsf_filter))

        # restore xylims if necessary and then redraw
        if keepaxes:
            self._fig.restore_xylims(keyword=xy_keyword)
        else:
            self._fig.center_view(keyword=xy_keyword)
        self._canvas.draw_idle()
        self._navbar.disable_tools()

        logger.debug("canvas: plot IDs {}".format(self._plotIDs))

    def _plot_spectrum(self, spectrumID):
        """Plots a spectrum and updates the xylims for a centered view."""
        energy = self._dh.get(spectrumID, "energy")
        cps = self._dh.get(spectrumID, "cps")
        if not self._doplot["region-background"]:
            new_cps = [0] * len(cps)
            for regionID in self._dh.children(spectrumID):
                emin = np.searchsorted(energy, self._dh.get(regionID, "emin"))
                emax = np.searchsorted(energy, self._dh.get(regionID, "emax"))
                background = self._dh.get(regionID, "background")
                new_cps[emin:emax] += cps[emin:emax] - background
            if any(new_cps):
                cps = new_cps

        if self._doplot["spectrum"]:
            lineprops = {
                "color": __colors__.get("plotting", "spectrum"),
                "linewidth": 1,
                "linestyle": "-",
                "alpha": 1
            }
            self._ax.plot(energy, cps, **lineprops)

        self._fig.update_xy_centerlims(
            min(energy),
            max(energy),
            min(cps),
            max(cps)
        )

    def _plot_region(self, regionID):
        """Plots a region by plotting its limits with DraggableVLines and
        plotting the background intensity."""
        emin = self._dh.get(regionID, "emin")
        emax = self._dh.get(regionID, "emax")
        energy = self._dh.get(regionID, "energy")
        background = self._dh.get(regionID, "background")
        if not self._doplot["region-background"]:
            background = [0] * len(background)
        fit_cps = self._dh.get(regionID, "fit_cps")

        if self._doplot["region-boundaries"]:
            if regionID == ActiveIDs.REGION:
                color = __colors__.get("plotting", "region-vlines-active")
            else:
                color = __colors__.get("plotting", "region-vlines")
            lineprops = {
                "color": color,
                "linewidth": 2,
                "linestyle": "--",
                "alpha": 1
            }
            line = self._ax.axvline(emin, 0, 1, **lineprops)
            self._draglines.append(DraggableAttributeLine(
                line, regionID, "emin", self.dragline_callback))
            line = self._ax.axvline(emax, 0, 1, **lineprops)
            self._draglines.append(DraggableAttributeLine(
                line, regionID, "emax", self.dragline_callback))

        if self._doplot["region-background"] and any(background):
            if regionID == ActiveIDs.REGION:
                color = __colors__.get("plotting", "region-background-active")
            else:
                color = __colors__.get("plotting", "region-background")
            lineprops = {
                "color": color,
                "linewidth": 1,
                "linestyle": "--"
            }
            self._ax.plot(energy, background, **lineprops)

        if self._doplot["region-fit"] and any(fit_cps):
            if regionID == ActiveIDs.REGION:
                color = __colors__.get("plotting", "peak-sum-active")
            else:
                color = __colors__.get("plotting", "peak-sum")
            lineprops = {
                "color": color,
                "linewidth": 1,
                "linestyle": "--"
            }
            self._ax.plot(energy, background + fit_cps, **lineprops)


    def _plot_peak(self, peakID):
        """Plots a peak."""
        energy = self._dh.get(peakID, "energy")
        background = self._dh.get(peakID, "background")
        if not self._doplot["region-background"]:
            background = [0] * len(background)
        fit_cps = self._dh.get(peakID, "fit_cps")

        if self._doplot["peak"]:
            if peakID == ActiveIDs.PEAK:
                color = __colors__.get("plotting", "peak-active")
                lineprops = {
                    "color": color,
                    "linewidth": 1,
                    "linestyle": "--",
                    "alpha": 0.2
                }
                self._ax.fill_between(
                    energy,
                    background + fit_cps,
                    background,
                    **lineprops
                )
                self._ax.plot(energy, background + fit_cps, **lineprops)
            else:
                color = __colors__.get("plotting", "peak")
                lineprops = {
                    "color": color,
                    "linewidth": 1,
                    "linestyle": "--"
                }
                self._ax.plot(energy, background + fit_cps, **lineprops)


    def _plot_rsfs(self, elements, source):
        """Plots RSF values for given elements with given X-ray souce."""
        # fetch data through the rsfhandler
        if not self._doplot["rsfs"]:
            return

        rsfs = []
        for element in elements:
            dicts = self.rsfhandler.get_element(element, source)
            if dicts:
                rsfs.append(dicts)
            else:
                logger.warning("canvas: element {} not found".format(element))

        # set up colors and maximum rsf intensity for scaling the vlines
        if not rsfs:
            return
        max_rsf = max(max([[d["RSF"] + 1e-9 for d in ds] for ds in rsfs]))
        colorstr = __colors__.get("plotting", "rsf-vlines").replace(" ", "")
        colors = colorstr.split(",") * 10

        # plot the individual vlines
        for i, peaks in enumerate(rsfs):
            for peak in peaks:
                if peak["RSF"] == 0:
                    rsf = 0.5 * self._fig.get_ymax()
                else:
                    rsf = peak["RSF"] * self._fig.get_ymax() / max_rsf * 0.8
                self._ax.vlines(peak["BE"], 0, rsf, colors=colors[i], lw=2)
                self._ax.annotate(
                    peak["Fullname"],
                    xy=(peak["BE"], rsf),
                    color=__colors__.get("plotting", "rsf-annotation"),
                    textcoords="data",
                    ha="center",
                    va="bottom"
                )

    def _on_dh_data_changed(self, ID=None, attr=None, isvalid=True):
        """Replots if the ID is affected. Depending on changed attr, the
        axislims are kept or not. The ID is affected either if it is
        in self._plotIDs or its parend is. Also, if no ID is given,
        we should replot."""
        if (ID is not None
                and ID not in self._plotIDs
                and self._dh.exists(ID)
                and self._dh.exists(self._dh.parent(ID))
                and self._dh.parent(ID) not in self._plotIDs):
            return
        trigger_attrs = (
            None,
            "norm", "smoothness", "calibration",
            "int_time", "energy", "cps",
            "bgtype", "emin", "emax",
            "height", "area", "fwhm", "center"
        )
        if attr not in trigger_attrs:
            return
        keepaxes = not attr in ("norm",)
        self.store_xylims()
        logger.debug("replotting because datahandler changed...")
        self.plot(keepaxes)


class XPLFitInterface():
    """Manages the second child of the main spectra_vs_region_stack that
    looks after the region and peak data."""
    def __init__(self, builder, datahandler):
        self._dh = datahandler
        self._builder = builder

        # register callbacks into the data handler to reflect changes in
        # the meta data values or added/deleted spectra
        handlers = {
            "added-region": self._on_region_added,
            "removed-region": self._on_region_removed,
            "cleared-regions": self._on_regions_cleared,
            "added-peak": self._on_peak_added,
            "changed-peak": self._on_peak_changed,
            "removed-peak": self._on_peak_removed,
            "cleared-peaks": self._on_peaks_cleared,
            "fit-region": self._on_fit_region,
        }
        for signal, handler in handlers.items():
            self._dh.connect(signal, handler)

        # get all the gui elements defined in the xml file
        self._region_chooser = self._builder.get_object("region_chooser_combo")
        self._peak_parambox = self._builder.get_object("peak_param_box")
        self._peak_treeview = self._builder.get_object("peak_view")
        self._peak_selection = self._builder.get_object("peak_selection")
        self._peak_model = self._builder.get_object("peak_treestore")

        self._make_peakview_columns()

    def activate_spectra(self, spectrumIDs):
        """Reacts upon changing the active spectra by refilling the
        region_chooser and updating the widgets. Always
        either calls activate_region() or _update_widgets()."""
        logger.debug("regionview: activate spectra {}".format(spectrumIDs))
        self._region_chooser.remove_all()
        if len(spectrumIDs) == 1:
            regionIDs = self._dh.children(spectrumIDs[0])
            for regionID in regionIDs:
                region_name = self._dh.get(regionID, "name")
                self._region_chooser.append(str(regionID), region_name)
            if regionIDs:
                self.activate_region(regionIDs[0])
            else:
                self.activate_region(None)
        else:
            self.activate_region(None)

    def activate_region(self, regionID):
        """Sets the active region. Always either calls activate_peak() or
        _update_widgets()."""
        if regionID == ActiveIDs.REGION:
            self._update_widgets()
            return
        ActiveIDs.REGION = regionID

        rchooserID = self._region_chooser.get_active_id()
        if str(regionID) != rchooserID:
            self._region_chooser.set_active_id(str(regionID))
        logger.debug("regionview: activated region {}".format(regionID))

        if self._peak_model.region != regionID:
            self._on_peaks_cleared(regionID)
            self._peak_model.region = regionID
            if regionID:
                peakIDs = self._dh.children(regionID)
                for peakID in peakIDs:
                    self._on_peak_added(peakID)
                if peakIDs:
                    self.activate_peak(peakIDs[0])
                else:
                    self.activate_peak(None)
            else:
                self._update_widgets()
        else:
            self._update_widgets()

    def activate_peak(self, peakID):
        """Sets the active peak. If peakID is None, the currently selected
        peak in the peakview is used."""
        if peakID == ActiveIDs.PEAK:
            self._update_widgets()
            return
        ActiveIDs.PEAK = peakID

        self._peak_selection.unselect_all()
        model = self._peak_selection.get_tree_view().get_model()
        for row in model:
            if int(row[0]) == peakID:
                self._peak_selection.select_iter(row.iter)
                break
        logger.debug("peakview: activated peak {}".format(peakID))
        self._update_widgets()

    def get_selected_peak(self):
        """Returns the peak that is currently selected or None."""
        peak_modelsort, pathlist = self._peak_selection.get_selected_rows()
        if not pathlist:
            return None
        iter_ = peak_modelsort.get_iter(pathlist[0])
        peakID = int(peak_modelsort.get(iter_, 0)[0])
        return peakID

    def _update_widgets(self):
        """Reacts upon changing the active spectra by updating the widgets."""
        regions_addbox = self._builder.get_object("region_add_box")
        regions_stack = self._builder.get_object("region_contentbox")
        peak_parambox = self._builder.get_object("peak_param_box")

        fwhm_entry = self._builder.get_object("peak_fwhm_entry")
        area_entry = self._builder.get_object("peak_area_entry")
        pos_entry = self._builder.get_object("peak_position_entry")
        name_entry = self._builder.get_object("peak_name_entry")

        regionID = ActiveIDs.REGION
        peakID = ActiveIDs.PEAK
        spectrumIDs = ActiveIDs.SPECTRA

        if peakID:
            def get_c_string(attr):
                """Get constraint string for peak.attr."""
                constraints = self._dh.get_peak_constraints(peakID, attr)
                cstring = ""
                if constraints["expr"]:
                    cstring += str(constraints["expr"])
                elif not constraints["vary"]:
                    cstring = str(self._dh.get(peakID, attr))
                else:
                    if constraints["min_"] not in (-np.inf, 0):
                        cstring += "> {:.2f} ".format(constraints["min_"])
                    if constraints["max_"] != np.inf:
                        cstring += "< {:.2f}".format(constraints["max_"])
                return cstring
            fwhm_entry.set_text(get_c_string("fwhm"))
            area_entry.set_text(get_c_string("area"))
            pos_entry.set_text(get_c_string("center"))
            name_entry.set_text(self._dh.get(peakID, "name"))

        regions_addbox.set_sensitive(
            len(spectrumIDs) == 1
        )
        self._region_chooser.set_sensitive(
            len(spectrumIDs) == 1
            and self._dh.children(spectrumIDs[0])
        )
        regions_stack.set_sensitive(
            len(spectrumIDs) == 1
            and regionID is not None
        )
        peak_parambox.set_sensitive(
            len(spectrumIDs) == 1
            and regionID is not None
            and peakID is not None
        )

    def _on_region_added(self, regionID):
        """Adds a region to the region combo box."""
        region_name = self._dh.get(regionID, "name")
        self._region_chooser.append(str(regionID), region_name)
        logger.debug("regionview: added region {}".format(regionID))
        self.activate_region(regionID)

    def _on_region_removed(self, regionID):
        """Removes a region from the region combo box."""
        model = self._region_chooser.get_model()
        for i, row in enumerate(model):
            if int(row[1]) == regionID:
                model.remove(row.iter)
                logger.debug("regionview: removed region {}".format(regionID))
                if i > 0:
                    self.activate_region(int(model[i-1][-1]))
                elif len(model): # pylint: disable=len-as-condition
                    self.activate_region(int(model[0][-1]))
                else:
                    self.activate_region(None)
                break
        else:
            self._update_widgets()

    def _on_regions_cleared(self, spectrumID):
        """Removes all regions from region combo box."""
        self._region_chooser.remove_all()
        logger.debug("regionview: cleared regions from spectrum {}"
                     "".format(spectrumID))
        self.activate_region(None)

    def _on_peak_added(self, peakID):
        """Add peak to the peakview. It must already be present
        in the DataHandler."""
        peakdict = dict(
            (attr, self._dh.get(peakID, attr))
            for attr in self._peak_model.titles
        )
        self._peak_model.append(peakdict)
        logger.debug("peakview: added peak {}".format(peakID))
        self.activate_peak(peakID)

    def _on_peak_removed(self, peakID):
        """Removes peak from the TreeView. It does not matter if it is
        still in the DataHandler."""
        for i, row in enumerate(self._peak_model):
            if int(row[0]) == peakID:
                self._peak_model.remove(row.iter)
                if i > 0:
                    self.activate_peak(int(self._peak_model[i-1][0]))
                elif len(self._peak_model): # pylint: disable=len-as-condition
                    self.activate_peak(int(self._peak_model[0][0]))
                else:
                    self.activate_peak(None)
                logger.debug("peakview: removed peak {}".format(peakID))
                return
        raise logger.warning("peakview: not shown peak {} cannot be removed"
                             "".format(peakID))

    def _on_peak_changed(self, peakID, attr, isvalid=True):
        """Refreshes peak param column."""
        if attr not in PEAK_TITLES:
            return
        col_index = self._peak_model.get_col_index(attr)
        for row in self._peak_model:
            peakID_other = int(row[0])
            value = str(self._dh.get(peakID_other, attr))
            self._peak_model.set(row.iter, col_index, value)
        if peakID == ActiveIDs.PEAK:
            entry = None
            if attr == "center":
                entry = self._builder.get_object("peak_position_entry")
            elif attr == "area":
                entry = self._builder.get_object("peak_area_entry")
            elif attr == "fwhm":
                entry = self._builder.get_object("peak_fwhm_entry")
            imgname = "dialog-warning-symbolic" if not isvalid else None
            if entry:
                entry.set_icon_from_icon_name(
                    Gtk.EntryIconPosition.SECONDARY,
                    imgname
                )
        logger.debug("peakview: updated peak {}, attr '{}' is now '{}'"
                     "".format(peakID, attr, value))

    def _on_peaks_cleared(self, regionID):
        """Removes all peaks from the model."""
        self._peak_model.clear()
        logger.debug("peakview: cleared peaks from region {}".format(regionID))
        self.activate_peak(None)

    def _on_fit_region(self, regionID):
        """Updates the treeview for changed peak data."""
        if regionID == ActiveIDs.REGION:
            for row in self._peak_model:
                peakID = int(row[0])
                attrs = PEAK_TITLES.keys()
                idxs = [self._peak_model.get_col_index(attr) for attr in attrs]
                values = [str(self._dh.get(peakID, attr)) for attr in attrs]
                self._peak_model.set(row.iter, idxs, values)
        else:
            logger.warning("inconsistency: region {} should be plotted, but "
                           "{} active".format(regionID, ActiveIDs.REGION))

    def _make_peakview_columns(self):
        """Initializes columns. Must therefore be called in __init__."""
        def render_func(_col, renderer, model, iter_, data):
            """Renders the the values so that numbers are rounded."""
            attr, col_index = data
            value = model.get_value(iter_, col_index)
            peakID = int(model.get_value(iter_, 0))
            if value.replace(".", "", 1).isdigit():
                if attr == "area":
                    value = str(int(float(value)))
                else:
                    value = "{:.2f}".format(float(value))
            if attr in ("center", "fwhm", "area"):
                constraints = self._dh.get_peak_constraints(peakID, attr)
                cstring = ""
                if constraints["expr"]:
                    cstring += " = {}".format(constraints["expr"])
                elif not constraints["vary"]:
                    cstring += " fixed"
                else:
                    if constraints["min_"] not in (-np.inf, 0):
                        cstring += " &gt; {:.2f}".format(constraints["min_"])
                    if constraints["max_"] != np.inf:
                        cstring += " &lt; {:.2f}".format(constraints["max_"])
                value = ("{}<span color='#999999' font_size='xx-small'>"
                         "{}</span>".format(value, cstring))
            renderer.set_property("markup", value)
        for attr, title in PEAK_TV_TITLES.items():
            renderer = Gtk.CellRendererText(xalign=0)
            col_index = self._peak_model.get_col_index(attr)
            column = Gtk.TreeViewColumn(title, renderer)
            column.set_cell_data_func(renderer, render_func, (attr, col_index))
            column.set_sort_column_id(col_index)
            column.set_resizable(True)
            column.set_reorderable(True)
            self._peak_treeview.append_column(column)
