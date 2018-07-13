"""This module provides a class with GUI-friendly methods for controling
the DataHandler from xpl.datahandler."""
# pylint: disable=invalid-name
# pylint: disable=wrong-import-position
# pylint: disable=too-many-instance-attributes
# pylint: disable=logging-format-interpolation

from collections import OrderedDict
import re
import logging
import os

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Pango

from xpl.fileio import RSFHandler
from xpl.canvas_selection import DraggableAttributeLine
from xpl import __config__, COLORS


logger = logging.getLogger(__name__)

SPECTRUM_TITLES = OrderedDict({
    "ID": "ID",
    "name": "Name",
    "notes": "Notes",
    "eis_region": "EIS Region",
    "filename": "File name",
    "raw_sweeps": "Sweeps",
    "raw_dwelltime": "Dwell Time [s]",
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
        "raw_sweeps",
        "raw_dwelltime",
        "pass_energy"
    )
)
EXCLUDING_KEY = " (multiple)"

PEAK_TITLES = OrderedDict({
    "name": "Name",
    "model": "Model"
})


class XPLView():
    """This class provides methods for manipulating the view, accessing the
    DataHandler only to retrieve data, not setting them."""
    def __init__(self, builder, datahandler):
        self._builder = builder
        self._dh = datahandler
        self._tviface = XPLTreeViewInterface(builder, datahandler)
        self._cviface = XPLCanvasInterface(builder, datahandler)
        self._fitiface = XPLFitInterface(builder, datahandler)

        self.get_selected_spectra = self._tviface.get_selected_spectra
        self.get_active_spectra = self._cviface.get_active_spectra
        self.get_active_region = self._fitiface.get_active_region
        self.get_active_peak = self._fitiface.get_active_peak
        self.filter_spectra = self._tviface.filter_treeview
        self.show_rsf = self._cviface.show_rsf

    def activate(self, IDs):
        """Plot only stuff with given IDs."""
        self._cviface.plot_only(IDs)
        specIDs = [ID for ID in IDs if self._dh.get(ID, "type") == "spectrum"]
        self._tviface.mark_active(specIDs)
        self._fitiface.activate_spectra(specIDs)
        self._revert_adjustment_widgets()

    def set_region_boundary_setter(self, setter):
        """Sets a callback to be called when a DraggableAttributeLine
        belonging to a region is moved."""
        self._cviface.dragline_callback = setter

    def pop_spectrum_menu(self, event):
        """Pops up the spectrum menu and returns True if the event was
        in an already selected treeview row."""
        self._tviface.pop_spectrum_context_menu(event)
        return self._tviface.get_clicked_in_selection(event)

    def get_edit_spectra_dialog(self, IDs):
        """Customizes the dialog for editing spectra."""
        dialog = self._builder.get_object("edit_spectrum_dialog")
        dialog.flush()
        if len(IDs) == 1:
            dialog.populate_filename(self._dh.get(IDs[0], "filename"))
            for attr, title in EDIT_TITLES.items():
                value = str(self._dh.get(IDs[0], attr))
                dialog.add_attribute(attr, title, value)
        else:
            fnames = ""
            for ID in IDs:
                fnames += self._dh.get(ID, "filename") + "\n"
            fnames = fnames.strip()
            dialog.populate_filename(fnames)
            for attr, title in EDIT_TITLES.items():
                valueset = set([str(self._dh.get(ID, attr)) for ID in IDs])
                value = " | ".join(valueset) + EXCLUDING_KEY
                dialog.add_attribute(attr, title, value)
        return dialog

    def _revert_adjustment_widgets(self):
        """If something different is selected, all adjustment widgets that
        affect spectra have to be reset."""
        cautions = (
            self._builder.get_object("caution_image1"),
            self._builder.get_object("caution_image2"),
            self._builder.get_object("caution_image3")
        )
        smooth = self._builder.get_object("smoothing_scale_adjustment")
        cal = self._builder.get_object("calibration_spinbutton_adjustment")
        norm = self._builder.get_object("normalization_switch")

        IDs = self._cviface.get_active_spectra()
        if len(IDs) != 1:
            for caution in cautions:
                caution.set_visible(True)
            # smooth.set_value(self._dh.get(IDs[0], "smoothness"))
            # cal.set_value(self._dh.get(IDs[0], "calibration"))
        else:
            for caution in cautions:
                caution.set_visible(False)
            smooth.set_value(self._dh.get(IDs[0], "smoothness"))
            cal.set_value(self._dh.get(IDs[0], "calibration"))
            norm.set_active(self._dh.get(IDs[0], "norm"))


class XPLTreeViewInterface():
    """This class provides methods for manipulating the view, accessing the
    DataHandler only to retrieve data."""
    def __init__(self, builder, datahandler):
        self._dh = datahandler

        # register callbacks into the data handler to reflect changes in
        # the meta data values or added/deleted spectra
        handlers = {
            "added-spectrum": self._add_spectrum,
            "removed-spectrum": self._remove_spectrum,
            "amended-spectrum": self._amend_spectrum,
            "cleared-spectra": self._clear_spectra
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
        IDs = [int(treemodelsort.get(iter_, 0)[0]) for iter_ in iters]
        logger.debug("get treeview selected IDs: {}".format(IDs))
        return IDs

    def set_selected_spectra(self, IDs):
        """Sets selection to given spectra IDs."""
        self._selection.unselect_all()
        for row in self._treemodelsort:
            if int(row[0]) in IDs:
                self._selection.select_iter(row.iter)
        logger.debug("set treeview selected IDs: {}".format(IDs))

    def get_clicked_in_selection(self, event):
        """Returns True if event location is on an already selected
        spectrum. This is needed for the context menu to know if the
        current selection should be kept when popping it up or if only
        the row that was clicked should be selected."""
        posx, posy = int(event.x), int(event.y)
        pathinfo = self._treeview.get_path_at_pos(posx, posy)
        if pathinfo is None:
            return False
        path, _col, _cellx, _celly = pathinfo
        return path in self._selection.get_selected_rows()[1]

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
        logger.debug("searched spectrum treeview: attr '{}', regex '{}'"
                     "".format(search_attr, search_term))

    def mark_active(self, IDs):
        """Mark only the spectra with IDs as active."""
        col_index_active = self._treemodel.get_col_index("active")
        for row in self._treemodel:
            if int(row[0]) in IDs:
                self._treemodel.set(row.iter, col_index_active, True)
            else:
                self._treemodel.set(row.iter, col_index_active, False)

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
                    COLORS["tv-highlight-bg"]
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

    def _add_spectrum(self, ID):
        """Adds spectrum to the the TreeView. It must already be present
        in the DataHandler."""
        specdict = dict(
            (attr, self._dh.get(ID, attr))
            for attr in self._treemodel.titles
        )
        specdict["active"] = False
        self._treemodel.append(specdict)
        logger.debug("added spectrum with ID {} to treeview".format(ID))

    def _remove_spectrum(self, ID):
        """Removes spectrum from the TreeView. It does not matter if it is
        still in the DataHandler."""
        for row in self._treemodel:
            if int(row[0]) == ID:
                self._treemodel.remove(row.iter)
        logger.debug("removed spectrum with ID {} from treeview".format(ID))

    def _amend_spectrum(self, ID, newdict):
        """Changes spectrum data by applying newdict which contains all
        (attr: newvalue) pairs that should be updated."""
        col_indexes = []
        values = []
        for attr, value in newdict.items():
            col_indexes.append(self._treemodel.get_col_index(attr))
            values.append(value)
        for row in self._treemodel:
            if int(row[0]) == ID:
                self._treemodel.set(row.iter, col_indexes, values)
        logger.debug("updated spectrum with ID {} in treeview with new"
                     "values {}".format(ID, newdict))

    def _clear_spectra(self):
        """Removes all spectra from the model."""
        self._treemodel.clear()
        logger.debug("cleared all spectra from treeview")


class XPLCanvasInterface():
    """Methods for showing stuff on the canvas."""
    def __init__(self, builder, datahandler):
        self._dh = datahandler

        # register callbacks into the data handler to reflect changes in
        # the meta data values or added/deleted spectra
        handlers = {
            "changed-data": self._on_dh_data_changed,
            "removed-spectrum": self._on_dh_data_changed,
            "cleared-spectra": self._on_dh_data_changed,
            "added-region": self._on_dh_data_changed,
            "removed-region": self._on_dh_data_changed,
            "cleared-regions": self._on_dh_data_changed
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
        rsf_path = os.path.join(__config__.get("general", "basedir"), "rsf.db")
        self.rsfhandler = RSFHandler(rsf_path)
        # set up which rsf data should be plotted
        # by rsf_filter = (elements, source)
        self.rsf_filter = ([], "")

        # dragline_callback is evoked when a DraggableVLine is moved
        # all of those that are avtive are stored in self._draglines
        self.dragline_callback = lambda ID, **kwargs: None
        self._draglines = []

        # contains all IDs for objects that should be plotted
        self._activeIDs = []

        self._plot()

    def get_active_spectra(self):
        """Returns which spectra are on the canvas right now."""
        spectrum_IDs = [
            ID for ID in self._activeIDs
            if self._dh.get(ID, "type") == "spectrum"
        ]
        logger.debug("get actived IDs: {}".format(spectrum_IDs))
        return spectrum_IDs

    def add_to_plot(self, IDs):
        """Add specific IDs to be plotted."""
        self._activeIDs.extend(IDs)
        self._plot()

    def remove_from_plot(self, IDs):
        """Unplot specific IDs."""
        for ID in IDs:
            self._activeIDs.remove(ID)
        self._plot()

    def clear_plot(self):
        """Clear plot."""
        self._activeIDs.clear()
        self._plot()

    def plot_only(self, IDs):
        """Plot only the given IDs."""
        self._activeIDs.clear()
        self.add_to_plot(IDs)

    def show_rsf(self, elements, source):
        """Add elements to the rsfs to be shown."""
        self.rsf_filter = (elements, source)
        self._plot()

    def _update_active(self):
        """Update self._activeIDs with child and grandchild IDs."""
        for ID in self._activeIDs:
            if not self._dh.ID_exists(ID):
                self._activeIDs.remove(ID)
                continue
            for child_ID in self._dh.children(ID):
                if child_ID not in self._activeIDs:
                    self._activeIDs.append(child_ID)
                for grandchild_ID in self._dh.children(child_ID):
                    if grandchild_ID not in self._activeIDs:
                        self._activeIDs.append(child_ID)

    def _plot(self, keepaxes=False):
        """Plots every ID in self._activeIDs after checking for child and
        grandchild IDs. Also plots RSF data"""
        self._update_active()
        logger.debug("plotting IDs {} ...".format(self._activeIDs))

        # prepare by deleting everything old and storing the current xylims
        self._fig.store_xylims()
        self._draglines.clear()
        self._ax.cla()

        # check if there actually is something to plot
        if self._activeIDs:
            self._fig.reset_xy_centerlims()
        else:
            self._canvas.draw_idle()

        # call the individual plot jobs
        for ID in self._activeIDs:
            type_ = self._dh.get(ID, "type")
            if type_ == "spectrum":
                self._plot_spectrum(ID)
            elif type_ == "region":
                self._plot_region(ID)
            elif type_ == "peak":
                self._plot_peak(ID)
        if self.rsf_filter[0]:
            logger.debug("plotting rsfs {}".format(self.rsf_filter))
            self._plot_rsfs(self.rsf_filter[0], self.rsf_filter[1])

        # restore xylims if necessary and then redraw
        if keepaxes:
            self._fig.restore_xylims()
        else:
            self._fig.center_view()
        self._canvas.draw_idle()

    def _plot_spectrum(self, ID):
        """Plots a spectrum and updates the xylims for a centered view."""
        lineprops = {
            "color": COLORS["spectrum"],
            "linewidth": 1,
            "linestyle": "-",
            "alpha": 1
        }
        energy, cps, name = self._dh.get_multiple(ID, "energy", "cps", "name")
        self._ax.plot(energy, cps, label=name, **lineprops)

        erange = max(energy) - min(energy)
        # crange = max(cps) - min(cps)
        self._fig.update_xy_centerlims(
            min(energy) - erange * 0.02,
            max(energy) + erange * 0.02,
            0,          # min(cps) - crange * 0.05,
            max(cps) * 1.1
        )

    def _plot_region(self, ID):
        """Plots a region by plotting its limits with DraggableVLines and
        plotting the background intensity."""
        lineprops = {
            "color": COLORS["region-vlines"],
            "linewidth": 2,
            "linestyle": "--",
            "alpha": 1
        }
        line = self._ax.axvline(self._dh.get(ID, "emin"), 0, 1, **lineprops)
        self._draglines.append(DraggableAttributeLine(
            line, ID, "emin", self.dragline_callback))
        line = self._ax.axvline(self._dh.get(ID, "emax"), 0, 1, **lineprops)
        self._draglines.append(DraggableAttributeLine(
            line, ID, "emax", self.dragline_callback))

        if self._dh.get(ID, "background") is None:
            return
        lineprops = {
            "color": COLORS["region-background"],
            "linewidth": 1,
            "linestyle": "--",
            "alpha": 1}
        energy, background = self._dh.get_multiple(ID, "energy", "background")
        self._ax.plot(energy, background, **lineprops)

    def _plot_peak(self, ID):
        """Plots a peak."""
        pass

    def _plot_rsfs(self, elements, source):
        """Plots RSF values for given elements with given X-ray souce."""
        # fetch data through the rsfhandler
        rsfs = []
        for element in elements:
            dicts = self.rsfhandler.get_element(element, source)
            if dicts:
                rsfs.append(dicts)
            else:
                logger.info("element {} not in database".format(element))

        # set up colors and maximum rsf intensity for scaling the vlines
        if not rsfs:
            return
        max_rsf = max(max([[d["RSF"] + 1e-9 for d in ds] for ds in rsfs]))
        colors = COLORS["rsf-vlines"] * 10

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
                    color=COLORS["rsf-annotation"],
                    textcoords="data",
                    ha="center",
                    va="bottom"
                )

    def _on_dh_data_changed(self, ID=None, attr=None):
        """Replots if the ID is affected. Depending on changed attr, the
        axislims are kept or not."""
        keepaxes = not attr in ("norm",)
        # the ID is affected either if it is active or its parent is
        # also, if no ID is given, we should replot
        if (ID is None
                or ID in self._activeIDs
                or self._dh.parent(ID) in self._activeIDs):
            logger.debug("replotting because datahandler changed...")
            self._plot(keepaxes)


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
            "removed-peak": self._on_peak_removed,
            "cleared-peaks": self._on_peaks_cleared
        }
        for signal, handler in handlers.items():
            self._dh.connect(signal, handler)

        # get all the gui elements defined in the xml file
        self._region_chooser = self._builder.get_object("region_chooser_combo")
        self._peak_parambox = self._builder.get_object("peak_param_box")
        self._peak_treeview = self._builder.get_object("peak_view")
        self._peak_selection = self._builder.get_object("peak_selection")
        self._peak_model = self._builder.get_object("peak_treestore")

        # this class needs the active spectra for updating the widgets
        # correctly (this list is updated through activate_spectra)
        self._active_specIDs = []

        self._make_peakview_columns()

    def get_active_region(self):
        """Returns the region that is currently selected or None."""
        activeID = self._region_chooser.get_active_id()
        if activeID is None:
            return activeID
        return int(activeID)

    def get_active_peak(self):
        """Returns the peak that is currently selected or None."""
        peak_modelsort, pathlist = self._peak_selection.get_selected_rows()
        if not pathlist:
            return None
        iter_ = peak_modelsort.get_iter(pathlist[0])
        ID = int(peak_modelsort.get(iter_, 0)[0])
        logger.debug("get peakview selected ID: {}".format(ID))
        return ID

    def activate_spectra(self, IDs=None):
        """Reacts upon changing the active spectra by refilling the
        region_chooser and updating the widgets."""
        if IDs is None or IDs == self._active_specIDs:
            self._update_widgets()
            return
        self._region_chooser.remove_all()
        if len(IDs) == 1:
            child_IDs = self._dh.children(IDs[0])
            for ID in child_IDs:
                self._region_chooser.append(str(ID), self._dh.get(ID, "name"))
                self._region_chooser.set_active_id(str(child_IDs[0]))
        self._active_specIDs.clear()
        self._active_specIDs.extend(IDs)
        self._update_widgets()

    def _update_widgets(self):
        """Reacts upon changing the active spectra by updating the widgets."""
        regions_addbox = self._builder.get_object("region_add_box")
        regions_stack = self._builder.get_object("region_contentbox")
        peak_parambox = self._builder.get_object("peak_param_box")

        regions_addbox.set_sensitive(len(self._active_specIDs) == 1)
        self._region_chooser.set_sensitive(
            len(self._active_specIDs) == 1
            and self._dh.children(self._active_specIDs[0])
            )
        regions_stack.set_sensitive(
            len(self._active_specIDs) == 1
            and self.get_active_region() is not None
            )
        peak_parambox.set_sensitive(
            len(self._active_specIDs) == 1
            and self.get_active_region() is not None
            and self.get_active_peak() is not None
            )

    def _on_region_added(self, ID):
        """Adds a region to the region combo box."""
        self._region_chooser.append(str(ID), self._dh.get(ID, "name"))
        self._region_chooser.set_active_id(str(ID))
        self._update_widgets()

    def _on_region_removed(self, ID):
        """Removes a region from the region combo box."""
        model = self._region_chooser.get_model()
        for i, row in enumerate(model):
            if int(row[1]) == ID:
                model.remove(row.iter)
                if i > 0:
                    self._region_chooser.set_active_iter(model[i-1].iter)
                # this is necessary because "if model:" is always True
                elif len(model): # pylint: disable=len-as-condition
                    self._region_chooser.set_active_iter(model[i].iter)
                break
        self._update_widgets()

    def _on_regions_cleared(self, ID):
        """Removes all regions from region combo box."""
        if len(self._active_specIDs) == 1 and self._active_specIDs[0] == ID:
            self._region_chooser.remove_all()
            self._update_widgets()

    def _on_peak_added(self, ID):
        """Add peak to the peakview. It must already be present
        in the DataHandler."""
        peakdict = dict(
            (attr, self._dh.get(ID, attr))
            for attr in self._peak_model.titles
        )
        self._peak_model.append(peakdict)
        logger.debug("added peak with ID {} to treeview".format(ID))

    def _on_peak_removed(self, ID):
        """Removes peak from the TreeView. It does not matter if it is
        still in the DataHandler."""
        for row in self._peak_model:
            if int(row[0]) == ID:
                self._peak_model.remove(row.iter)
        logger.debug("removed peak with ID {} from treeview".format(ID))

    def _on_peaks_cleared(self):
        """Removes all peaks from the model."""
        self._peak_model.clear()
        logger.debug("cleared all peaks from peakview")

    def _make_peakview_columns(self):
        """Initializes columns. Must therefore be called in __init__."""
        for attr, title in PEAK_TITLES.items():
            renderer = Gtk.CellRendererText(xalign=0)
            col_index = self._peak_model.get_col_index(attr)
            column = Gtk.TreeViewColumn(title, renderer, text=col_index)
            column.set_sort_column_id(col_index)
            column.set_resizable(True)
            column.set_reorderable(True)
            self._peak_treeview.append_column(column)
