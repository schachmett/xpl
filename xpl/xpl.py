"""Main GUI module, manages the main window and the application class where
all user accessible actions are defined."""
# pylint: disable=wrong-import-position
# pylint: disable=invalid-name
# pylint: disable=logging-format-interpolation

import re
import logging
import sys
import os

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Gio, GLib, Gdk

from xpl import (__appname__, __version__, __authors__, __website__,
                 __config__, __colors__,
                 BASEDIR, COLOR_CFG_PATH, CFG_PATH, LOG_PATH)
from xpl.view import XPLView
import xpl.fileio as fileio
from xpl.datahandler import DataHandler

# imports for Gtk.Builder:
# pylint: disable=unused-import
import xpl.gui
# pylint: enable=unused-import

logger = logging.getLogger(__name__)
settings = Gtk.Settings.get_default()
settings.set_property("gtk-application-prefer-dark-theme", True)


def make_option(long_name, short_name=None, arg=GLib.OptionArg.NONE, **kwargs):
    """Make GLib option for the command line. Uses kwargs description, flags,
    arg_data and arg_description."""
    # surely something like this should exist inside PyGObject itself?!
    option = GLib.OptionEntry()
    option.long_name = long_name.lstrip('-')
    option.short_name = 0 if not short_name else ord(short_name.lstrip('-'))
    option.arg = arg
    option.flags = kwargs.get("flags", 0)
    option.arg_data = kwargs.get("arg_data", None)
    option.description = kwargs.get("description", None)
    option.arg_description = kwargs.get("arg_description", None)
    return option


class XPL(Gtk.Application):
    """Application class, this has action and signal callbacks."""
    # pylint: disable=arguments-differ
    # pylint: disable=too-many-public-methods
    def __init__(self):
        logger.debug("instantiating application...")
        app_id = "org.{}.app".format(__appname__.lower())
        super().__init__(
            application_id=app_id,
            flags=Gio.ApplicationFlags.HANDLES_COMMAND_LINE
        )
        GLib.set_application_name(__appname__)

        self.builder = Gtk.Builder.new_from_file(
            str(BASEDIR / "xpl/xpl.glade"))
        self.builder.add_from_file(str(BASEDIR / "xpl/menubar.ui"))

        self.dh = DataHandler()
        self.view = XPLView(self.builder, self.dh)
        self.view.set_region_boundary_setter(self.dh.manipulate_region)
        self.parser = fileio.FileParser()

        statusbar = self.builder.get_object("statusbar")
        self.statusbar_id = statusbar.get_context_id("")

        self.win = None

        self.add_main_option_entries([
            make_option("--verbosity", "-v", arg=GLib.OptionArg.INT,
                        description="Value from 1 (only errors) to 4 (debug)"),
            make_option("--version", description="show program version"),
        ])

    def do_activate(self):
        """Creates MainWindow."""
        self.win = self.builder.get_object("main_window")
        self.win.set_application(self)
        self.win.set_helpers(self.builder, self.view)
        self.win.set_title(__appname__)
        self.win.show_all()

        def alter_dh(altered=True):
            """
            Stars the filename in the window title when datahandler
            gets altered.
            """
            title = self.win.get_title()
            if altered and u"—" in title and "*" not in title:
                fname = title.split(u"—")[0].strip()
                self.win.set_title(u"{}* — {}".format(fname, __appname__))
            elif not altered and "*" in title:
                self.win.set_title(title.replace("*", ""))

        self.dh.connect("altered", alter_dh)


        fname = __config__.get("io", "project_file")
        if fname != "None":
            try:
                self.open_project(fname)
            except FileNotFoundError:
                self.dh.emit_init_ready()
                logger.warning("File '{}' not found".format(fname))
                __config__.set("io", "project_file", "None")
            else:
                logger.info("loaded file {}".format(fname))
        else:
            self.dh.emit_init_ready()

        handlers = {
            "on_main_window_delete_event": self.on_quit,
            "on_smoothing_scale_adjustment_value_changed": self.on_smoothen,
            "on_calibration_spinbutton_adjustment_value_changed":
                self.on_calibrate,
            "on_normalization_switch_activate": self.on_normalize,
            "on_region_background_type_combo_changed": self.on_change_bgtype,
            "on_peak_entry_activate": self.on_peak_entry_activate,
            "on_peak_name_entry_changed": self.on_peak_name_entry_changed,
        }
        handlers.update(self.win.handlers)
        self.builder.connect_signals(handlers)

    def do_startup(self):
        """Adds the actions and the menubar."""
        logger.info("starting application...")
        Gtk.Application.do_startup(self)
        self.set_menubar(self.builder.get_object("main_menubar"))

        actions = {
            "new": self.on_new,
            "save-project": self.on_save,
            "save-project-as": self.on_save_as,
            "open-project": self.on_open_project,
            "import-spectra": self.on_import_spectra,
            "remove-spectra": self.on_remove_spectra,
            "edit-spectra": self.on_edit_spectra,
            "remove-region": self.on_remove_region,
            "clear-regions": self.on_clear_regions,
            "add-region": self.on_add_region,
            "add-peak": self.on_add_peak,
            "add-guessed-peak": self.on_add_guessed_peak,
            "remove-peak": self.on_remove_peak,
            "clear-peaks": self.on_clear_peaks,
            "avg-selected-spectra": self.on_avg_selected_spectra,
            "fit": self.on_fit,
            "quit": self.on_quit
        }
        for name, callback in actions.items():
            simple = Gio.SimpleAction.new(name, None)
            simple.connect("activate", callback)
            self.add_action(simple)

    def do_command_line(self, command_line):
        """Handles command line arguments"""
        Gtk.Application.do_command_line(self, command_line)
        options = command_line.get_options_dict()
        if options.contains("verbosity"):
            verb = options.lookup_value("verbosity", GLib.VariantType("i"))
            levels = (
                None,
                logging.ERROR,
                logging.WARNING,
                logging.INFO,
                logging.DEBUG
                )
            logging.getLogger().handlers[0].setLevel(levels[verb.unpack()])
        if options.contains("version"):
            print("{} version: {}".format(__appname__, __version__))
            self.quit()
        self.activate()
        return 0

    def ask_for_save(self):
        """Opens a AskForSaveDialog and runs the appropriate methods,
        then returns True if user really wants to close current file."""
        if not self.dh.altered:
            logger.debug("no need for saving, datahandler not altered")
            return True
        if self.dh.isempty():
            logger.debug("no need for saving, datahandler empty")
            return True
        dialog = self.builder.get_object("save_confirmation_dialog")
        response = dialog.run()
        if response == Gtk.ResponseType.YES:
            dialog.hide()
            is_saved = self.on_save(dialog)
            return is_saved
        if response == Gtk.ResponseType.NO:
            dialog.hide()
            logger.debug("user says: do not save")
            return True
        dialog.hide()
        logger.debug("doing nothing, user aborts ask_for_save")
        return False

    def on_new(self, _widget, *_args):
        """Clear data and start a new project after asking if the user is
        sure."""
        really_do_it = self.ask_for_save()
        if really_do_it:
            self.view.activate_spectra([])
            self.dh.clear_spectra()
            self.win.set_title(u"{}* — {}".format("Untitled", __appname__))
            __config__.set("io", "project_file", "None")
            logger.info("emptied datahandler, new project")

    def on_save(self, widget, *_args):
        """Save the current project, call self.on_save_as if project file
        name is not set."""
        fname = __config__.get("io", "project_file")
        if fname != "None":
            fileio.save_project(fname, self.dh)
            logger.info("saved project file to {}".format(fname))
            return True
        return self.on_save_as(widget)

    def on_save_as(self, widget, *_args):
        """Saves project to a new file chosen by the user."""
        dialog = Gtk.FileChooserDialog(
            "Save as...",
            self.win,
            Gtk.FileChooserAction.SAVE,
            ("_Cancel", Gtk.ResponseType.CANCEL, "_Save", Gtk.ResponseType.OK),
        )
        dialog.set_current_folder(__config__.get("io", "project_dir"))
        dialog.set_do_overwrite_confirmation(True)
        dialog.add_filter(SimpleFileFilter(".xpl", ["*.xpl"]))
        dialog.set_current_name("untitled.xpl")
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            fname = dialog.get_filename()
            __config__.set("io", "project_file", fname)
            __config__.set("io", "project_dir", dialog.get_current_folder())
            self.on_save(widget)
            dialog.destroy()
            return True
        logger.debug("abort project file saving")
        dialog.destroy()
        return False

    def on_open_project(self, _widget, *_args):
        """Let the user choose a project file to open and open it through
        self.open_project."""
        really_do_it = self.ask_for_save()
        if not really_do_it:
            return
        dialog = Gtk.FileChooserDialog(
            "Open...",
            self.win,
            Gtk.FileChooserAction.OPEN,
            ("_Cancel", Gtk.ResponseType.CANCEL, "_Open", Gtk.ResponseType.OK),
        )
        dialog.set_current_folder(__config__.get("io", "project_dir"))
        dialog.add_filter(SimpleFileFilter(".xpl", ["*.xpl"]))
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            fname = dialog.get_filename()
            __config__.set("io", "project_file", fname)
            __config__.set("io", "project_dir", dialog.get_current_folder())
            self.open_project(fname)
        else:
            logger.debug("abort project file opening")
        dialog.destroy()

    def open_project(self, fname):
        """Load a project file."""
        self.view.activate_spectra([])
        fileio.load_project(fname, self.dh)
        self.win.set_title(u"{} — {}".format(fname, __appname__))
        logger.info("opened project file {}".format(fname))

    def on_export_as_txt(self, _action, *_args):    #TODO gui action
        """Export currently viewed stuff as txt."""
        dialog = Gtk.FileChooserDialog(
            "Export as txt...",
            self.win,
            Gtk.FileChooserAction.SAVE,
            ("_Cancel", Gtk.ResponseType.CANCEL, "_Save", Gtk.ResponseType.OK),
        )
        dialog.set_current_folder(__config__.get("io", "export_dir"))
        pfile = __config__.get("io", "project_file")
        if pfile != "None":
            bname = "{}_data.txt".format(os.path.basename(pfile).split(".")[0])
        else:
            bname = "Untitled.txt"
        dialog.set_current_name(bname)
        dialog.set_do_overwrite_confirmation(True)
        dialog.add_filter(SimpleFileFilter(".txt", ["*.txt"]))
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            fname = dialog.get_filename()
            spectrumIDs = self.view.get_active_spectra()
            if len(spectrumIDs) != 1:
                logger.warning("txt export only supports single spectra")
            else:
                fileio.export_txt(self.dh, spectrumIDs[0], fname)
            dialog.destroy()
            return True
        logger.debug("abort file export")
        dialog.destroy()
        return False

    def on_import_spectra(self, _widget, *_args):
        """Load one or more spectra from a file chosen by the user."""
        dialog = Gtk.FileChooserDialog(
            "Import data...",
            self.win,
            Gtk.FileChooserAction.OPEN,
            ("_Cancel", Gtk.ResponseType.CANCEL, "_Open", Gtk.ResponseType.OK),
        )
        dialog.set_current_folder(__config__.get("io", "data_dir"))
        dialog.set_select_multiple(True)
        dialog.add_filter(
            SimpleFileFilter("all files", ["*.xym", "*.txt", "*.xy"])
            )
        dialog.add_filter(SimpleFileFilter(".xym", ["*.xym"]))
        dialog.add_filter(SimpleFileFilter(".xy", ["*.xy"]))
        dialog.add_filter(SimpleFileFilter(".txt", ["*.txt"]))

        specdicts = []
        response = dialog.run()
        if response == Gtk.ResponseType.OK:
            __config__.set("io", "data_dir", dialog.get_current_folder())
            for fname in dialog.get_filenames():
                ext = fname.split(".")[-1]
                if ext in ["xym", "txt"]:
                    specdicts.extend(self.parser.parse_spectrum_file(fname))
                    logger.info("parsed and added file {}".format(fname))
                elif ext in ["xy"]:
                    logger.warning("{} not yet supported".format(ext))
                else:
                    logger.warning("file {} not recognized".format(fname))
            for specdict in specdicts:
                self.dh.add_spectrum(**specdict)
        else:
            logger.debug("nothing selected to import")
        dialog.destroy()

    def on_remove_spectra(self, _widget, *_args):
        """Remove the selected spectra from the project."""
        spectrumIDs = self.view.get_selected_spectra()
        for spectrumID in spectrumIDs:
            self.dh.remove_spectrum(spectrumID)
            logger.info("removed spectrum {}".format(spectrumID))

    def on_edit_spectra(self, _widget, *_args):
        """Edit the selected spectra."""
        spectrumIDs = self.view.get_selected_spectra()
        if not spectrumIDs:
            logger.debug("no spectrum selected for editing")
            return
        dialog = self.view.get_edit_spectra_dialog(spectrumIDs)
        response = dialog.run()
        if response == Gtk.ResponseType.APPLY:
            newdict = dialog.get_newdict()
            for spectrumID in spectrumIDs:
                self.dh.amend_spectrum(spectrumID, **newdict)
                logger.info("spectrum {} amended".format(spectrumID))
        dialog.hide()

    def on_add_region(self, _widget, *_args):
        """Adds a region to the currently selected spectrum."""
        spectrumIDs = self.view.get_active_spectra()
        if len(spectrumIDs) != 1:
            return
        spectrumID = spectrumIDs[0]
        def create_region(emin, emax):
            """Add region"""
            regionID = self.dh.add_region(
                spectrumID,
                emin=emin,
                emax=emax
            )
            logger.info("added region {} to spectrum {}"
                        "".format(regionID, spectrumID))
        rectcolor = __colors__.get("plotting", "region-vlines")
        rectprops = {"edgecolor": rectcolor, "linewidth": 2}
        plot_toolbar = self.builder.get_object("plot_toolbar")
        plot_toolbar.get_span(create_region, **rectprops)

    def on_remove_region(self, _widget, *_args):
        """Removes selected regions from the current spectrum."""
        regionID = self.view.get_active_region()
        self.dh.remove_region(regionID)
        logger.info("removed region {}".format(regionID))

    def on_clear_regions(self, _widget, *_args):
        """Delete regions from selected spectra."""
        spectrumIDs = self.view.get_selected_spectra()
        for spectrumID in spectrumIDs:
            self.dh.clear_regions(spectrumID)
        logger.info("removed all regions from spectra {}".format(spectrumIDs))

    def on_add_peak(self, *_args):
        """Adds a peak to the currently active region."""
        regionID = self.view.get_active_region()
        def create_peak(center, height, angle):
            """Add peak"""
            if self.view.get_visible("region-background"):
                peakID = self.dh.add_peak(
                    regionID,
                    totalheight=height,
                    angle=angle,
                    center=center
                )
            else:
                peakID = self.dh.add_peak(
                    regionID,
                    height=height,
                    angle=angle,
                    center=center
                )
            logger.info("added peak {} to region {}".format(peakID, regionID))
        wedgeprops = {
            "edgecolor": __colors__.get("plotting", "peak-wedge-edge"),
            "facecolor": __colors__.get("plotting", "peak-wedge-face")
        }
        plot_toolbar = self.builder.get_object("plot_toolbar")
        rmin = self.dh.get(regionID, "emin")
        rmax = self.dh.get(regionID, "emax")
        plot_toolbar.get_wedge(create_peak, **wedgeprops, limits=(rmin, rmax))

    def on_add_guessed_peak(self, *_args):
        """Adds a peak where the parameters are guessed by lmfit."""
        regionID = self.view.get_active_region()
        peakID = self.dh.add_peak(regionID, guess=True)
        logger.info("guessed peak {} for region {}".format(peakID, regionID))

    def on_remove_peak(self, *_args):
        """Removes currently selected peak."""
        peakID = self.view.get_active_peak()
        if peakID is not None:
            self.dh.remove_peak(peakID)
            logger.info("removed peak {}".format(peakID))

    def on_clear_peaks(self, *_args):
        """Removes all peaks from selected region."""
        regionID = self.view.get_active_region()
        if regionID is not None:
            self.dh.clear_peaks(regionID)
            logger.info("removed all peaks from region {}".format(regionID))

    def on_smoothen(self, widget, *_args):
        """Smoothen selected spectra by value from widget Gtk.Adjustment."""
        smoothness = int(widget.get_value())
        for spectrumID in self.view.get_active_spectra():
            self.dh.manipulate_spectrum(spectrumID, smoothness=smoothness)

    def on_calibrate(self, widget, *_args):
        """Calibrate selectred spectrum energy axes by value from widget
        Gtk.Adjustment."""
        calibration = float(widget.get_value())
        for spectrumID in self.view.get_active_spectra():
            self.dh.manipulate_spectrum(spectrumID, calibration=calibration)

    def on_normalize(self, widget, *_args):
        """Normalizes selected spectra."""
        normalization = widget.get_active()
        for spectrumID in self.view.get_active_spectra():
            self.dh.manipulate_spectrum(spectrumID, norm=normalization)

    def on_avg_selected_spectra(self, _action, *_args):
        """Averages selected spectra and adds that average."""
        spectrumIDs = self.view.get_selected_spectra()
        self.dh.add_averaged_spectrum(spectrumIDs)

    def on_change_bgtype(self, combo, *_args):
        """Changes the background type to combo.get_active_text()."""
        regionID = self.view.get_active_region()
        if regionID:
            bgtype = combo.get_active_text()
            self.dh.manipulate_region(regionID, bgtype=bgtype)

    def on_peak_entry_activate(self, *_args):
        """Apply constraint strings from peak entries."""
        peakID = self.view.get_active_peak()
        def apply_constraint_string(c_string, param):
            """Applies the text of a param_entry to the model."""
            if not c_string:
                self.dh.constrain_peak(peakID, param)
            elif "<" in c_string or ">" in c_string:
                try:
                    min_ = float(c_string.split(">")[1].split()[0])
                except IndexError:
                    min_ = None
                try:
                    max_ = float(c_string.split("<")[1].split()[0])
                except IndexError:
                    max_ = None
                self.dh.constrain_peak(peakID, param, min_=min_, max_=max_)
            else:
                try:
                    value = float(c_string.strip())
                except ValueError:
                    expr = c_string.strip()
                    self.dh.constrain_peak(peakID, param, expr=expr)
                else:
                    self.dh.constrain_peak(
                        peakID, param, vary=False, value=value
                    )
        c_fwhm = self.builder.get_object("peak_fwhm_entry").get_text()
        apply_constraint_string(c_fwhm, "fwhm")
        c_fwhm = self.builder.get_object("peak_area_entry").get_text()
        apply_constraint_string(c_fwhm, "area")
        c_fwhm = self.builder.get_object("peak_position_entry").get_text()
        apply_constraint_string(c_fwhm, "center")

    def on_peak_name_entry_changed(self, *_args):
        """Change name of the peak."""
        peakID = self.view.get_active_peak()
        peakname = self.builder.get_object("peak_name_entry").get_text()
        self.dh.amend_peak(peakID, name=peakname.strip())

    def on_fit(self, *_args):
        """Fits the current peaks."""
        # self.on_peak_entry_activate()
        regionID = self.view.get_active_region()
        if regionID:
            self.dh.fit_region(regionID)

    def on_quit(self, *_args):
        """Quit program, write to config file."""
        really_do_it = self.ask_for_save()
        if not really_do_it:
            return True
        xsize, ysize = self.win.get_size()
        xpos, ypos = self.win.get_position()
        __config__.set("window", "xsize", str(xsize))
        __config__.set("window", "ysize", str(ysize))
        __config__.set("window", "xpos", str(xpos))
        __config__.set("window", "ypos", str(ypos))
        with open(str(CFG_PATH), "w") as cfg_file:
            logger.info("writing config file...")
            __config__.write(cfg_file)
        logger.info("quitting...")
        self.quit()
        return False


class XPLAppWindow(Gtk.ApplicationWindow):
    """Main window handling window specific actions and signal callbacks.
    This is loaded through xpl_catalog.xml from the xpl.glade file."""
    __gtype_name__ = "XPLAppWindow"
    def __init__(self):
        logger.debug("instantiating main window...")
        super().__init__()
        self.app = None
        self.builder = None
        self.view = None
        self.statusbar_id = None
        xsize = int(__config__.get("window", "xsize"))
        ysize = int(__config__.get("window", "ysize"))
        xpos = int(__config__.get("window", "xpos"))
        ypos = int(__config__.get("window", "ypos"))
        self.set_default_size(xsize, ysize)
        self.move(xpos, ypos)

        self.handlers = {
            "on_spectrum_view_search_entry_changed":
                self.on_search_spectrum_view,
            "on_spectrum_view_search_combo_changed":
                self.on_search_spectrum_view,
            "on_spectrum_view_button_press_event":
                self.on_spectrum_view_clicked,
            "on_peak_view_row_activated":
                self.on_peak_view_row_activated,
            "on_region_chooser_combo_changed":
                self.on_region_chooser_changed
        }
        actions = {
            "about": self.on_about,
            "show-selected-spectra": self.on_show_selected_spectra,
            "show-atomlib": self.on_show_atomlib,
            "center-plot": self.on_center_plot,
            "pan-plot": self.on_pan_plot,
            "zoom-plot": self.on_zoom_plot,
            "view-logfile": self.on_view_logfile,
            "edit-colors": self.on_edit_colors
        }
        for name, callback in actions.items():
            simple = Gio.SimpleAction.new(name, None)
            simple.connect("activate", callback)
            self.add_action(simple)
        toggle_appearance_actions = {
            "toggle-bg": "region-background",
            "toggle-peaks": "peak",
            "toggle-region-vlines": "region-boundaries"
        }
        for name, keyword in toggle_appearance_actions.items():
            simple_toggle = Gio.SimpleAction.new_stateful(
                name, None, GLib.Variant.new_boolean(True))
            simple_toggle.connect(
                "change-state", self.on_toggle_appearance, keyword)
            self.add_action(simple_toggle)

    def set_helpers(self, builder, view):
        """Sets the builder because this class does not get any construtor
        parameters since it is loaded itself by a Gtk.Builder."""
        logger.debug("configuring main window...")
        self.builder = builder
        self.view = view

        statusbar = self.builder.get_object("statusbar")
        self.statusbar_id = statusbar.get_context_id("")
        self.__connect_loggers_to_display()

        plot_toolbar = self.builder.get_object("plot_toolbar")
        plot_toolbar_message = self.builder.get_object("mpl_message_label")
        plot_toolbar_coord = self.builder.get_object("mpl_coord_label")
        canvas = self.builder.get_object("main_canvas")
        plot_toolbar.reinit(canvas, plot_toolbar_message, plot_toolbar_coord)

    def on_search_spectrum_view(self, _widget):
        """Applies search term from entry.get_text() to the TreeView in column
        combo.get_active_text()."""
        # damn user_data from glade does not allow to pass both widgets here
        # as arguments, so they must be fetched from the builder
        combo = self.builder.get_object("spectrum_view_search_combo")
        entry = self.builder.get_object("spectrum_view_search_entry")
        self.view.filter_spectra(combo.get_active_text(), entry.get_text())

    def on_region_chooser_changed(self, combo):
        """Sets chosen region in the view."""
        if not combo.has_focus():
            return
        regionID = combo.get_active_id()
        if regionID is not None:
            regionID = int(regionID)
        self.view.activate_region(regionID)

    def on_spectrum_view_clicked(self, treeview, event):
        """Callback for button-press-event, popups the menu on right click
        and calls show_selected for double left click. Return value
        determines if the selection on self persists, True if not."""
        if event.type == Gdk.EventType.BUTTON_PRESS and event.button == 3:
            self.view.pop_spectrum_menu(event)
            pathinfo = treeview.get_path_at_pos(int(event.x), int(event.y))
            if pathinfo is None:
                return False
            selected_rows = treeview.get_selection().get_selected_rows()[1]
            return pathinfo[0] in selected_rows
        # pylint: disable=protected-access
        if event.type == Gdk.EventType._2BUTTON_PRESS and event.button == 1:
            spectrumIDs = self.view.get_selected_spectra()
            self.view.activate_spectra(spectrumIDs)
            return True
        return False

    def on_peak_view_row_activated(self, _treeview, _path, _column):
        """Callback for button-press-event, calls self.view.activate_peak()
        on left (single!) click. Must return False so the clicked peak
        will be selected."""
        peakID = self.view.get_selected_peak()
        self.view.activate_peak(peakID)

    def on_show_selected_spectra(self, _widget, *_ignore):
        """Shows the spectra selected in the treeview in the canvas."""
        spectrumIDs = self.view.get_selected_spectra()
        self.view.activate_spectra(spectrumIDs)

    def on_pan_plot(self, _widget, *_ignore):
        """Activates plot panning."""
        plot_toolbar = self.builder.get_object("plot_toolbar")
        plot_toolbar.pan()

    def on_zoom_plot(self, _widget, *_ignore):
        """Activates plot panning."""
        plot_toolbar = self.builder.get_object("plot_toolbar")
        plot_toolbar.zoom()

    def on_center_plot(self, _widget, *_ignore):
        """Activates plot panning."""
        plot_toolbar = self.builder.get_object("plot_toolbar")
        plot_toolbar.center()

    def on_show_atomlib(self, _widget, *_ignore):
        """Callback for the "show rsf" button, invoking a dialog for element
        and source selection."""
        dialog = self.builder.get_object("rsf_dialog")
        combo = self.builder.get_object("rsf_combo")
        entry = self.builder.get_object("rsf_entry")
        response = dialog.run()
        if response == Gtk.ResponseType.APPLY:
            source = combo.get_active_text()
            elements = re.findall(r"[\w]+", entry.get_text())
            elements = [element.title() for element in elements]
            logger.info("plotting rsf: source {}, elements {}"
                        "".format(source, elements))
            self.view.show_rsf(elements, source)
        elif response == Gtk.ResponseType.REJECT:
            logger.info("removing rsfs from plot...")
            entry.set_text("")
            self.view.show_rsf([], "")
        dialog.hide()

    def on_toggle_appearance(self, action, state, keyword):
        """Gets keywords and True/False if specific objects should be
        plotted."""
        action.set_state(state)
        self.view.set_visible(keyword, bool(state))

    def on_about(self, _widget, *_ignore):
        """Show 'About' dialog."""
        dialog = self.builder.get_object("about_dialog")
        dialog.set_program_name(__appname__)
        dialog.set_authors(__authors__)
        dialog.set_version(__version__)
        dialog.set_license_type(Gtk.License.GPL_3_0)
        commentstring = """If you encounter any bugs, mail me or open an
                        issue on my github. Please include a logfile, it is
                        located at '{}'.
                        """.format(str(LOG_PATH))
        commentstring = " ".join(commentstring.split())
        dialog.set_website(__website__)
        dialog.set_comments(commentstring)
        dialog.run()
        dialog.hide()

    @staticmethod
    def on_view_logfile(_action, *_args):
        """Views logfile in external text editor."""
        if sys.platform.startswith("linux"):
            os.system("xdg-open {}".format(str(LOG_PATH)))
        else:
            logger.warning("logfile viewing only implemented for linux")

    @staticmethod
    def on_edit_colors(_action, *_args):
        """Views colors.ini file in external text editor."""
        if sys.platform.startswith("linux"):
            os.system("xdg-open {}".format(str(COLOR_CFG_PATH)))
        else:
            logger.warning("color file editing only implemented for linux")

    def display(self, message, dolog=True, timetolive=3):
        """Displays a message in the statusbar."""
        if dolog:
            logger.info("statusbar message: {}".format(message))
        statusbar = self.builder.get_object("statusbar")
        message_id = statusbar.push(self.statusbar_id, message)
        def erase_message():
            """Pop message from statusbar."""
            statusbar.remove(self.statusbar_id, message_id)
            return False
        GLib.timeout_add_seconds(timetolive, erase_message)

    def __connect_loggers_to_display(self):
        """Makes a custom log handler using self.display and connects
        it to rootlogger and ExceptionLogger.
        """
        displayfunc = self.display
        class DisplayFormatter(logging.Formatter):
            """Simplest logging Formatter ever."""
            def format(self, record):
                """Simplest formatting ever."""
                string = "{}: {}".format(record.levelname.lower(), record.msg)
                return string

        class DisplayHandler(logging.Handler):
            """Logging Handler that uses statusbar display."""
            def emit(self, record):
                """Push message to statusbar."""
                message = self.format(record)
                displayfunc(message)

        handler = DisplayHandler()
        handler.setLevel(logging.WARNING)
        handler.setFormatter(DisplayFormatter())
        logging.getLogger().addHandler(handler)
        logging.getLogger("ExceptionLogger").addHandler(handler)


class SimpleFileFilter(Gtk.FileFilter):
    """Simpler FileFilter for FileChooserDialogs with better constructor."""
    def __init__(self, name, patterns):
        """ filter for file chooser dialogs """
        super().__init__()
        for pattern in patterns:
            self.add_pattern(pattern)
        self.set_name(name)
