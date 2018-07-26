"""Provides a custom Gtk.Treeview for viewing spectrum metadata and
filtering/sorting/selecting them."""
# pylint: disable=wrong-import-position
# pylint: disable=invalid-name
# pylint: disable=logging-format-interpolation

import logging

import gi
gi.require_version('Gtk', '3.0')
from gi.repository import Gtk, Pango, Gdk
from matplotlib.backends.backend_gtk3agg import (
    FigureCanvasGTK3Agg as FigureCanvas)
from matplotlib.backends.backend_gtk3 import (
    NavigationToolbar2GTK3 as NavigationToolbar)
from matplotlib.figure import Figure
import numpy as np

from xpl import __colors__
from xpl.view import SPECTRUM_TITLES, EXCLUDING_KEY, PEAK_TITLES
from xpl.canvas_selection import SpanSelector, PeakSelector


logger = logging.getLogger(__name__)

class XPLSpectrumTreeStore(Gtk.TreeStore):
    """A TreeModel that has methods for filling with spectrum info.
    ID is always the first attribute in self.titles and the last element of
    each row is True or False for plotting!"""
    __gtype_name__ = "XPLSpectrumTreeStore"
    def __init__(self):
        self.titles = SPECTRUM_TITLES
        assert list(self.titles.keys())[0] == "ID"
        types = [str] * len(self.titles) + [bool]
        super().__init__(*types)

    #pylint: disable=arguments-differ
    def append(self, rowdict):
        """Adds values from rowdict to the Store. rowdict has to contain
        values for all keys in the XPLTreeModel.titles dict."""
        try:
            values = [str(rowdict[attr]) for attr in self.titles.keys()]
            values += [rowdict["active"]]
        except KeyError:
            logger.error("XPLSpectrumTreeStore does not contain all necessary"
                         "keys for adding: {}".format(rowdict))
            raise
        super().append(parent=None, row=values)

    def get_col_index(self, attr):
        """Returns column index corresponing to attribute attr."""
        try:
            index = list(self.titles.keys()).index(attr)
            return index
        except ValueError:
            if attr == "active":
                return len(self.titles)
            logger.error("column with name {} is not in the"
                         "XPLSpectrumTreeStore".format(attr))
            raise

    def get_iter_list(self):
        """Returns list of all iters in this model."""
        iterlist = []
        def iterate_level(iter_):
            """Iterate over all children of iter_ and add their iter to
            iterlist."""
            while iter_ is not None:
                iterlist.append(iter_)
                if self.iter_has_child(iter_):
                    iterate_level(iter_)
                iter_ = self.iter_next(iter_)
        root = self.get_iter_first()
        iterate_level(root)
        return iterlist


class XPLPeakTreeStore(Gtk.TreeStore):
    """A TreeModel that has methods for filling with spectrum info.
    ID is always the first attribute in self.titles."""
    __gtype_name__ = "XPLPeakTreeStore"
    def __init__(self):
        self.titles = PEAK_TITLES
        assert list(self.titles.keys())[0] == "ID"
        self.region = None
        types = [str] * len(self.titles)
        super().__init__(*types)

    #pylint: disable=arguments-differ
    def append(self, rowdict):
        """Adds values from rowdict to the Store. rowdict has to contain
        values for all keys in the XPLTreeModel.titles dict."""
        try:
            values = [str(rowdict[attr]) for attr in self.titles.keys()]
        except KeyError:
            logger.error("XPLPeakTreeStore does not contain all necessary"
                         "keys for adding: {}".format(rowdict))
            raise
        super().append(parent=None, row=values)

    def get_col_index(self, attr):
        """Returns column index corresponing to attribute attr."""
        try:
            index = list(self.titles.keys()).index(attr)
            return index
        except ValueError:
            if attr == "active":
                return len(self.titles)
            logger.error("column with name {} is not in the"
                         "XPLPeakTreeStore".format(attr))
            raise

    def get_iter_list(self):
        """Returns list of all iters in this model."""
        iterlist = []
        def iterate_level(iter_):
            """Iterate over all children of iter_ and add their iter to
            iterlist."""
            while iter_ is not None:
                iterlist.append(iter_)
                if self.iter_has_child(iter_):
                    iterate_level(iter_)
                iter_ = self.iter_next(iter_)
        root = self.get_iter_first()
        iterate_level(root)
        return iterlist


class EditSpectrumDialogAttributeBox(Gtk.Box):
    """Glade Template to get a box with a label and an entry for the
    dialog where a spectrum can be edited."""
    def __init__(self, labeltext, entrytext):
        super().__init__(
            margin_top=5,
            margin_bottom=5
        )
        self.label = Gtk.Label(
            label=labeltext,
            width_chars=25,
            max_width_chars=25,
            wrap=True,
            wrap_mode=Pango.WrapMode.WORD,
            justify=Gtk.Justification.CENTER
        )
        self.entry = Gtk.Entry(
            text=entrytext,
            margin_left=10,
            margin_right=10
        )
        self.pack_start(self.label, False, False, 0)
        self.pack_start(self.entry, True, True, 0)
        self.show_all()


class XPLEditSpectrumDialog(Gtk.Dialog):
    """A dialog for editing spectrum metadata."""
    __gtype_name__ = "XPLEditSpectrumDialog"
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.excluding_key = EXCLUDING_KEY
        self.box = self.get_content_area()
        self.newdict_entries = {}

    def populate_filename(self, fnames):
        """Changes the filename label to fnames."""
        flabel = self.box.get_children()[0].get_children()[1]
        flabel.set_text(fnames)

    def flush(self):
        """Removes all EditSpectrumDialogAttributeBox."""
        for child in self.box.get_children():
            if isinstance(child, EditSpectrumDialogAttributeBox):
                self.box.remove(child)
        self.newdict_entries.clear()

    def add_attribute(self, attr, title, value):
        """Adds an EditSpectrumDialogAttributeBox."""
        attr_box = EditSpectrumDialogAttributeBox(title, value)
        self.box.pack_start(attr_box, False, False, 0)
        self.newdict_entries[attr] = attr_box.entry

    def get_newdict(self):
        """Returns the newdict (attr, new_value) with new_value from
        the entries."""
        newdict = dict(
            (attr, self.newdict_entries[attr].get_text())
            for attr in self.newdict_entries
            if self.excluding_key not in self.newdict_entries[attr].get_text()
        )
        return newdict


class XPLFigure(Figure):
    """A figure object for the main canvas."""
    def __init__(self):
        super().__init__(figsize=(10, 10), dpi=80)
        self.ax = self.add_axes([-0.005, 0.05, 1.01, 1.005])
        self.ax.set_facecolor("#3A3A3A")
        self.patch.set_facecolor("#3A3A3A")

        self._xy_buffer = {"default": [0, 1, 0, 1]}
        self._xy = [np.inf, -np.inf, np.inf, -np.inf]

    def reset_xy_centerlims(self):
        """Sets self._xy to infinite values again, so it can be incrementally
        updated by update_xy."""
        self._xy = [np.inf, -np.inf, np.inf, -np.inf]

    def update_xy_centerlims(self, xmin, xmax, ymin, ymax):
        """Updates self._xy where min values are only assumed when they
        are lower than the current min. Analogous for max values."""
        self._xy = [
            min(self._xy[0], xmin),
            max(self._xy[1], xmax),
            min(self._xy[2], ymin),
            max(self._xy[3], ymax)
        ]

    def get_ymax(self):
        """Returns the current value of self._xy[3]."""
        return self._xy[3]

    def store_xylims(self, keyword="default"):
        """Stores axis limits in self._xy_buffer."""
        xmin = min(self.ax.get_xlim())
        xmax = max(self.ax.get_xlim())
        ymin = min(self.ax.get_ylim())
        ymax = max(self.ax.get_ylim())
        self._xy_buffer[keyword] = [xmin, xmax, ymin, ymax]
        # self._xy_buffer[1], self._xy_buffer[0] = self.ax.get_xlim()
        # self._xy_buffer[2], self._xy_buffer[3] = self.ax.get_ylim()

    def restore_xylims(self, keyword="default"):
        """Sets the axis limits."""
        if keyword not in self._xy_buffer:
            raise KeyError("key {} not in figure xy buffer".format(keyword))
        if np.all(np.isfinite(self._xy_buffer[keyword])):
            xmin, xmax, ymin, ymax = self._xy_buffer[keyword]
            if np.all(np.isfinite(self._xy)) and (
                    ymin > self._xy[3] or ymax < self._xy[2]):
                self.center_view()
                return
            self.ax.set_xlim(xmax, xmin)
            self.ax.set_ylim(ymin, ymax)
            self.set_ticks()
        else:
            self.restore_xylims()

    def isstored(self, keyword):
        """Returns if the keyword is already known."""
        return keyword in self._xy_buffer

    def center_view(self, keyword="default"):
        """Focuses view on current plot."""
        if np.all(np.isfinite(self._xy)):
            bordered_center = [
                self._xy[0] - (self._xy[1] - self._xy[0]) * 0.02,
                self._xy[1] + (self._xy[1] - self._xy[0]) * 0.02,
                0,
                self._xy[3] * 1.1
            ]
            self._xy_buffer[keyword] = bordered_center
        else:
            logger.warning("xylims not properly adjusted")
        self.restore_xylims(keyword)

    def set_ticks(self):
        """Configures axes ticks."""
        self.ax.spines["bottom"].set_visible(False)
        self.ax.tick_params(
            reset=True,
            axis="both",
            direction="out",
            # pad=-20,
            labelsize="large",
            labelcolor=__colors__.get("plotting", "axisticks"),
            color=__colors__.get("plotting", "axisticks"),
            labelleft=False,
            top=False,
            left=False,
            right=False,
            bottom=False
        )
        if self._xy[0] == np.inf:
            self.ax.tick_params(
                which="both",
                bottom=False,
                top=False,
                left=False,
                right=False,
                labelbottom=False
            )


class XPLCanvas(FigureCanvas):
    """Canvas for plotting spectra with matplotlib."""
    __gtype_name__ = "XPLCanvas"
    def __init__(self):
        figure = XPLFigure()
        super().__init__(figure)


# pylint: disable=too-few-public-methods
class Cursors(object):
    """Simple namespace for cursor reference"""
    HAND, POINTER, SELECT_REGION, MOVE, WAIT, DRAG = list(range(6))


# pylint: disable=too-many-instance-attributes
class XPLPlotToolbar(NavigationToolbar, Gtk.Toolbar):
    """Navbar for the canvas."""
    __gtype_name__ = "XPLPlotToolbar"
    # pylint: disable=super-init-not-called
    # pylint: disable=attribute-defined-outside-init
    def __init__(self, *args, **kwargs):
        Gtk.Toolbar.__init__(self, *args, **kwargs)
        self.cursors = Cursors()
        self.cursord = {
            self.cursors.MOVE: Gdk.Cursor.new(Gdk.CursorType.FLEUR),
            self.cursors.HAND: Gdk.Cursor.new(Gdk.CursorType.HAND2),
            self.cursors.POINTER: Gdk.Cursor.new(Gdk.CursorType.LEFT_PTR),
            self.cursors.SELECT_REGION: Gdk.Cursor.new(Gdk.CursorType.TCROSS),
            self.cursors.WAIT: Gdk.Cursor.new(Gdk.CursorType.WATCH),
            self.cursors.DRAG: Gdk.Cursor.new(Gdk.CursorType.SB_H_DOUBLE_ARROW)
        }

    def reinit(self, canvas, messagelabel, coordlabel):
        """Call the real super __init__ after the canvas is known and give
        it to this function."""
        self.message = messagelabel
        self.mode_label = messagelabel
        self.xy_label = coordlabel
        super().__init__(canvas, None)

        self.span_selector = SpanSelector(
            self.canvas.figure.ax,
            lambda *args: None,
            "horizontal",
            minspan=0.2,
            span_stays=False,
            useblit=True
        )
        self.span_selector.active = False
        self.peak_selector = PeakSelector(
            self.canvas.figure.ax,
            lambda *args: None,
            peak_stays=False,
            useblit=True
        )
        self.peak_selector.active = False

    def _init_toolbar(self):
        """Normally, this would create the buttons and connect them to
        the tools, but now GtkBuilder does the job and the connections
        are done in the Gtk.Application. This function is automatically
        called during NavigationToolbar.__init__"""

    def disable_tools(self):
        """Release widgetlock and disconnect all signals associated with
        native matplotlib toolbar tools."""
        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)

        self._active = None
        self.mode = ""
        self.release_all_tools()
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self._active)
        self.set_message(self.mode)

    def get_span(self, callback, **kwargs):
        """Gets a span and then calls callback(min, max). Also takes care
        of widgetlock and such."""
        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ""
        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ""

        mode = callback.__doc__

        self.release_all_tools()
        if self._active == mode:
            self._active = None
            self.mode = ""
            self.span_selector.active = False
            self.set_message(self.mode)
            return

        self._active = mode
        self.mode = mode
        self.canvas.widgetlock(self.span_selector)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(None)
        self.set_message(self.mode)

        def on_selected(emin, emax):
            """Callback caller."""
            self._active = None
            self.mode = ""
            self.span_selector.active = False
            self.release_all_tools()
            self.set_message(self.mode)
            callback(emin, emax)
        rectprops = {
            "alpha": 1,
            "fill": False,
            "edgecolor": "black",
            "linewidth": 1,
            "linestyle": "-"
        }
        rectprops.update(kwargs)
        self.span_selector.set_rectprops(rectprops)
        self.span_selector.onselect = on_selected
        self.span_selector.active = True

    def get_wedge(self, callback, **kwargs):
        """Gets a wegde and then calls callback(center, height, fwhm).
        Also takes care of widgetlock and such."""
        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ""
        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ""

        mode = callback.__doc__

        self.release_all_tools()
        if self._active == mode:
            self._active = None
            self.mode = ""
            self.peak_selector.active = False
            self.set_message(self.mode)
            return

        self._active = mode
        self.mode = mode
        self.canvas.widgetlock(self.peak_selector)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(None)
        self.set_message(self.mode)

        def on_selected(center, height, angle):
            """Callback caller."""
            self._active = None
            self.mode = ""
            self.peak_selector.active = False
            self.release_all_tools()
            self.set_message(self.mode)
            callback(center, height, angle)
        wedgeprops = {
            "alpha": 0.5,
            "fill": True,
            "edgecolor": "black",
            "linewidth": 1,
            "linestyle": "-"
        }
        if kwargs["limits"]:
            self.peak_selector.set_limits(kwargs.pop("limits"))
        else:
            self.peak_selector.set_limits((-np.inf, np.inf))
        wedgeprops.update(kwargs)
        self.peak_selector.set_wedgeprops(wedgeprops)
        self.peak_selector.onselect = on_selected
        self.peak_selector.active = True

    def pan(self, *_ignore):
        """Activate the pan/zoom tool. pan with left button, zoom with right
        OVERWRITE because of widgetlock release."""
        # set the pointer icon and button press funcs to the
        # appropriate callbacks
        if self._active == 'PAN':
            self._active = None
        else:
            self._active = 'PAN'
        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''
        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''
        self.release_all_tools()        # changed here
        if self._active:
            self._idPress = self.canvas.mpl_connect(
                'button_press_event', self.press_pan)
            self._idRelease = self.canvas.mpl_connect(
                'button_release_event', self.release_pan)
            self.mode = 'Pan / Zoom'
            self.canvas.widgetlock(self)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self._active)
        self.set_message(self.mode)

    def zoom(self, *args):
        """Activate zoom to rect mode.
        OVERWRITE because of widgetlock release."""
        if self._active == 'ZOOM':
            self._active = None
        else:
            self._active = 'ZOOM'
        if self._idPress is not None:
            self._idPress = self.canvas.mpl_disconnect(self._idPress)
            self.mode = ''
        if self._idRelease is not None:
            self._idRelease = self.canvas.mpl_disconnect(self._idRelease)
            self.mode = ''
        self.release_all_tools()        # changed here
        if self._active:
            self._idPress = self.canvas.mpl_connect(
                'button_press_event', self.press_zoom)
            self._idRelease = self.canvas.mpl_connect(
                'button_release_event', self.release_zoom)
            self.mode = 'Zoom Rectangle'
            self.canvas.widgetlock(self)
        for a in self.canvas.figure.get_axes():
            a.set_navigate_mode(self._active)
        self.set_message(self.mode)

    def center(self):
        """Centers view and disables navbar tools."""
        self.disable_tools()
        self.canvas.figure.center_view()
        self.canvas.draw_idle()

    def release_all_tools(self):
        """Release all tools used in this class from self.canvas.widgetlock
        because the locks have different owners."""
        try:
            self.canvas.widgetlock.release(self)
        except ValueError:
            try:
                self.canvas.widgetlock.release(self.span_selector)
            except ValueError:
                self.canvas.widgetlock.release(self.peak_selector)

    def mouse_move(self, event):
        self._set_cursor(event)
        if event.inaxes and event.inaxes.get_navigate():
            try:
                s = event.inaxes.format_coord(event.xdata, event.ydata)
                self.xy_label.set_markup("<tt>{}</tt>".format(s))
            except (ValueError, OverflowError):
                pass
        if self.mode:
            self.mode_label.set_text("{}".format(self.mode))

    def set_cursor(self, cursor):
        self.canvas.get_property("window").set_cursor(self.cursord[cursor])
        Gtk.main_iteration()

    def _set_cursor(self, event):
        if not event.inaxes or not self._active:
            if self._lastCursor != self.cursors.POINTER:
                self.set_cursor(self.cursors.POINTER)
                self._lastCursor = self.cursors.POINTER
        else:
            if (self._active == 'ZOOM'
                    and self._lastCursor != self.cursors.SELECT_REGION):
                self.set_cursor(self.cursors.SELECT_REGION)
                self._lastCursor = self.cursors.SELECT_REGION
            elif (self._active == 'PAN' and
                  self._lastCursor != self.cursors.MOVE):
                self.set_cursor(self.cursors.MOVE)
                self._lastCursor = self.cursors.MOVE
            elif (self._active == 'Add region' and
                  self._lastCursor != self.cursors.DRAG):
                self.set_cursor(self.cursors.DRAG)
                self._lastCursor = self.cursors.DRAG
            elif self._active == "Add peak":
                lims = self.peak_selector.limits
                if lims[0] < event.xdata < lims[1]:
                    if self._lastCursor != self.cursors.SELECT_REGION:
                        self.set_cursor(self.cursors.SELECT_REGION)
                        self._lastCursor = self.cursors.SELECT_REGION
                else:
                    if self._lastCursor != self.cursors.POINTER:
                        self.set_cursor(self.cursors.POINTER)
                        self._lastCursor = self.cursors.POINTER
