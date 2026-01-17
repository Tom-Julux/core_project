import json
from pathlib import Path

import threading
import time
import torch
import os
import cv2
from magicgui import magicgui
from napari.layers import Image
import numpy as np
from napari.utils.notifications import show_info, show_warning, show_error, show_console_notification
from napari import Viewer
from napari.layers import Labels, Shapes, Points, Image, Layer
from napari_toolkit.containers import setup_scrollarea, setup_vcollapsiblegroupbox, setup_vgroupbox, setup_vscrollarea
from napari_toolkit.containers.boxlayout import hstack
from napari_toolkit.utils import set_value
from napari_toolkit.data_structs import setup_list
from napari_toolkit.utils.widget_getter import get_value
from napari_toolkit.widgets import (
    setup_checkbox,
    setup_combobox,
    setup_editcolorpicker,
    setup_editdoubleslider,
    setup_iconbutton,
    setup_label,
    setup_layerselect,
    setup_lineedit,
    setup_fileselect,
    setup_savefileselect,
    setup_labeledslider,
    setup_pushbutton,
    setup_hswitch,
    setup_radiobutton,
    setup_savefileselect,
    setup_dirselect,
    setup_spinbox,
)
from qtpy.QtWidgets import (
    QVBoxLayout,
    QWidget,
)
from napari.qt.threading import thread_worker

import traceback

from napari_promptable import BaseWidget3D


class ViewSwitchWidget(QWidget):
    def __init__(self, viewer:Viewer):
        super().__init__()

        self._viewer = viewer

        self._silence = False

        main_layout = QVBoxLayout(self)
        _scroll_layout = main_layout
        #_scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)
        setup_label(_scroll_layout, "Switch to different view:")
        self.view_select = setup_hswitch(
            _layout, ["Axial", "Saggital", "Coronal", "Other"], default=0, function=lambda: self.set_view())

        self._viewer.dims.events.order.connect(self.on_dims_event)

    def set_view(self):
        """Switch the viewer dimension ordering based on the selected view.

        This method updates ``self._viewer.dims.order`` so that the napari
        display presents the chosen orthogonal slice ordering. If a caller
        wants to automatically set prompts on view change, they can call
        ``set_current_view_prompt`` (optionally controlled by a checkbox).
        """

        if self._silence:
            return

        # Set the current view based on the selected option in the view_select widget
        selected_view = get_value(self.view_select)[1]
        print(f"Selected View: {selected_view}")

        current_view = self._viewer.dims.order[0]
        print(f"Current Order: {current_view}")

        if selected_view == 0:
            # Set the order of dimensions to A
            self._viewer.dims.order = (0, 1, 2)
        elif selected_view == 1:
            self._viewer.dims.order = (1, 0, 2)
        elif selected_view == 2:
            self._viewer.dims.order = (2, 0, 1)

    def on_dims_event(self, event):
        self._silence = True

        order = self._viewer.dims.order
        if order == (0, 1, 2):
            set_value(self.view_select, 0)
        elif order == (1, 0, 2):
            set_value(self.view_select, 1)
        elif order == (2, 0, 1):
            set_value(self.view_select, 2)
        else:
            set_value(self.view_select, 4)

        self._silence = False

    def closeEvent(self, event):
        self._viewer.dims.events.order.disconnect(self.on_dims_event)

    def hideEvent(self, event):
        self._viewer.dims.events.order.disconnect(self.on_dims_event)

    def showEvent(self, event):
        self._viewer.dims.events.order.connect(self.on_dims_event)
