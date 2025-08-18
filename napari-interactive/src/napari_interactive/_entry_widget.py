import json
from pathlib import Path

import threading
import time
import cv2
from magicgui import magicgui
from napari.layers import Image
from typing import TYPE_CHECKING
from functools import partial
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
    setup_labeledslider,
    setup_pushbutton,
    setup_radiobutton,
    setup_savefileselect,
    setup_dirselect,
    setup_spinbox,
)
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QFileDialog,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

class InteractiveSegmentationWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        self.build_gui()

    # GUI
    def build_gui(self):
        """
        """
        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)

        self._scroll_layout = _scroll_layout
        # log view
        _container, _layout = setup_vgroupbox(_scroll_layout, "Setup")

        #_container, _layout = setup_vgroupbox(_scroll_layout, "Metric mode:")
        _ = setup_label(_layout, "Select Mode:")
        self.mode_select = setup_combobox(
            _layout, ["2D", "2D+t", "2D cross layer (WIP)", "3D (WIP)", "3D+t (WIP)", "3D cross layer (WIP)"],\
                 "QComboBox", function=lambda: print("QComboBox")
        )

        _ = setup_label(_layout, "Select Model:")
        self.model_select = setup_combobox(
            _layout, ["SAM2", "nnInteractive (WIP)"],\
                 "QComboBox", function=lambda: print("QComboBox")
        )

        _ = setup_iconbutton(
            _layout, "Load Model", "new_labels", self._viewer.theme, self.load_model
        )
        

        _ = setup_label(_scroll_layout, "This plugin is a work in progress and may not work as expected. Please report any issues on the GitHub repository.")



        _ = setup_label(_scroll_layout, "It is designed to work work in a mutlidiue.")

        _ = setup_label(_scroll_layout, "This entrypoint widget can be used to load the different widgets for different models and modes.")

        _ = setup_label(_scroll_layout, """The following Modes are available:
            - 2D: Single 2D image segmentation
            - 2D+t: 2D image segmentation with time dimension (WIP)
            - 2D cross layer: 2D image segmentation across multiple layers (WIP)
            - 3D: 3D image segmentation (WIP)
            - 3D+t: 3D image segmentation with time dimension (WIP)
            - 3D cross layer: 3D image segmentation across multiple layers (WIP)

            Alternatively, these plugins can also be loaded directly from the menubar.""".replace("\n            ", "\n"))

        _ = setup_label(_scroll_layout, "Plugin by LMU")
        
        #self.load_model()
        
    def load_model(self):
        mode = get_value(self.mode_select)[0]
        model = get_value(self.model_select)[0]

        print(f"Loading model {model} for mode {mode}...")
        # load the appropriate widget based on the selected model and mode
        if model == "SAM2" and mode == "2D":
            self._viewer.window.add_plugin_dock_widget(
                "napari-interactive",
                "SAM2 2D",
            )
        elif model == "SAM2" and mode == "2D+t":
            self._viewer.window.add_plugin_dock_widget(
                "napari-interactive",
                "SAM2 2D",
            )
        # close self
        self._viewer.window.remove_dock_widget(self)
