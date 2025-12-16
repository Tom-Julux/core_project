import json
from pathlib import Path

import threading
import time
import torch
import os
import cv2
from magicgui import magicgui
from napari.layers import Image
from typing import TYPE_CHECKING
from functools import partial
import numpy as np
from napari.utils.colormaps import CyclicLabelColormap, DirectLabelColormap, label_colormap
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
    setup_lineedit,
    setup_fileselect,
    setup_savefileselect,
    setup_plaintextedit,
    setup_labeledslider,
    setup_pushbutton,setup_lineedit,
    setup_hswitch,
    setup_radiobutton,setup_textedit,
    setup_savefileselect,
    setup_doublespinbox,
)
from .file_select import setup_dirselect
from .layer_select import setup_layerselect
from napari.utils.action_manager import action_manager
from napari.utils.events.event import WarningEmitter
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QFileDialog,
    QSizePolicy,
    QVBoxLayout,
    QWidget
)
from qtpy.QtGui import (
    QTextOption
)
import traceback
import glob
from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls

from napari.utils.notifications import show_info, show_warning, show_error, show_console_notification
from napari import Viewer
import SimpleITK as sitk
from scipy.interpolate import interpn



class SizeEstimatorWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        # Label select -> Labels layer

        # Size variables x,y,z

        # Output layer
        # Volume, Surface area, Mean Diameter, etc.
        self.current_layer = None
        self.silence_events = False
        self.is_running = False
        self.should_run = False

        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)
        _scroll_layout.setContentsMargins(0,0,0,0)
        _scroll_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        _ = setup_label(
            _scroll_layout, "Select label layer:")
        # layer select for image layer
        self.layerselect_a = setup_layerselect(
            _scroll_layout, self._viewer, Labels, function=lambda: self.on_layer_change()
        )
  
        _container, _layout = setup_vcollapsiblegroupbox(
            _scroll_layout, "Resolution:", collapsed=False)

        _resolution_label_x = setup_label(
            None, "Resolution in x:")
        self.resolution_x_spinbox = setup_doublespinbox(
            None,
            default=1.0,
            minimum=0.01,
            maximum=1000.0,
            step_size=0.1,
            )
        _ = hstack(_layout, [_resolution_label_x, self.resolution_x_spinbox], stretch=[0, 1])

        _resolution_label_y = setup_label(
            None, "Resolution in y:")
        self.resolution_y_spinbox = setup_doublespinbox(
            None,
            default=1.0,
            minimum=0.01,
            maximum=1000.0,
            step_size=0.1,
            )
        _ = hstack(_layout, [_resolution_label_y, self.resolution_y_spinbox], stretch=[0, 1])

        _resolution_label_z = setup_label(
            None, "Resolution in z:")
        self.resolution_z_spinbox = setup_doublespinbox(
            None,
            default=1.0,
            minimum=0.01,
            maximum=1000.0,
            step_size=0.1,
            )
        _ = hstack(_layout, [_resolution_label_z, self.resolution_z_spinbox], stretch=[0, 1])

        self.run_button = setup_iconbutton(
            _scroll_layout,
            "Run",
            "right_arrow",
            self._viewer.theme,
            function=lambda: self.run_size_estimation(),
        )
        
        _ = setup_label(
            _scroll_layout, "Output:")
              
        self.text_output = setup_plaintextedit(
            _scroll_layout,
            "",
            readonly=True,
        )

        #self.autorun_checkbox = setup_checkbox(
        #    _scroll_layout,
        #    "Auto Run",
        #    False)
        
        self.setup_connections()

    def run_size_estimation(self):
        label_layer, img_layer_idx = get_value(self.layerselect_a)
        if label_layer is None or img_layer_idx == -1 or label_layer not in self._viewer.layers:
            show_warning("Please select a valid label layer.")
            return
        label_layer = self._viewer.layers[label_layer]

        # get resolution
        res_x = self.resolution_x_spinbox.value()
        res_y = self.resolution_y_spinbox.value()
        res_z = self.resolution_z_spinbox.value()
        res = np.array((res_z, res_y, res_x))  # z,y,x order

        output = {}
        # iterate over labels
        for label in np.unique(label_layer.data):
            if label == 0:
                continue  # skip background

            # create binary mask for current label
            binary_mask = (label_layer.data == label).astype(np.uint8)

            # compute volume
            voxel_volume = np.prod(res)
            label_volume = np.sum(binary_mask) * voxel_volume
            output[str(label)] = float(label_volume)
        
        self.text_output.setPlainText(
            json.dumps(output, indent=4)
        )
    
    def on_layer_change(self):
        pass
        """
        Connect transform events (scale/rotate/translate) of the selected
        image layer to preview and prompt layers so they follow image
        transformations.
        """
        if self.current_layer is not None:
            self.current_layer.events.set_data.disconnect(self._on_set_data)
            self.current_layer.events.labels_update.disconnect(self._on_labels_update)
            self.current_layer = None

        label_layer, img_layer_idx = get_value(self.layerselect_a)

        if label_layer not in self._viewer.layers:
            return
        
        self.setup_connections()

    def setup_connections(self):
        label_layer, img_layer_idx = get_value(self.layerselect_a)
        if label_layer is None or img_layer_idx == -1 or label_layer not in self._viewer.layers:
            return
        label_layer = self._viewer.layers[label_layer]
        self.current_layer = label_layer

        self.current_layer.events.set_data.connect(self._on_set_data)
        self.current_layer.events.labels_update.connect(self._on_labels_update)
    
    def _on_set_data(self, event):
        """sync data modification from additional viewers"""
        # Ignore in-progress events for performance reasons
        if hasattr(event, 'action') and event.action in ['adding', 'removing', 'changing']:
            return

        if self.silence_events:
            return

    def _on_labels_update(self, event):
        """sync data modification from additional viewers"""
        # Ignore in-progress events for performance reasons
        if hasattr(event, 'action') and event.action in ['adding', 'removing', 'changing']:
            return

        if self.silence_events:
            return
        
        #if self.is_running:
        #    self.should_run = True
        #else:
        self.run_()

        # add current frame to manual frames

    # Setup / Teardown events
    def showEvent(self, event):
        pass
    
    def closeEvent(self, event):
        self.hideEvent(event)

    def hideEvent(self, event):
        if self.current_layer is None:
            return
        self.current_layer.events.set_data.disconnect(self._on_set_data)
        self.current_layer.events.labels_update.disconnect(self._on_labels_update)
        self.current_layer = None


class FileListWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)
        _scroll_layout.setContentsMargins(0,0,0,0)
        _scroll_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.file_list_widget = FileListWidget2(self._viewer, _scroll_layout, on_file_selected=self.on_file_selected)

        self.sam2d_widget = None
        self.size_estimator_widget = None
        self.image_layer = None
    
    def on_file_selected(self, file_path):
        print("Selected file:", file_path)

        # Remove existing layers to avoid confusion
        for layer in self._viewer.layers:
            self._viewer.layers.remove(layer)

        if self.image_layer is not None:
            self.image_layer = None

        if self.sam2d_widget is not None:
            self._viewer.window.remove_dock_widget(self.sam2d_widget)
            self.sam2d_widget.close()
        from napari_interactive_sam2._widget_2d_sam import InteractiveSegmentationWidget2DSAM
        widget = InteractiveSegmentationWidget2DSAM(self._viewer, hide_model_setup=True, hide_prompt_type_select=True, hide_prompt_import=True, hide_export=True)
        self._viewer.window.add_dock_widget(
            widget, name="Interactive Segmentation", area="right"
        )
        self.sam2d_widget = widget

        if self.size_estimator_widget is not None:
            self._viewer.window.remove_dock_widget(self.size_estimator_widget)
            self.size_estimator_widget.close()
        size_widget = SizeEstimatorWidget(self._viewer)
        self._viewer.window.add_dock_widget(
            size_widget, name="Size Estimator", area="right"
        )
        self.size_estimator_widget = size_widget

        import SimpleITK as sitk
        base_dir = self.file_list_widget.import_dir_select.get_dir()
        img_sitk = sitk.ReadImage(
            os.path.join(base_dir, file_path)
        )

        img = sitk.GetArrayFromImage(img_sitk)
        img = img = img[img.shape[0]//2-16:img.shape[0]//2+32]
        self.image_layer = self._viewer.add_image(
            img,
            name='Example Image',
            colormap='gray'
        )
        resolution = img_sitk.GetSpacing()  # x,y,z
        print("Image resolution:", resolution)

        self.size_estimator_widget.resolution_x_spinbox.setValue(resolution[0])
        self.size_estimator_widget.resolution_y_spinbox.setValue(resolution[1])
        self.size_estimator_widget.resolution_z_spinbox.setValue(resolution[2])


    def showEvent(self, event):
        pass
    
    def closeEvent(self, event):
        self.hideEvent(event)

    def hideEvent(self, event):
        pass

class FileListWidget2(QWidget):
    def __init__(self, viewer, parent_layout, on_file_selected=lambda x: None):
        super().__init__()
        self._viewer = viewer
        self.on_file_selected = on_file_selected
        # File input
        _scroll_layout = parent_layout

        _container, _layout = setup_vcollapsiblegroupbox(
            _scroll_layout, "Folder selection:", collapsed=False)
        _ = setup_label(
            _layout, "Select a folder from which to ingress files.")

        self.import_dir_select = setup_dirselect(
            _layout,
            "Folder:",
            function=lambda: print("QDirSelect")
        )

        _ = setup_label(_layout, "Optionally provide a glob pattern: (relative to folder)")

        self.import_glob = setup_lineedit(_layout, "**/*.mha", "**/*.mha", function=lambda: print("QTextEdit"))

        _ = setup_iconbutton(
            _layout, "Load glob", "right_arrow", self._viewer.theme, self.load_file_list
        )
        self.count_label = setup_label(_layout, "No files loaded.")
        self.count_label.setVisible(False)

        self.select_file_container, _layout = setup_vcollapsiblegroupbox(
            _scroll_layout, "File selection:", collapsed=False)
        self.select_file_container.setVisible(False)
        # select qbox
        _ = setup_label(
            _layout, "Select a file:")
    
        self.file_select = setup_combobox(_layout, [], "No files available", function=lambda: None)
        
        #self.file_select =setup_list(_layout, [], "No files available", function=lambda: None)
        # buttons for cycling through labels

        # load button
        hstack(_layout, [ 
            setup_iconbutton(
                _layout, "Load", "right_arrow", self._viewer.theme, self.load_selected_file
            ),
            setup_iconbutton(
                _layout, "Load Next", "right_arrow", self._viewer.theme, self.load_next_file
            )], stretch=[0,0])


    def load_file_list(self):
        base_dir = self.import_dir_select.get_dir()
        glob_pattern = self.import_glob.text()

        file_list = glob.glob(os.path.join(base_dir, glob_pattern), recursive=True)
        file_list = sorted(file_list)
        file_list = [os.path.relpath(f, base_dir) for f in file_list]
        print(file_list)
        self.file_select.clear()
        self.file_select.addItems(file_list)
        self.file_select.setCurrentIndex(0)
        if len(file_list) == 0:
            show_warning("No files found.")
            self.count_label.setText("No files found.")
            self.count_label.setVisible(True)
            self.select_file_container.setVisible(False)
        else:
            self.count_label.setText(f"{len(file_list)} files found.")
            self.count_label.setVisible(True)
            self.select_file_container.setVisible(True)

    def load_selected_file(self):
        show_info(f"Loading file:")
        base_dir = self.import_dir_select.get_dir()
        selected_file = self.file_select.currentText()
        full_path = os.path.join(base_dir, selected_file)
        print(full_path)
        self.on_file_selected(full_path)

    def load_next_file(self):
        current_index = self.file_select.currentIndex()
        if current_index + 1 < self.file_select.count():
            self.file_select.setCurrentIndex(current_index + 1)
            self.load_selected_file()
        else:
            show_info("No more files to load.")


    # Setup / Teardown events
    def showEvent(self, event):
        pass
    
    def closeEvent(self, event):
        self.hideEvent(event)

    def hideEvent(self, event):
        pass