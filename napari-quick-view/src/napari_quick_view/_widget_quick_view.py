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
from napari_toolkit.widgets import *
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
from ._widget_file_list import FileListWidget

class QuickViewWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)
        _scroll_layout.setContentsMargins(0,0,0,0)
        _scroll_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.file_list_widget = FileListWidget(self._viewer, _scroll_layout, on_file_selected=self.on_file_selected)

        container, _layout = setup_vcollapsiblegroupbox(
            _scroll_layout, "Instructions:", collapsed=True)
        
        # select qbox
        _ = setup_label(
            _layout, "Select a transpoing:")
        self.transpose_selection = setup_combobox(
            _layout, ["0,1,2", "2,1,0", "2,0,1", "1,2,0", "0,2,1"], "0,1,2",)

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

        import SimpleITK as sitk
        base_dir = self.file_list_widget.import_dir_select.get_dir()
        img_sitk = sitk.ReadImage(
            os.path.join(base_dir, file_path)
        )

        img = sitk.GetArrayFromImage(img_sitk)

        #img = np.transpose(img, transpose_order)

        self.image_layer = self._viewer.add_image(
            img,
            name='Example Image',
            colormap='gray'
        )

        resolution = img_sitk.GetSpacing()  # x,y,z
        transpose_order = tuple(int(i) for i in self.transpose_selection.currentText().split(","))

        self._viewer.dims.order = transpose_order

        print("Image resolution:", resolution)

    def showEvent(self, event):
        pass
    
    def closeEvent(self, event):
        self.hideEvent(event)

    def hideEvent(self, event):
        pass


class QuickSizeWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)
        _scroll_layout.setContentsMargins(0,0,0,0)
        _scroll_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.file_list_widget = FileListWidget(self._viewer, _scroll_layout, on_file_selected=self.on_file_selected)

        container, _layout = setup_vcollapsiblegroupbox(
            _scroll_layout, "Instructions:", collapsed=True)
        
        # select qbox
        _ = setup_label(
            _layout, "Select a transpoing:")
        self.transpose_selection = setup_combobox(
            _layout, ["0,1,2", "2,1,0", "2,0,1", "1,2,0", "0,2,1"], "0,1,2",)

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

        #if self.size_estimator_widget is not None:
        #    self._viewer.window.remove_dock_widget(self.size_estimator_widget)
        #    self.size_estimator_widget.close()
        #from napari_size_estimator import SizeEstimatorWidget
        #size_widget = SizeEstimatorWidget(self._viewer)
        #self._viewer.window.add_dock_widget(
        #    size_widget, name="Size Estimator", area="right"
        #)
        #self.size_estimator_widget = size_widget

        import SimpleITK as sitk
        base_dir = self.file_list_widget.import_dir_select.get_dir()
        img_sitk = sitk.ReadImage(
            os.path.join(base_dir, file_path)
        )

        img = sitk.GetArrayFromImage(img_sitk)

        self.image_layer = self._viewer.add_image(
            img,
            name='Example Image',
            colormap='gray'
        )
        resolution = img_sitk.GetSpacing()  # x,y,z
        print("Image resolution:", resolution)

        #self.size_estimator_widget.resolution_x_spinbox.setValue(resolution[0])
        #self.size_estimator_widget.resolution_y_spinbox.setValue(resolution[1])
        #self.size_estimator_widget.resolution_z_spinbox.setValue(resolution[2])

        transpose_order = tuple(int(i) for i in self.transpose_selection.currentText().split(","))
        self._viewer.dims.order = transpose_order
        
    def showEvent(self, event):
        pass
    
    def closeEvent(self, event):
        self.hideEvent(event)

    def hideEvent(self, event):
        pass


class QuickSize3DWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)
        _scroll_layout.setContentsMargins(0,0,0,0)
        _scroll_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.file_list_widget = FileListWidget(self._viewer, _scroll_layout, on_file_selected=self.on_file_selected)

        container, _layout = setup_vcollapsiblegroupbox(
            _scroll_layout, "Instructions:", collapsed=True)
        
        # select qbox
        _ = setup_label(
            _layout, "Select a transpoing:")
        self.transpose_selection = setup_combobox(
            _layout, ["0,1,2", "2,1,0", "2,0,1", "1,2,0", "0,2,1"], "0,1,2",)

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
        from napari_interactive_sam2._widget_3d_sam import InteractiveSegmentationWidget3DSAM
        widget = InteractiveSegmentationWidget3DSAM(self._viewer, hide_model_setup=True, hide_prompt_type_select=True, hide_prompt_import=True, hide_export=True)
        self._viewer.window.add_dock_widget(
            widget, name="Interactive Segmentation", area="right"
        )
        self.sam2d_widget = widget

        #if self.size_estimator_widget is not None:
        #    self._viewer.window.remove_dock_widget(self.size_estimator_widget)
        #    self.size_estimator_widget.close()
        #from napari_size_estimator import SizeEstimatorWidget
        #size_widget = SizeEstimatorWidget(self._viewer)
        #self._viewer.window.add_dock_widget(
        #    size_widget, name="Size Estimator", area="right"
        #)
        #self.size_estimator_widget = size_widget

        import SimpleITK as sitk
        base_dir = self.file_list_widget.import_dir_select.get_dir()
        img_sitk = sitk.ReadImage(
            os.path.join(base_dir, file_path)
        )

        img = sitk.GetArrayFromImage(img_sitk)

        self.image_layer = self._viewer.add_image(
            img,
            name='Example Image',
            colormap='gray'
        )
        resolution = img_sitk.GetSpacing()  # x,y,z
        print("Image resolution:", resolution)

        #self.size_estimator_widget.resolution_x_spinbox.setValue(resolution[0])
        #self.size_estimator_widget.resolution_y_spinbox.setValue(resolution[1])
        #self.size_estimator_widget.resolution_z_spinbox.setValue(resolution[2])

        transpose_order = tuple(int(i) for i in self.transpose_selection.currentText().split(","))
        self._viewer.dims.order = transpose_order
        
    def showEvent(self, event):
        pass
    
    def closeEvent(self, event):
        self.hideEvent(event)

    def hideEvent(self, event):
        pass


class QuickSize3DNNIWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)
        _scroll_layout.setContentsMargins(0,0,0,0)
        _scroll_widget.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)

        self.file_list_widget = FileListWidget(self._viewer, _scroll_layout, on_file_selected=self.on_file_selected)

        container, _layout = setup_vcollapsiblegroupbox(
            _scroll_layout, "Instructions:", collapsed=True)
        
        # select qbox
        _ = setup_label(
            _layout, "Select a transpoing:")
        self.transpose_selection = setup_combobox(
            _layout, ["0,1,2", "2,1,0", "2,0,1", "1,2,0", "0,2,1"], "0,1,2",)

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
        from napari_interactive_nni._widget_3d_nni import InteractiveSegmentationWidget3DNNI
        widget = InteractiveSegmentationWidget3DNNI(self._viewer, hide_model_setup=True, hide_prompt_type_select=True, hide_prompt_import=True, hide_export=True)
        self._viewer.window.add_dock_widget(
            widget, name="Interactive Segmentation", area="right"
        )
        self.sam2d_widget = widget

        #if self.size_estimator_widget is not None:
        #    self._viewer.window.remove_dock_widget(self.size_estimator_widget)
        #    self.size_estimator_widget.close()
        #from napari_size_estimator import SizeEstimatorWidget
        #size_widget = SizeEstimatorWidget(self._viewer)
        #self._viewer.window.add_dock_widget(
        #    size_widget, name="Size Estimator", area="right"
        #)
        #self.size_estimator_widget = size_widget

        import SimpleITK as sitk
        base_dir = self.file_list_widget.import_dir_select.get_dir()
        img_sitk = sitk.ReadImage(
            os.path.join(base_dir, file_path)
        )

        img = sitk.GetArrayFromImage(img_sitk)

        self.image_layer = self._viewer.add_image(
            img,
            name='Example Image',
            colormap='gray'
        )
        resolution = img_sitk.GetSpacing()  # x,y,z
        print("Image resolution:", resolution)

        #self.size_estimator_widget.resolution_x_spinbox.setValue(resolution[0])
        #self.size_estimator_widget.resolution_y_spinbox.setValue(resolution[1])
        #self.size_estimator_widget.resolution_z_spinbox.setValue(resolution[2])

        transpose_order = tuple(int(i) for i in self.transpose_selection.currentText().split(","))
        self._viewer.dims.order = transpose_order
        
    def showEvent(self, event):
        pass
    
    def closeEvent(self, event):
        self.hideEvent(event)

    def hideEvent(self, event):
        pass
