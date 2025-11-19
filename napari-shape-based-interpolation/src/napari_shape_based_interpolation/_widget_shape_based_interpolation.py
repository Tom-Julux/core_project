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
    setup_labeledslider,
    setup_pushbutton,
    setup_hswitch,
    setup_radiobutton,
    setup_savefileselect,
    setup_dirselect,
    setup_spinbox,
)
from .layer_select import setup_layerselect
from napari.utils.action_manager import action_manager
from napari.utils.events.event import WarningEmitter
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QFileDialog,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
import traceback

from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls

from napari.utils.notifications import show_info, show_warning, show_error, show_console_notification
from napari import Viewer
import SimpleITK as sitk
from scipy.interpolate import interpn

def interpolate_labels(label_a, label_b, num_steps):
    sitk_img_a = sitk.GetImageFromArray(label_a)
    sitk_distance_map_a = sitk.GetArrayFromImage(sitk.ApproximateSignedDistanceMap(sitk_img_a))
    
    sitk_img_b = sitk.GetImageFromArray(label_b)
    sitk_distance_map_b = sitk.GetArrayFromImage(sitk.ApproximateSignedDistanceMap(sitk_img_b))
    
    distance_map= np.zeros((2,) + label_a.shape, dtype=np.float32)
    distance_map[0,:,:] = sitk_distance_map_a
    distance_map[1,:,:] = sitk_distance_map_b
    
    x, y, z = np.meshgrid(np.linspace(0, num_steps-1, num_steps), np.arange(label_a.shape[0]), np.arange(label_a.shape[1]))
    
    labels_interp = interpn((np.array([0,num_steps-1]), np.arange(label_a.shape[0]), np.arange(label_a.shape[1])), distance_map, np.array([x,y,z]).T, bounds_error=False, fill_value=0)
    labels_interp = labels_interp.transpose(1,2,0)
    labels_interp = (labels_interp <= 0).astype(np.uint8)
    return labels_interp
    
class ShapeBasedInterpolationWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        self.active_layer = None
        
        self.current_index = None
        self.manual_frames = None
        self.silence_events = False

        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)

        # layer select for image layer
        self.layerselect_a = setup_layerselect(
            _scroll_layout, self._viewer, Labels, function=lambda: self.on_layer_change()
        )

        self.run_button = setup_iconbutton(
            _scroll_layout,
            "Start",
            "right_arrow",
            self._viewer.theme,
            function=lambda: self.run_shape_based_interpolation()
        )

        self.setup_connections()

        self.current_index = None


    def on_layer_change(self):
        pass
        """
        Connect transform events (scale/rotate/translate) of the selected
        image layer to preview and prompt layers so they follow image
        transformations.
        """
        label_layer, img_layer_idx = get_value(self.layerselect_a)

        if label_layer not in self._viewer.layers:
            return
        
        self.setup_connections()

    def clear(self):
        pass

    def run_shape_based_interpolation(self):
        label_layer, img_layer_idx = get_value(self.layerselect_a)
        
        label_data = np.transpose(
            self._viewer.layers[label_layer].data, self._viewer.dims.order)
        N = len(label_data.shape)
        if N != 3:
            show_warning("Shape base interpolation only works on 3D data.")
            return
        current_index = self._viewer.dims.current_step[self._viewer.dims.order[0]]

        if self.manual_frames is not None:
            non_empty_labels = self.manual_frames
        else:
            non_empty_labels = np.where(label_data.sum(axis=(1,2)) != 0)[0]
            self.manual_frames = non_empty_labels.tolist()
        print(f"Non empty labels found at frames: {non_empty_labels}")

        if len(non_empty_labels) < 2:
            show_warning("At least two frames with labels are required for shape based interpolation.")
            return
        #label_data_sbi = label_data.transpose(1,2,0)

        print(non_empty_labels)
        new_labels = label_data.copy()
        # for each pair of non empty labels, interpolate
        for i in range(len(non_empty_labels) - 1):
            frame_a = non_empty_labels[i]
            frame_b = non_empty_labels[i+1]
            num_steps = frame_b - frame_a + 1
            if num_steps <= 2:
                continue
            print(f"Interpolating between frames {frame_a} and {frame_b} with {num_steps} steps.")
            labels_a = label_data[frame_a]
            labels_b = label_data[frame_b]
            labels_interp = interpolate_labels(labels_a, labels_b, num_steps)

            new_labels[frame_a+1:frame_b] = labels_interp[1:-1]

        non_empty_labels_new = np.where(new_labels.sum(axis=(1,2)) != 0) 
        print(f"Non empty labels found at frames: {non_empty_labels_new}")

        print("Shape based interpolation done.")
        self.silence_events = True
        np.copyto(label_data, new_labels)
        self.silence_events = False
        self._viewer.layers[label_layer].refresh()

    def setup_connections(self):
        label_layer, img_layer_idx = get_value(self.layerselect_a)
        if label_layer is None or img_layer_idx == -1 or label_layer not in self._viewer.layers:
            return
        print(f"Label layer selected: {label_layer}, idx: {img_layer_idx}")

        label_layer = self._viewer.layers[label_layer]

        label_layer.events.set_data.connect(self._on_set_data)
        label_layer.events.labels_update.connect(self._on_labels_update)
    
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

        # add current frame to manual frames
        current_index = self._viewer.dims.current_step[self._viewer.dims.order[0]]
        if self.current_index is None or self.current_index != current_index:
            self.current_index = current_index
        else:
            return

        if self.manual_frames is None:
            self.manual_frames = [current_index]
        elif current_index not in self.manual_frames:
            self.manual_frames.append(current_index)
            self.manual_frames.sort()
            print(f"Added frame {current_index} to manual frames: {self.manual_frames}")        

    def showEvent(self, event):
        self.setup_connections()
    
    def closeEvent(self, event):
        self.hideEvent(event)

    def hideEvent(self, event):
        label_layer, img_layer_idx = get_value(self.layerselect_a)
        if label_layer is None or img_layer_idx == -1 or label_layer not in self._viewer.layers:
            return
        print(f"Label layer selected: {label_layer}, idx: {img_layer_idx}")

        label_layer = self._viewer.layers[label_layer]

        label_layer.events.set_data.disconnect(self._on_set_data)
        label_layer.events.labels_update.disconnect(self._on_labels_update)
