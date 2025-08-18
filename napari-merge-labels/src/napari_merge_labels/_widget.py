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
from napari.layers import Labels
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
    setup_plaintextedit,
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


class MergeLabelsWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        self.metrics = {
            "dsc": DiceMetric(),
            "hd95": HausdorffDistanceMetric(percentile=95),
            "Sur": SurfaceDistanceMetric(),
            "CenterDistance": EuclideanCenterDistanceMetric()
        }

        self.build_gui()

    # GUI
    def build_gui(self):
        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)


        _container, _layout = setup_vgroupbox(_scroll_layout, "Label layer selection:")
       
        self.layerselect_a = setup_layerselect(
            None, self._viewer, Labels, function=lambda: None
        )

        self.layerselect_b = setup_layerselect(
            None, self._viewer, Labels, function=lambda: self.on_labels_update_event
        )

        _ = hstack(_layout, [self.layerselect_a, self.layerselect_b])

        _container, _layout = setup_vgroupbox(_scroll_layout, "Metric mode:")
        _ = setup_label(_layout, "Select how to merge the labels.")
        _ = setup_combobox(
            _layout, ["STAPLE", "Majority", "Union", "Intersect"], "QComboBox", function=lambda: print("QComboBox")
        )
        _container, _layout = setup_vcollapsiblegroupbox(_scroll_layout, "Settings", False)
        self.compute_on_change = setup_checkbox(_layout, "Rerun on change", True)
        #self.compute_slow_change = setup_checkbox(_layout, "Compute slow metrics", True)

        self.compute_metrics_btn = setup_pushbutton(None, "Compute Metrics", function=lambda: self.compute_metrics())
        _ = hstack(_scroll_layout, [self.compute_metrics_btn])
        
        # Labels to show depending on the selection of layers
        self.no_labels_selected = setup_label(_scroll_layout, "Please select two label layers to compute metrics.")
        self.no_labels_selected.setVisible(True)

        self.same_labels_selected = setup_label(_scroll_layout, "Please select two different label layers to compute metrics.")
        self.same_labels_selected.setVisible(False)

        # Metrics container
        self.metrics_container, _layout = setup_vgroupbox(_scroll_layout, "Metrics:")
        self.metrics_container.setVisible(False)

        self.metrics_text_file = setup_plaintextedit(
            _layout, "QPlainTextEdit", "QPlainTextEdit", function=None
        )
        self.metrics_text_file.setReadOnly(True)

        setup_label(_layout, "DSC: 0.8")
        setup_label(_layout, "Jaccard: 0.7")
        setup_label(_layout, "Precision: 0.9")
        setup_label(_layout, "Recall: 0.85")
        
        # Explanation section
        _container, _layout = setup_vcollapsiblegroupbox(_scroll_layout, "Explaination", False)
        _ = setup_label(_layout, "Got To <a href='https://metrics-reloaded.dkfz.de/metric?id=dsc' target='_blank'>Metrics Reloaded</a> for more information.")
        _.setOpenExternalLinks(True)

    def compute_metrics(self):
        layer_a_name, layer_a_index  = get_value(self.layerselect_a)
        layer_b_name, layer_b_index = get_value(self.layerselect_b)
        print(layer_a_name, layer_a_index, layer_b_name, layer_b_index)

        if layer_a_index is None or layer_b_index is None:
            self.no_labels_selected.setVisible(True)
            self.same_labels_selected.setVisible(False)
            return

        if layer_a_index == layer_b_index:
            self.no_labels_selected.setVisible(False)
            self.same_labels_selected.setVisible(True)
            return

        self.no_labels_selected.setVisible(False)
        self.same_labels_selected.setVisible(False)
        self.metrics_container.setVisible(True)

        y_true = self._viewer.layers[layer_a_index].data > 0
        y_pred = self._viewer.layers[layer_b_index].data > 0

        print(y_true.shape, y_pred.shape)
        
        # Ensure y_true and y_pred transpose to (Z, H, W)
        y_true = y_true.transpose(*self._viewer.dims.order[::-1])
        y_true = y_true.transpose(2,0,1)
        y_pred = y_pred.transpose(*self._viewer.dims.order[::-1])
        y_pred = y_pred.transpose(2,0,1)

        print(y_true.shape, y_pred.shape)

        T, H, W = y_true.shape

        # Convert to the shape required by monai (T,C,H,W) where C = 1 is the channel dimension

        y_true_monai = y_true.reshape(T,1,H,W)
        y_pred_monai = y_pred.reshape(T,1,H,W)


        for metric_name, metric in self.metrics.items():
            result = metric(y_true_monai, y_pred_monai)
            self.metrics_text_file.appendPlainText(f"{metric_name}: {result.mean():.4f}")

    def on_labels_update_event(self, event):
        """Handle the labels update event. The data of a layer has changed."""
        pass
        # recompute the metrics if the compute_on_change checkbox is checked
        if self.compute_on_change.isChecked():
            self.compute_metrics()
        
    def on_layer_select_change_event(self, event):
        """Handle layer events such as insertion, removal, or change."""
        # Check if the event is related to labels
        # If the compute_on_change checkbox is checked, recompute the metrics
        if self.compute_on_change.isChecked():
            self.compute_metrics()