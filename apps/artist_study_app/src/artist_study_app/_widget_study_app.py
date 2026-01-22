import json
from pathlib import Path

import threading
import time
import torch
from yaml import safe_load
import os
import cv2
from magicgui import magicgui
from napari.layers import Image
from typing import TYPE_CHECKING
from functools import partial
import numpy as np
import random
from itertools import product

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
from napari.utils.action_manager import action_manager
from napari.utils.events.event import WarningEmitter
from napari.utils.notifications import show_info
from napari.qt.threading import thread_worker
from napari_toolkit.widgets import setup_iconbutton, setup_label
from qtpy.QtWidgets import (
    QFileDialog,
    QSizePolicy,
    QVBoxLayout,
    QPushButton,
    QWidget
)
from qtpy.QtCore import Qt  # type: ignore[attr-defined]

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
from napari_quick_view._widget_file_list import FileListWidget

from napari_quick_view.file_select import setup_dirselect
from napari_quick_view.layer_select import setup_layerselect
from napari_nninteractive.layers.fixed_image_layer import FixedImageLayer
from napari_nninteractive.layers.manual_labels_layer import ManualLabelsLayer
from napari_nninteractive.widget_manual import ManualSegmentationWidget

class StudyAppWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        main_layout = QVBoxLayout(self)

        _layout = main_layout

        self.user_id_input = setup_lineedit(
            _layout, "Physician ID", "Physician ID", function=lambda: None
        )
        self.file_select = setup_fileselect(
            _layout, "Select study file", filtering="YAML files (*.yaml *.yml)", function=lambda: None
        )
        self.file_select.set_file("/Users/tomjulius/Developer/core_project/apps/artist_study_app/study.yaml")

        self.init_button = setup_iconbutton(
            _layout, "Initialize", "right_arrow", self._viewer.theme, self.initalize
        )
        #self.initalize()

    def initalize(self):
        # initialize
        physicsian_id = get_value(self.user_id_input)
        study_protocol_path = get_value(self.file_select)
        widget = StudyAppFullWidget(self._viewer, physicsian_id, study_protocol_path)
        self._viewer.window.add_dock_widget(
            widget, name="ARTIST study", area="left"
        )
        # close self
        self.close()
        self._viewer.window.remove_dock_widget(self)
        self.deleteLater()

    def hideEvent(self, event):
        # ignore
        event.ignore()
        pass

class StudyAppFullWidget(QWidget):
    def __init__(self, viewer: Viewer, user_id, study_protocol_path):
        super().__init__()
        self._viewer = viewer

        self.user_id = user_id
        self.study_protocol_path = study_protocol_path

        # read study
        with open(study_protocol_path, 'r') as f:
            self.study_protocol = safe_load(f)
        
        print("Loaded study protocol:", self.study_protocol)

        study_methods = self.study_protocol.get("methods", [])
        study_cases = self.study_protocol.get("cases", [])
        cases_root_dir = self.study_protocol.get("cases_root_dir", "")
        # prepend root dir to case paths
        if cases_root_dir != "":
            for case in study_cases:
                case["file"] = os.path.join(cases_root_dir, case["file"])

        order = self.study_protocol.get("order", "random")
        output_folder = self.study_protocol.get("output_folder")
        if output_folder is not None and output_folder != "":
                os.makedirs(output_folder, exist_ok=True)
        # cathesian product of methods and cases
        self.study_tasks = []
        self.completed_study_tasks = {}

        for method, case in product(study_methods, study_cases):
            task = ({
                "task_id": f"{method}_{case['id']}",
                "method": method,
                "file": case["file"],
                "case_id": case["id"]
            }) 

            # check if there are existing approved segmentations
            if output_folder is not None and output_folder != "":
                existing_files = glob.glob(os.path.join(
                    output_folder,
                    f"{self.user_id}_case{case['id']}_method{method}_layer*.mha"
                ))
                if len(existing_files) > 0:
                    self.completed_study_tasks[task["task_id"]] = True
            
            self.study_tasks.append(task)

        if order == "random":
            # seed with sum of user_id characters
            # this way the order is reproducible for each user
            random.seed(sum(ord(c) for c in self.user_id))
            # shuffle tasks
            random.shuffle(self.study_tasks)
        elif order == "sequential-methods":
            # group by methods
            self.study_tasks.sort(key=lambda x: (x["method"], x["case_id"]))
        elif order == "sequential-cases":
            # group by cases
            self.study_tasks.sort(key=lambda x: (x["case_id"], x["method"]))

        main_layout = QVBoxLayout(self)

        self.current_task_index = 0
        self.image_layer = None

        self.manual_segmentation_widget = None
        self.automatic_segmentation_widget = None

        _layout = main_layout

        self.task_counter_label = setup_label(
            _layout, f"Task: {self.current_task_index+1} / {len(self.study_tasks)} ({len(self.completed_study_tasks)} completed)")

        hstack(_layout, [ 
            setup_iconbutton(
                _layout, "Previous", "step_left", self._viewer.theme, self.load_previous_task),
            setup_iconbutton(
                _layout, "Next", "step_right", self._viewer.theme, self.load_next_task
            )], stretch=[0,0])

        #self.pause_button = setup_iconbutton(
        #    _layout, "Pause", "erase", self._viewer.theme, lambda: None
        #)

        self.approve_button = setup_iconbutton(
            _layout, "Approve", "erase", self._viewer.theme, self.approve
        )
        self.approve_button.setToolTip("Approve current segmentation and move to next case.")

        self.modify_napari_ui()
        self.load_task(self.study_tasks[self.current_task_index])
    
    def update_task_counter(self):
        self.task_counter_label.setText(
            f"Task: {self.current_task_index+1} / {len(self.study_tasks)} ({len(self.completed_study_tasks)} completed)"
        )
    def load_next_task(self):
        if self.current_task_index < len(self.study_tasks) - 1:
            self.current_task_index += 1
            self.clear_task()
            self.load_task(self.study_tasks[self.current_task_index])
    
    def load_previous_task(self):
        if self.current_task_index > 0:
            self.current_task_index -= 1
            self.clear_task()
            self.load_task(self.study_tasks[self.current_task_index])

    def clear_task(self):
        if self.image_layer is not None:
            self._viewer.layers.remove(self.image_layer)
            self.image_layer = None
        
        # remove other labels layers
        for layer in self._viewer.layers:
            if isinstance(layer, Labels):
                self._viewer.layers.remove(layer)

        if self.manual_segmentation_widget is not None:
            self.manual_segmentation_widget.allow_close = True
            self.manual_segmentation_widget.parent().hide()
        if self.automatic_segmentation_widget is not None:
            self.automatic_segmentation_widget.parent().hide()

    def load_task(self, task):
        method = task["method"]
        path = task["file"]
        case_id = task["case_id"]

        img_sitk = sitk.ReadImage(path)

        img = sitk.GetArrayFromImage(img_sitk)

        self.image_layer = FixedImageLayer(
            img,
            name=f'Image {case_id}',
            colormap='gray'
        )
        print(img_sitk.GetSpacing())
        self.image_layer.scale = np.array([-1,1,1]) *np.array(img_sitk.GetSpacing()[::-1])  # reverse for napari xyz vs sitk zyx

        self._viewer.add_layer(self.image_layer)

        # load approved segmentations if existing
        output_folder = self.study_protocol.get("output_folder", "")
        if output_folder != "":
            for file in glob.glob(os.path.join(
                output_folder,
                f"{self.user_id}_case{case_id}_method{method}_layer*.mha"
            )):
                layer_name = os.path.basename(file).split(f"{self.user_id}_case{case_id}_method{method}_layer")[-1].replace(".mha", "")
                seg_sitk = sitk.ReadImage(file)
                seg = sitk.GetArrayFromImage(seg_sitk)
                seg_layer = ManualLabelsLayer(
                    seg,
                    name=f"{layer_name}",
                )
                seg_layer.scale = np.array([-1,1,1]) * np.array(img_sitk.GetSpacing()[::-1])  # reverse for napari xyz vs sitk zyx

                self._viewer.add_layer(seg_layer)
        if method == "manual":
            if self.automatic_segmentation_widget is not None:
                self.automatic_segmentation_widget.parent().hide()
            if self.manual_segmentation_widget is None:
                self.manual_segmentation_widget = ManualSegmentationWidget(self._viewer)
                self._viewer.window.add_dock_widget(
                    self.manual_segmentation_widget, name="Manual Segmentation", area="right"
                )
                self.manual_segmentation_widget.parent()._close_btn = False
            else:
                self.manual_segmentation_widget.allow_close = False
                self.manual_segmentation_widget.parent().show()
        elif method == "nnInteractive":
            if self.manual_segmentation_widget is not None:
                self.manual_segmentation_widget.allow_close = True
                self.manual_segmentation_widget.parent().hide()
            if self.automatic_segmentation_widget is None:
                from napari_nninteractive import nnInteractiveWidget
                self.automatic_segmentation_widget = nnInteractiveWidget(self._viewer)
                self._viewer.window.add_dock_widget(
                    self.automatic_segmentation_widget, name="nnInteractive Segmentation", area="right"
                )
                self.automatic_segmentation_widget.parent()._close_btn=False
            else:
                self.automatic_segmentation_widget.parent().show()
        
        self.update_task_counter()

    def approve(self):
        show_info(f"Approved task {self.study_tasks[self.current_task_index]['task_id']}. Saving results...")
        # write all label layers to disk
        method = self.study_tasks[self.current_task_index]["method"]
        case_id = self.study_tasks[self.current_task_index]["case_id"]
        output_folder = self.study_protocol.get("output_folder", "")

        for layer in self._viewer.layers:
            if isinstance(layer, Labels):
                output_path = os.path.join(
                    output_folder,
                    f"{self.user_id}_case{case_id}_method{method}_layer{layer.name}.mha"
                )
                layer_data = layer.data.astype(np.uint8)
                sitk_img = sitk.GetImageFromArray(layer_data)
                sitk.WriteImage(sitk_img, output_path)
                #show_info(f"Saved layer {layer.name} to {output_path}")
        
        show_info(f"Saved results for task {self.study_tasks[self.current_task_index]['task_id']}.")
        self.completed_study_tasks[self.study_tasks[self.current_task_index]["task_id"]] = True
        
        self.update_task_counter()
    
    def modify_napari_ui(self):
        viewer = self._viewer

        def set_axial_view():
            viewer.dims.order = (0,1,2)
        def set_coronal_view():
            viewer.dims.order = (1,0,2)
        def set_saggital_view():
            viewer.dims.order = (2,0,1)
        axial_button = QPushButton()
        axial_button.setText("A")
        axial_button.clicked.connect(set_axial_view)
        axial_button.setStyleSheet("""
            min-width : 28px;
            max-width : 28px;
            min-height : 28px;
            max-height : 28px;
            padding: 0px;
            """)
           
        viewer.window._qt_viewer._viewerButtons.layout().insertWidget(-1,axial_button)
        axial_button = QPushButton()
        axial_button.setText("C")
        axial_button.clicked.connect(set_coronal_view)
        axial_button.setStyleSheet("""
            min-width : 28px;
            max-width : 28px;
            min-height : 28px;
            max-height : 28px;
            padding: 0px;
            """)
        viewer.window._qt_viewer._viewerButtons.layout().insertWidget(-1,axial_button)
        axial_button = QPushButton()
        axial_button.setText("S")
        axial_button.clicked.connect(set_saggital_view)
        axial_button.setStyleSheet("""
            min-width : 28px;
            max-width : 28px;
            min-height : 28px;
            max-height : 28px;
            padding: 0px;
            """)
        viewer.window._qt_viewer._viewerButtons.layout().insertWidget(-1,axial_button)

        # Hide viewer buttons since we offer our own functionality
        viewer.window._qt_viewer._viewerButtons.rollDimsButton.setHidden(True)
        viewer.window._qt_viewer._viewerButtons.transposeDimsButton.setHidden(True)
        viewer.window._qt_viewer._viewerButtons.consoleButton.setHidden(True)
        viewer.window._qt_viewer._viewerButtons.gridViewButton.setHidden(True)
        viewer.window._qt_viewer._viewerButtons.ndisplayButton.setHidden(True)

        # Hide layer list buttons
        viewer.window._qt_viewer._layersButtons.setHidden(True)

        # Hotwire to disable delete/backspace/enter keys in layer list
        self._prev_layer_keyPressEvent_handler = viewer.window._qt_viewer._layers.keyPressEvent
        def new_func(e):
            if e is None:
                return
            if e.key() in (Qt.Key.Key_Backspace, Qt.Key.Key_Delete):
                e.ignore()
            else:
                self._prev_layer_keyPressEvent_handler(e)
        viewer.window._qt_viewer._layers.keyPressEvent = new_func
    
    def revert_napari_ui(self):
        viewer = self._viewer
        viewer.window._qt_viewer._viewerButtons.rollDimsButton.setHidden(False)
        viewer.window._qt_viewer._viewerButtons.transposeDimsButton.setHidden(False)
        viewer.window._qt_viewer._viewerButtons.consoleButton.setHidden(False)
        viewer.window._qt_viewer._viewerButtons.gridViewButton.setHidden(False)
        viewer.window._qt_viewer._viewerButtons.ndisplayButton.setHidden(False)

        viewer.window._qt_viewer._layersButtons.setHidden(False)

        viewer.window._qt_viewer._layers.keyPressEvent = self._prev_layer_keyPressEvent_handler
        del self._prev_layer_keyPressEvent_handler

        for i in range(3):
            viewer.window._qt_viewer._viewerButtons.layout().removeWidget(
                viewer.window._qt_viewer._viewerButtons.layout().itemAt(
                    viewer.window._qt_viewer._viewerButtons.layout().count()-1
                ).widget()
            )

    def showEvent(self, event):
        pass
    
    def closeEvent(self, event):
        #event.ignore()
        self.revert_napari_ui()

        if self.image_layer is not None:
            self._viewer.layers.remove(self.image_layer)
            self.image_layer = None
        if self.manual_segmentation_widget is not None:
            self.manual_segmentation_widget.allow_close = True
            self._viewer.window.remove_dock_widget(self.manual_segmentation_widget)
            self.manual_segmentation_widget.close()
            self.manual_segmentation_widget = None
            self.manual_segmentation_widget.deleteLater()
        if self.automatic_segmentation_widget is not None:
            self._viewer.window.remove_dock_widget(self.automatic_segmentation_widget)
            self.automatic_segmentation_widget.close()
            self.automatic_segmentation_widget = None
            self.automatic_segmentation_widget.deleteLater()

        # reopen the study app widget
        widget = StudyAppWidget(self._viewer)
        self._viewer.window.add_dock_widget(
            widget, name="ARTIST study", area="left"
        )

    def hideEvent(self, event):
        # ignore
        event.ignore()
        pass
