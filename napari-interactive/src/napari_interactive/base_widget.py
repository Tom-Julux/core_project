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
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QFileDialog,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)
import traceback

from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls

from napari_promptable.controls.bbox_controls import CustomQtBBoxControls
from napari_promptable.controls.lasso_controls import CustomQtLassoControls
from napari_promptable.controls.point_controls import CustomQtPointsControls
from napari_promptable.controls.scribble_controls import CustomQtScribbleControls
from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls

from napari.layers import Shapes, Points, Labels
from contextlib import contextmanager


class BoxPromptLayer(Shapes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class PointPromptLayer(Points):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ContourPromptLayer(Shapes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


class ScribblePromptLayer(Labels):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)


layer_to_controls[PointPromptLayer] = CustomQtPointsControls
layer_to_controls[BoxPromptLayer] = CustomQtBBoxControls
layer_to_controls[ScribblePromptLayer] = CustomQtScribbleControls
layer_to_controls[ContourPromptLayer] = QtShapesControls


class InteractiveSegmentationWidgetBase(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        # layers managed by this widget
        self.prompt_layers = {}
        self.preview_layer = None  # Layer to show the preview of the segmentation
        self.preview_label_data = None  # Data Label for the preview layer

        self.propagating_lock = threading.Lock()
        self.rerun_after_lock = False
        self.prevent_auto_run_on_change = False

        self.build_gui()
        self.load_model()
        print("InteractiveSegmentationWidgetBase initialized")
        if get_value(self.layerselect_a)[0] is not None:
            self.setup_preview_layer()
            self.update_prompt_type()
            self.connect_image_layer_events()
            self.on_image_layer_scale_or_rotate()
            self.run_button.setEnabled(True)

    @property
    def supported_prompt_types(self):
        """
        Supported prompt types for the widget.

        Return a list of prompt type names that this widget supports. Subclasses
        should override this property to advertise supported prompt types such
        as "Points", "BBox" or "Mask".

        Returns:
            list[str]: list of supported prompt type strings.
        """
        return ["Mask", "Manual"]  # ["Points", "BBox", "Mask"]

    def load_model(self):
        """
        Load or initialize the segmentation model or predictor.

        Contract: subclasses should implement model loading here. The method may
        allocate GPU resources and should respect availability (for example
        check torch.cuda.is_available()). This method is called during widget
        initialization. Implementations should leave the widget in a usable
        state (or raise) and should be idempotent where possible.
        """
        pass

    def predict(self):
        """
        Run a prediction using the current prompt state and update preview.

        Contract: gather prompts from prompt layers (points, bbox, mask,
        etc.), prepare the image/frame data, call the model/predictor API and
        then call add_prediction_to_preview(...) to merge results into the
        preview. Should be safe to call from a background thread (run_predict_in_thread
        handles locking).
        """
        pass

    def reset_model(self):
        """
        Reset or clear any loaded model/state to free resources.

        Contract: implementations should clear predictor objects, GPU memory or
        cached state so that the widget can be reinitialized or closed without
        leaking resources. This is called when the image layer changes and on
        widget close.
        """
        pass

    def setup_hyperparameter_gui(self, _layout):
        """
        Populate the hyperparameter section of the GUI.

        Args:
            _layout: a layout/container provided by the GUI builder where widgets
                for model hyperparameters (thresholds, sliders, checkboxes)
                should be added. Subclasses should add widgets and connect
                their change callbacks to on_hyperparameter_update().
        """
        pass

    def setup_model_selection_gui(self, _scroll_layout):
        """
        Add model selection controls to the GUI.

        Args:
            _scroll_layout: the scrollable layout provided by build_gui where
                model selection controls (file selectors, model presets) can
                be placed. Subclasses may implement a choice of checkpoints
                or GPU/CPU switching here.
        """
        pass

    def setup_view_control_gui(self, _scroll_layout):
        """
        Add view-selection and view-control widgets to the GUI.

        Args:
            _scroll_layout: layout provided by build_gui where view controls
                (for example multi-view switches, set-prompt buttons) should
                be added. Subclasses implementing 2D/3D behavior should
                override this to provide appropriate controls.
        """
        pass

    def setup_second_propagation_gui(self, _scroll_layout):
        """
        Add view-selection and view-control widgets to the GUI.

        Args:
            _scroll_layout: layout provided by build_gui where view controls
                (for example multi-view switches, set-prompt buttons) should
                be added. Subclasses implementing 2D/3D behavior should
                override this to provide appropriate controls.
        """
        pass

    # region GUI
    def build_gui(self):
        """
        Build and wire the widget GUI.

        Creates the main layout and adds model selection, image selection,
        prompt controls, hyperparameters, propagation controls and export
        buttons. This method only constructs UI widgets and wires callbacks.
        """

        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)

        self.setup_model_selection_gui(_scroll_layout)

        _container, _layout = setup_vgroupbox(
            _scroll_layout, "Image Selection:")
        # layer select for image layer
        self.layerselect_a = setup_layerselect(
            _layout, self._viewer, Image, function=lambda: self.on_image_layer_change()
        )

        _container, _layout = setup_vgroupbox(
            _scroll_layout, "Prompt Selection:")

        self.prompt_type_select = setup_combobox(  # , "Points", "BBox"],\
            _layout, self.supported_prompt_types, "QComboBox", function=lambda: self.update_prompt_type()
        )

        self.setup_view_control_gui(_scroll_layout)

        _container, _layout = setup_vgroupbox(
            _scroll_layout, "Hyperparameters:")

        self.setup_hyperparameter_gui(_layout)

        _group_box, _layout = setup_vgroupbox(
            _scroll_layout, text="Predict:")

        self.run_button = setup_iconbutton(
            _layout,
            "Predict",
            "right_arrow",
            self._viewer.theme,
            function=lambda: self.run_predict_in_thread(),
            tooltips="Run the predict step",
        )
        self.run_button.setEnabled(False)

        # Status label shown under the predict button. Updated when a
        # prediction worker is running, finished, or when a re-run is
        # scheduled while a run is active.
        self.status_label = setup_label(_layout, "Status: Ready")

        self.autorun_ckbx = setup_checkbox(
            _layout,
            "Auto Run Prediction",
            False,
            tooltips="Run automatically after each interaction once all three view prompts are set.",
        )

        self.setup_second_propagation_gui(_scroll_layout)

        _container, _layout = setup_vcollapsiblegroupbox(
            _scroll_layout, "Multi Mode:", collapsed=False)
        _ = setup_label(
            _layout, "Create multiple (non-overlapping) predictions.")

        self.overwrite_existing_mm_ckbx = setup_checkbox(
            _layout,
            "Overwrite existing",
            False,
            tooltips="By default, new predictions will only be added to empty regions of the preview layer. If checked, new predictions will overwrite existing labels in the preview layer.",
            function=lambda: self.run_predict_in_thread()
        )

        _label = setup_label(None, "Current Object:")
        def on_spin_box_change():
            if self.preview_layer is not None:
                self.preview_layer.selected_label = get_value(self.object_id_spinbox)

            self.run_predict_in_thread()
        self.object_id_spinbox = setup_spinbox(
            None, 1, 255, 1, function=on_spin_box_change)

        _ = hstack(_layout, [_label, self.object_id_spinbox], stretch=[0, 1])

        _ = setup_iconbutton(
            _layout, "Next Object", "right_arrow", self._viewer.theme, self.increment_object_id
        )

        _container, _layout = setup_vgroupbox(
            _scroll_layout, "Export to layer:")
        _ = setup_label(
            _layout, "Export the contents of the preview layer to a separate layer.")

        # self.export_to_new_layer_ckbx = setup_checkbox(
        #    _layout, "Accumulate",True, tooltips="If unchecked, the preview layer will be exported to the currently selected layer.")

        _ = setup_iconbutton(
            _layout, "Export to layer", "pop_out", self._viewer.theme, self.export_preview
        )
        _container, _layout = setup_vcollapsiblegroupbox(
            _scroll_layout, "Export to file:", collapsed=True)
        _ = setup_label(
            _layout, "Export the contents of the exported layer (or the preview) to a new file.")

        self.export_file_select = setup_savefileselect(
            _layout,
            "Export File:",
            read_only=False,
            filtering="Images (*.mha)",
            tooltips="File to export the current preview layer to.",
            function=lambda: None,
        )

        _ = setup_iconbutton(
            _layout, "Export to file", "copy_to_clipboard", self._viewer.theme, self.export_to_file
        )

        # Reset group
        _container, _layout = setup_vgroupbox(_scroll_layout, "Reset:")

        _ = setup_label(
            _layout, "Reset the widget and clear all prompt and preview layers.")

        _ = setup_iconbutton(
            _layout, "Reset", "erase", self._viewer.theme, lambda: self.on_image_layer_change()
        )

    def showEvent(self, event=None):
        self.on_image_layer_change()

    def hideEvent(self, event=None):
        self.closeEvent()

    def closeEvent(self, event=None):
        """
        Clean up widget resources before closing.

        Called by Qt when the widget is being closed. This method clears
        prompt and preview layers, disconnects image events and resets the
        model to free resources (GPU memory, threads, etc.).
        """
        self.clear_prompt_layers()
        self.clear_preview_layer()
        self.disconnect_image_layer_events()
        self.reset_model()

        self._viewer.layers.events.inserted.disconnect(
            self.layerselect_a._update)
        self._viewer.layers.events.removed.disconnect(
            self.layerselect_a._update)
        layer_names = {layer: layer.name for layer in self._viewer.layers if self.layerselect_a.layer_types is None or isinstance(
            layer, self.layerselect_a.layer_types)}
        for layer in layer_names:
            layer.events.name.disconnect(self.layerselect_a._update)
    # endregion

    # region Image Layer Management

    def on_image_layer_change(self, event=None):
        """
        Handle changes to the selected image layer.

        This routine clears existing prompts and previews, resets the model
        state and, if a valid image layer is selected, sets up a preview layer
        and connects image transform events so prompts and preview follow the
        image when scaled/rotated/translated.
        """
        self.clear_prompt_layers()
        self.clear_preview_layer()
        self.disconnect_image_layer_events()
        self.reset_model()

        img_layer = get_value(self.layerselect_a)[0]
        if img_layer is None or img_layer not in self._viewer.layers:
            show_warning("Please select a valid image layer.")
            self.run_button.setEnabled(False)
            return

        self.setup_preview_layer()
        self.update_prompt_type()
        self.connect_image_layer_events()
        self.on_image_layer_scale_or_rotate()
        self.run_button.setEnabled(True)

    def connect_image_layer_events(self):
        """
        Connect transform events (scale/rotate/translate) of the selected
        image layer to preview and prompt layers so they follow image
        transformations.
        """
        # connect the image layer's scale and rotation changes to the preview layer
        img_layer = get_value(self.layerselect_a)[1]
        img_layer = self._viewer.layers[img_layer]
        img_layer.events.scale.connect(self.on_image_layer_scale_or_rotate)
        img_layer.events.rotate.connect(self.on_image_layer_scale_or_rotate)
        img_layer.events.translate.connect(self.on_image_layer_scale_or_rotate)

    def disconnect_image_layer_events(self):
        """
        Disconnect transform event handlers from the currently selected
        image layer if present. Safe to call repeatedly.
        """
        img_layer = get_value(self.layerselect_a)[1]
        if img_layer is not None and img_layer in self._viewer.layers:
            img_layer = self._viewer.layers[img_layer]
            img_layer.events.scale.disconnect(
                self.on_image_layer_scale_or_rotate)
            img_layer.events.rotate.disconnect(
                self.on_image_layer_scale_or_rotate)
            img_layer.events.translate.disconnect(
                self.on_image_layer_scale_or_rotate)

    def on_image_layer_scale_or_rotate(self, event=None):
        """
        Update preview and prompt layers to follow the selected image layer's
        transforms (scale, rotate, translate).

        This keeps prompts aligned with the image when the user zooms,
        rotates or translates the image layer.
        """
        img_layer = get_value(self.layerselect_a)[1]

        if self.preview_layer is not None:
            self.preview_layer.scale = self._viewer.layers[img_layer].scale
            self.preview_layer.rotate = self._viewer.layers[img_layer].rotate
            self.preview_layer.translate = self._viewer.layers[img_layer].translate

        for layer in self.prompt_layers.values():
            layer.scale = self._viewer.layers[img_layer].scale
            layer.rotate = self._viewer.layers[img_layer].rotate
            layer.translate = self._viewer.layers[img_layer].translate
    # endregion

    # region Preview Layer Management
    def setup_preview_layer(self):
        """
        Create and add the preview Labels layer used to display predictions.

        The preview layer is initialized with zeros and matched to the selected
        image layer's shape and transforms. The created Labels layer is stored
        in self.preview_layer and its backing array in self.preview_label_data.
        """
        img_layer = get_value(self.layerselect_a)[1]

        img_layer_shape = self._viewer.layers[img_layer].data.shape

        self.preview_label_data = np.zeros(img_layer_shape, dtype=np.uint8)
        self.preview_layer = Labels(
            name='Preview Layer', data=self.preview_label_data.copy(), opacity=0.5)
        self.preview_layer.contour = 1
        self.preview_layer.scale = self._viewer.layers[img_layer].scale
        self.preview_layer.rotate = self._viewer.layers[img_layer].rotate
        self.preview_layer.translate = self._viewer.layers[img_layer].translate

        self._viewer.add_layer(self.preview_layer)
        self.preview_layer.translate = self._viewer.layers[img_layer].translate

        def on_label_change():
            if self.preview_layer.selected_label != 0:
                set_value(self.object_id_spinbox, self.preview_layer.selected_label)

        self.preview_layer.events.selected_label.connect(on_label_change)

    def clear_preview_layer(self):
        """
        Remove the preview layer from the viewer and clear its backing data.
        """
        if self.preview_layer and self.preview_layer in self._viewer.layers:
            self._viewer.layers.remove(self.preview_layer)
            self.preview_label_data = None
            self.preview_layer = None
        set_value(self.object_id_spinbox, 1)
    # endregion
    
    # region Prompt Layer Management
    def update_prompt_type(self):
        """
        Create prompt layers according to the selected prompt type.

        Supported prompt types include Points, BBox and Mask. Existing prompt
        layers are cleared before new ones are created. The new prompt layers
        are added to the viewer and connected to on_prompt_update_event so the
        widget can react to prompt edits.
        """
        prompt_type = get_value(self.prompt_type_select)[0]

        self.clear_prompt_layers()

        # If not multi object mode clear the preview on prompt type change
        if not self.is_multi_object and not prompt_type == "Manual":
            self.preview_layer.data = np.zeros_like(self.preview_layer.data)

        img_layer = get_value(self.layerselect_a)[1]

        img_layer_shape = self._viewer.layers[img_layer].data.shape

        if prompt_type == "Points":
            point_layer_positive = PointPromptLayer(
                name='Point Point Layer (Positive)', ndim=len(img_layer_shape))
            point_layer_negative = PointPromptLayer(
                name='Point Point Layer (Negative)', ndim=len(img_layer_shape))
            self._viewer.add_layer(point_layer_positive)
            self._viewer.add_layer(point_layer_negative)
            self.prompt_layers['point_positive'] = point_layer_positive
            self.prompt_layers['point_negative'] = point_layer_negative

            point_layer_positive.size = 5
            point_layer_negative.size = 5

            point_layer_positive.events.data.connect(
                self.on_prompt_update_event)
            point_layer_negative.events.data.connect(
                self.on_prompt_update_event)

            # set active layer to positive point layer
            self._viewer.layers.selection.active = point_layer_positive
        elif prompt_type == "BBox":
            bbox_layer = BoxPromptLayer(
                name='BBox Prompt Layer', ndim=len(img_layer_shape),
                face_color = "#ffffff00",edge_color = "#ffffffff",opacity = 0.7)

            self._viewer.add_layer(bbox_layer)
            self.prompt_layers['bbox'] = bbox_layer


            bbox_layer.events.data.connect(self.on_prompt_update_event)
            # set active layer to bbox layer
            self._viewer.layers.selection.active = bbox_layer
        elif prompt_type == "Mask":
            data = np.zeros(img_layer_shape, dtype=np.uint8)
            mask_layer = ScribblePromptLayer(
                name='Mask Prompt Layer', data=data)
            mask_layer.contour = 1

            color_dict = {None: [0, 0, 0, 0], 0: [0, 0, 0, 0], 1: [255, 255, 255, 255]}
            mask_layer.colormap = DirectLabelColormap(color_dict = color_dict)

            self._viewer.add_layer(mask_layer)
            self.prompt_layers['mask'] = mask_layer

            #mask_layer.events.data.connect(self.on_prompt_update_event)
            # For the Labels layer, use the paint event to catch changes after they occured
            mask_layer.events.paint.connect(self.on_prompt_update_event)

            # set active layer to bbox layer
            self._viewer.layers.selection.active = mask_layer

        elif prompt_type == "Manual":
            self._viewer.layers.selection.active = self.preview_layer

    def clear_prompt_layer_content(self):
        """
        Clear the content of existing prompt layers without removing the layers
        themselves. Labels are zeroed and point/shape layers are emptied. Layers
        are refreshed so the UI updates immediately.
        """
        # Remove all existing prompt layers
        for layer in self.prompt_layers.values():
            if isinstance(layer, Labels):
                layer.data = np.zeros_like(layer.data, dtype=np.uint8)
            elif isinstance(layer, (Points, Shapes, BoxPromptLayer, PointPromptLayer, ContourPromptLayer)):
                layer.data = np.empty((0, layer.ndim))
            layer.refresh()

    def clear_prompt_layers(self):
        """
        Remove all prompt layers from the viewer and clear internal tracking.
        """
        # Remove all existing prompt layers
        for layer in self.prompt_layers.values():
            if layer in self._viewer.layers:
                self._viewer.layers.remove(layer)
        self.prompt_layers.clear()
    # endregion

    def on_hyperparameter_update(self):
        """
        Called when hyperparameters change to optionally trigger a prediction.

        If "Auto Run Prediction" is enabled this will start a background
        prediction run so the preview updates after parameter changes.
        """
        if get_value(self.autorun_ckbx) and self.run_button.isEnabled():
            self.run_predict_in_thread()

    def on_prompt_update_event(self, event):
        """
        Handle prompt layer change events.

        The event is ignored while add/remove/change actions are in progress.
        If Auto Run Prediction is enabled this will enqueue a background
        prediction run.
        """
        # Ignore in progress events like adding, removing, changing
        if hasattr(event, 'action') and event.action in ['adding', 'removing', 'changing']:
            return
        
        if self.prevent_auto_run_on_change:
            return
        if get_value(self.autorun_ckbx) and self.run_button.isEnabled():
            self.run_predict_in_thread()

    @contextmanager
    def no_autopredict(self):
        """
        Context manager that temporarily suppresses automatic predictions.

        Usage:
            with self.no_autopredict():
                # make multiple prompt edits without triggering predict

        The context manager supports nesting. It only prevents automatic
        triggers (prompt/hyperparameter events) from starting background
        predictions; manual calls to run_predict_in_thread() are not blocked.
        """
        self.prevent_auto_run_on_change = True
        yield
        self.prevent_auto_run_on_change = False

    def run_predict_in_thread(self):
        """Start a background worker that runs predict() safely with a lock.

        Ensures only one prediction runs at a time. If multiple triggers occur
        while a prediction is running, a single additional rerun will be
        scheduled and reflected in the status label as "Re-run scheduled".
        """
        print("run_predict_in_thread")

        # If a prediction is currently running, schedule a rerun and update
        # the status immediately.
        if self.propagating_lock.locked():
            self.rerun_after_lock = True
            self.run_button.setEnabled(False)
            self.status_label.setText("Status: Running (Re-run scheduled)")
            return

        # No need to start for manual prompting
        if get_value(self.prompt_type_select)[0] == "Manual":
            return

        @thread_worker
        def _worker():
            # Acquire the lock and run predictions; the loop allows a single
            # additional run if self.rerun_after_lock becomes True while
            # executing.
            with self.propagating_lock:
                self.rerun_after_lock = True
                while self.rerun_after_lock:
                    self.rerun_after_lock = False
                    try:  
                        self.predict()
                    except Exception as e:
                        print(f"Error in on_prompt_update_event: {e}")
                        print(f"Traceback: {traceback.format_exc()}")

        worker = _worker()

        # Update UI when worker starts
        def _on_started():
            self.run_button.setEnabled(False)
            self.status_label.setText("Status: Running...")
        # Update UI when worker finishes or errors

        def _on_done(*args, **kwargs):
            self.status_label.setText("Status: Ready")
            self.run_button.setEnabled(True)
        # Connect signals (thread_worker exposes started, finished, errored)
        worker.started.connect(_on_started)
        worker.finished.connect(_on_done)
        worker.errored.connect(_on_done)

        worker.start()

    def add_prediction_to_preview(self, new_mask, indices=None, transposed=False):
        """
        Merge a predicted binary mask into the preview Labels layer.

        Args:
            new_mask (np.ndarray): binary mask (non-zero = foreground) matching
                the slice/indices layout expected by the preview.
            indices: optional index/slice into the transposed preview where the
                mask should be applied. If None the entire preview is written.

        Behavior:
            - Converts non-zero mask pixels to the current object ID.
            - If overwrite checkbox is enabled, writes mask regardless of
              existing labels. Otherwise only writes into empty (0) pixels.
        """
        if self.preview_layer is None:
            setup_warning("No preview layer to add prediction to.")
            self.setup_preview_layer()
            return

        # update new_mask to object id
        object_id = get_value(self.object_id_spinbox)
        new_mask = np.where(new_mask > 0, object_id, 0).astype(np.uint8)

        # set preview layer data to preview label data with overwrites from the new mask at the indices
        out_mask = self.preview_label_data.copy() if self.is_multi_object else self.preview_layer.data.copy()

        if indices is None:
            indices = np.s_[:]

        transposed_out_mask = np.transpose(
            out_mask, self._viewer.dims.order) if not transposed else out_mask

        if get_value(self.overwrite_existing_mm_ckbx):
            # overwrite current preview_label_data at indices with new_mask where new_mask > 0
            if self.is_multi_object:
                np.copyto(transposed_out_mask[indices],
                      new_mask, where=(new_mask > 0))
            else:
                np.copyto(transposed_out_mask[indices], new_mask)
        else:
            np.copyto(transposed_out_mask[indices], new_mask, where=(
                new_mask > 0) & (transposed_out_mask[indices] == 0))

        self.preview_layer.data = out_mask
        self.preview_layer.refresh()

    @property
    def is_multi_object(self):
        object_ids = np.unique(self.preview_layer.data)
        return len(object_ids) > 2

    def increment_object_id(self):
        """
        Finalize the current object in the preview and increment the object ID.

        If "Overwrite existing" is set the internal preview backing array is
        replaced by the current layer contents. Otherwise only empty pixels are
        updated. The object ID spinbox is incremented (max 255) and prompt
        layers are cleared so the next object's prompts can be created.
        """
        # copy current preview label data to preview layer data
        if get_value(self.overwrite_existing_mm_ckbx):
            self.preview_label_data = self.preview_layer.data.copy()
        else:
            self.preview_label_data = np.where(
                self.preview_label_data == 0, self.preview_layer.data, self.preview_label_data).astype(np.uint8)

        current_id = get_value(self.object_id_spinbox)
        if current_id < 255:
            set_value(self.object_id_spinbox, current_id + 1)
        else:
            show_warning(
                "Max object count reached. Object ID cannot be greater than 255.")
            return

        # reset prompts
        self.clear_prompt_layer_content()

    # region Export

    def export_preview(self):
        """
        Export the preview as a new Labels layer in the viewer.

        Creates a new Labels layer using the preview data and copies transform
        metadata so the exported layer aligns with the source image.
        """
        # Export the contents of the preview layer to a separate layer
        if self.preview_layer is None:
            show_warning("No preview layer to export.")
            return

        # Create a new Labels layer with the data from the preview layer
        new_layer = Labels(name='Exported Layer',
                           data=self.preview_layer.data.copy())
        new_layer.scale = self.preview_layer.scale
        new_layer.rotate = self.preview_layer.rotate
        new_layer.translate = self.preview_layer.translate
        new_layer.opacity = 1.0
        new_layer.contour = 1
        # Add the new layer to the viewer
        self._viewer.add_layer(new_layer)
        show_info("Preview layer exported successfully.")

    def export_to_file(self):
        """
        Write the preview layer to disk using SimpleITK.

        The user-selected file path is used. The preview data is converted to a
        SimpleITK image and written with compression. Any exceptions are shown
        to the user via a notification.
        """
        if self.preview_layer is None:
            show_warning("No preview layer to export.")
            return

        file_path = get_value(self.export_file_select)
        if not file_path:
            show_warning(
                "Please select a file path to export the preview layer.")
            return

        try:
            import SimpleITK as sitk
            # Convert the numpy array to a SimpleITK image
            sitk_image = sitk.GetImageFromArray(
                self.preview_layer.data.astype(np.uint8))
            # Save the image to the specified file path
            sitk.WriteImage(sitk_image, file_path, useCompression=True)
            show_info(f"Preview layer exported successfully to {file_path}.")
        except Exception as e:
            show_error(f"Failed to export preview layer: {e}")
    # endregion


class InteractiveSegmentationWidget3DBase(InteractiveSegmentationWidgetBase):
    def __init__(self, viewer: Viewer):
        super().__init__(viewer)
        self.prompt_frame_set_view_1 = False
        self.prompt_frame_set_view_2 = False
        self.prompt_frame_set_view_3 = False

        self.prompt_frame_index_view_1 = 0
        self.prompt_frame_index_view_2 = 0
        self.prompt_frame_index_view_3 = 0

    @property
    def supported_prompt_types(self):
        return ["Mask"]  # ["Points", "BBox", "Mask"]

    def load_model(self):
        pass

    def predict(self):
        pass

    def reset_model(self):
        pass

    def setup_hyperparameter_gui(self, _layout):
        pass

    def setup_model_selection_gui(self, _scroll_layout):
        pass

    def setup_model_selection_gui(self, _scroll_layout):
        pass

    # GUI
    def setup_view_control_gui(self, _scroll_layout):
        _container, _layout = setup_vgroupbox(_scroll_layout, "View Control:")
        self.view_select = setup_hswitch(
            _layout, ["View A", "View B", "View C"], default=0, function=lambda: self.set_view())

        # Progress indicatoin
        setup_label(_layout, "Progress:")
        self.progress_indicator_1 = setup_checkbox(
            _layout, "Contour in view 1", False)
        self.progress_indicator_1.setDisabled(True)
        self.progress_indicator_2 = setup_checkbox(
            _layout, "Contour in view 2", False)
        self.progress_indicator_2.setDisabled(True)
        self.progress_indicator_3 = setup_checkbox(
            _layout, "Contour in view 3", False)
        self.progress_indicator_3.setDisabled(True)

        # Button to set the current mask as prompt
        self.set_prompt_button = setup_iconbutton(
            _layout,
            "Set Prompt",
            "check",
            self._viewer.theme,
            function=lambda: self.set_current_view_prompt(),
            tooltips="Set the current mask as prompt.",
        )

    def set_view(self):
        # Set the current view based on the selected option in the view_select widget
        selected_view = get_value(self.view_select)[1]

        print(f"Selected View: {selected_view}")

        current_view = self._viewer.dims.order[0]
        print(f"Current Order: {current_view}")

        # Update the prompt type based on the current view
        # if get_value(self.auto_set_prompt_ckbx):
        #    self.set_current_view_prompt(view=selected_view)

        if selected_view == 0:
            # Set the order of dimensions to A
            self._viewer.dims.order = (0, 1, 2)
        elif selected_view == 1:
            self._viewer.dims.order = (1, 0, 2)
        elif selected_view == 2:
            self._viewer.dims.order = (2, 0, 1)

    def set_current_view_prompt(self, view=None):
        if view is None:
            view = get_value(self.view_select)[1]

        print(f"Setting prompt for view: {view}")

        mask_prompt_layer = self.prompt_layers['mask']
        prompt_frames = self._viewer.dims.current_step
        # prompt_frames = [self.prompt_frame_index_view_1, self.prompt_frame_index_view_2, self.prompt_frame_index_view_3]
        # mask_0 = np.take(mask_prompt_layer.data,prompt_frames[view], axis=view)  # Take the mask along the current view axis

        if view == 0:
            self.prompt_frame_set_view_1 = True
            self.prompt_frame_index_view_1 = prompt_frames[0]
            set_value(self.progress_indicator_1, self.prompt_frame_set_view_1)
            self.progress_indicator_1.setText(
                f"Contour in view 1 (slice {self.prompt_frame_index_view_1})")
        elif view == 1:
            self.prompt_frame_set_view_2 = True
            self.prompt_frame_index_view_2 = prompt_frames[1]
            set_value(self.progress_indicator_2, self.prompt_frame_set_view_2)
            self.progress_indicator_2.setText(
                f"Contour in view 2 (slice {self.prompt_frame_index_view_2})")
        elif view == 2:
            self.prompt_frame_set_view_3 = True
            self.prompt_frame_index_view_3 = prompt_frames[2]
            set_value(self.progress_indicator_3, self.prompt_frame_set_view_3)
            self.progress_indicator_3.setText(
                f"Contour in view 3 (slice {self.prompt_frame_index_view_3})")

        if self.prompt_frame_set_view_1 and self.prompt_frame_set_view_2 and self.prompt_frame_set_view_3:
            self.run_button.setEnabled(True)
            if get_value(self.autorun_ckbx):
                self.run_predict_in_thread()

    def update_prompt_type(self):
        super().update_prompt_type()

    def run_predict_in_thread(self):
        super().run_predict_in_thread()

    def closeEvent(self, event=None):
        super().closeEvent(event=event)
        self.prompt_frame_set_view_1 = False
        self.prompt_frame_set_view_2 = False
        self.prompt_frame_set_view_3 = False


class InteractiveSegmentationWidget2DBase(InteractiveSegmentationWidgetBase):
    def __init__(self, viewer: Viewer):
        super().__init__(viewer)

    @property
    def supported_prompt_types(self):
        return ["Points", "BBox", "Mask"]

    def load_model(self):
        pass

    def predict(self):
        pass

    def reset_model(self):
        pass

    def setup_hyperparameter_gui(self, _layout):
        pass

    def setup_model_selection_gui(self, _scroll_layout):
        pass

    def update_prompt_type(self):
        super().update_prompt_type()
        self.run_button.setEnabled(True)

    # GUI

    def setup_view_control_gui(self, _scroll_layout):
        pass

    def closeEvent(self, event=None):
        super().closeEvent(event=event)


class InteractiveSegmentationWidgetINSTANCE(InteractiveSegmentationWidget3DBase):
    def __init__(self, viewer: Viewer):
        super().__init__(viewer)

    def setup_hyperparameter_gui(self, _layout):
        pass

    def setup_model_selection_gui(self, _scroll_layout):
        pass

    def load_model(self):
        pass

    def reset_model(self):
        pass

    def predict(self):
        pass
