import json
from pathlib import Path
import os
import threading
import time
import torch
import numpy as np
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
import traceback
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QFileDialog,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari.layers import Shapes, Points, Labels

from .base_widget import InteractiveSegmentationWidget2DBase
from ._widget_2d_sam import InteractiveSegmentationWidget2DSAM


class InteractiveSegmentationWidget2DTSAM(InteractiveSegmentationWidget2DSAM):
    def __init__(self, viewer: Viewer):
        super().__init__(viewer)
        if get_value(self.layerselect_a)[0] is not None:
            img_layer = get_value(self.layerselect_a)[0]
            N_dim = len(self._viewer.layers[img_layer].data.shape)
            self.propagation_dim_spinbox.setMaximum(N_dim-2)

    def on_image_layer_change(self):
        super().on_image_layer_change()
        if get_value(self.layerselect_a)[0] is not None:
            img_layer = get_value(self.layerselect_a)[0]
            N_dim = len(self._viewer.layers[img_layer].data.shape)
            self.propagation_dim_spinbox.setMaximum(N_dim-2)

    def setup_second_propagation_gui(self, _scroll_layout):
        _group_box, _layout = setup_vcollapsiblegroupbox(
            _scroll_layout, text="Propagation:", collapsed=False)

        _ = setup_label(
            _layout, "Propagate the masks from the current preview to the next frame.")

        self.repropagate_button = setup_iconbutton(
            None,
            "Rerun",
            "left_arrow",
            self._viewer.theme,
            function=lambda: self.run_propagate_in_thread(),
        )

        self.propagate_button = setup_iconbutton(
            None,
            "Step",
            "right_arrow",
            self._viewer.theme,
            function=lambda: self.run_propagate_in_thread(),
        )
        _ = hstack(_layout, [self.repropagate_button,
                   self.propagate_button], stretch=[1, 1, 1])

        self.propagate_status_label = setup_label(_layout, "Status: Ready")

        _label = setup_label(None, "Propagation Dim:")
        self.propagation_dim_spinbox = setup_spinbox(
            None, 1, 4, 1)

        _ = hstack(
            _layout, [_label, self.propagation_dim_spinbox], stretch=[0, 1])

        self.reverse_direction = setup_checkbox(
            _layout,
            "Reverse Direction",
            False,
            tooltips="",
            function=lambda: None
        )
        self.propagate_overwrite_cbx = setup_checkbox(
            _layout,
            "Overwrite Existing",
            False,
            tooltips="",
            function=lambda: None
        )

        pass

    def run_propagate_in_thread(self):
        """Start a background worker that runs predict() safely with a lock.

        Ensures only one prediction runs at a time. If multiple triggers occur
        while a prediction is running, a single additional rerun will be
        scheduled and reflected in the status label as "Re-run scheduled".
        """
        print("run_propagate_in_thread")

        # If a prediction is currently running, schedule a rerun and update
        # the status immediately.
        print(f"locked {self.propagating_lock.locked()}")
        if self.propagating_lock.locked():
            return

        @thread_worker
        def _worker():
            # Acquire the lock and run predictions; the loop allows a single
            # additional run if self.rerun_after_lock becomes True while
            # executing.
            with self.propagating_lock:
                try:
                    self.propagate()
                except Exception as e:
                    print(f"Error in on_prompt_update_event: {e}")
                    print(f"Traceback: {traceback.format_exc()}")

        worker = _worker()

        # Update UI when worker starts
        def _on_started():
            self.propagate_status_label.setText("Status: Running...")
        # Update UI when worker finishes or errors

        def _on_done(*args, **kwargs):
            self.propagate_status_label.setText("Status: Ready")
        # Connect signals (thread_worker exposes started, finished, errored)
        worker.started.connect(_on_started)
        worker.finished.connect(_on_done)
        worker.errored.connect(_on_done)

        worker.start()

    def propagate(self):

        img_layer = get_value(self.layerselect_a)[1]

        self.predictor.mask_threshold = get_value(self.threshold_slider)

        frames = np.transpose(
            self._viewer.layers[img_layer].data, self._viewer.dims.order)

        if len(frames.shape) != 3:
            show_warning("The selected image layer is not 2D+t.")
            return

        current_frame_idx = self._viewer.dims.current_step[self._viewer.dims.order[0]]

        next_frame_idx = current_frame_idx + \
            1 if not get_value(
                self.reverse_direction) else current_frame_idx - 1

        if next_frame_idx >= self._viewer.layers[img_layer].data.shape[self._viewer.dims.order[0]] or next_frame_idx < 0:
            show_warning("No more frames to propagate to.")
            return

        current_frame = frames[current_frame_idx]
        current_frame = cv2.normalize(current_frame, None, alpha=0,
                                      beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        current_frame = cv2.cvtColor(current_frame, cv2.COLOR_GRAY2RGB)
        next_frame = frames[next_frame_idx]
        next_frame = cv2.normalize(next_frame, None, alpha=0,
                                   beta=255, norm_type=cv2.NORM_MINMAX).astype(np.uint8)
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_GRAY2RGB)

        try:
            self.predictor.reset_state()
        except:
            pass

        self.predictor.load_first_frame(current_frame)

        if self.preview_layer is None:
            show_warning(
                "No preview layer found. Please run prediction first.")
            return

        transposed_preview_layer_data = np.transpose(
            self.preview_layer.data, self._viewer.dims.order)

        current_mask = transposed_preview_layer_data[current_frame_idx]

        if np.sum(current_mask) == 0:
            show_warning(
                "No mask found in the current frame of the preview layer. Please run prediction first.")
            return

        current_mask = current_mask
        visibile_object_ids = np.unique(current_mask)
        visibile_object_ids = visibile_object_ids[visibile_object_ids != 0]

        if len(visibile_object_ids) == 0:
            show_warning(
                "No object found in the current frame of the preview layer. Please run prediction first.")
            return
        print(visibile_object_ids)
        for object_id in visibile_object_ids:
            obj_mask = (current_mask == object_id).astype(np.uint8)
            _, out_obj_ids, out_mask_logits = self.predictor.add_new_mask(
                frame_idx=0, obj_id=object_id, mask=obj_mask)

        out_obj_ids, out_mask_logits = self.predictor.track(next_frame)

        out_mask_masks = (
            out_mask_logits > self.predictor.mask_threshold).cpu().numpy().astype(np.uint8)

        current_next_mask = transposed_preview_layer_data[next_frame_idx]
        visibile_object_ids_in_next = np.unique(current_next_mask)
        visibile_object_ids_in_next = visibile_object_ids_in_next[visibile_object_ids_in_next != 0]
        print(visibile_object_ids_in_next)
        updated_next_mask = np.zeros_like(current_next_mask)
        # Write the new masks to the next frame, respecting the overwrite setting
        for i, object_id in enumerate(visibile_object_ids):
            if object_id in visibile_object_ids_in_next and not get_value(self.propagate_overwrite_cbx):
                updated_next_mask[current_next_mask == object_id] = object_id
                continue
            print(out_mask_masks[i, 0].sum())
            next_object_mask = (
                out_mask_masks[i, 0] * object_id).astype(np.uint8)
            # only write to pixels that are not already occupied if not overwriting
            # -> lower ids have priority
            np.copyto(updated_next_mask, next_object_mask, where=(
                (next_object_mask > 0) & (updated_next_mask == 0)))

        np.copyto(
            transposed_preview_layer_data[next_frame_idx], updated_next_mask)
        self.preview_label_data = self.preview_layer.data.copy()
        self._viewer.dims.set_current_step(
            self._viewer.dims.order[0], next_frame_idx)
        self.preview_layer.refresh()
        # go to next frame
