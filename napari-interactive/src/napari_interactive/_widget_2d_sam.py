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

class InteractiveSegmentationWidget2DSAM(InteractiveSegmentationWidget2DBase):
    def __init__(self, viewer: Viewer):
        super().__init__(viewer)
        set_value(self.autorun_ckbx, True)

    @property
    def supported_prompt_types(self):
        return ["Points", "BBox", "Mask", "Manual"]

    def load_model(self):
        from sam2.build_sam import build_sam2_camera_predictor, build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = "/app/MedSAM2_latest.pt"
        if not os.path.exists("/app/MedSAM2_latest.pt"):
            base_path = os.path.dirname(os.path.dirname(
                os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))
            checkpoint = os.path.join(base_path, "MedSAM2_latest.pt")

        model_cfg = "configs/sam2.1_hiera_t512.yaml"
        self.predictor = build_sam2_camera_predictor(
            model_cfg, checkpoint, device="cuda" if torch.cuda.is_available() else "cpu")
        set_value(self.threshold_slider, self.predictor.mask_threshold)

    def setup_hyperparameter_gui(self, _layout):
        _ = setup_label(_layout, "Threshold:")
        self.threshold_slider = setup_editdoubleslider(
            _layout, 2, -3, 3.0, 0.5, function=lambda: self.on_hyperparameter_update(), include_buttons=False
        )
        pass

    def predict(self):
        prompt_type = get_value(self.prompt_type_select)[0]
        img_layer = get_value(self.layerselect_a)[1]
        self.predictor.mask_threshold = get_value(self.threshold_slider)

        # current_frame_idx = self._viewer.dims.current_step[self._viewer.dims.order[0]]
        # [current_frame_idx]
        frame = np.transpose(
            self._viewer.layers[img_layer].data, self._viewer.dims.order)

        if len(frame.shape) == 1:
            show_warning("The selected image layer is not at least 2D.")
            return
        if len(frame.shape) == 3:
            frame = frame[self._viewer.dims.current_step[self._viewer.dims.order[0]]]
        if len(frame.shape) == 4:
            frame = frame[self._viewer.dims.current_step[self._viewer.dims.order[0]],
                          self._viewer.dims.current_step[self._viewer.dims.order[1]]]

        # normalize to 0-255 and convert to uint8
        frame = cv2.normalize(frame, None, alpha=0,
                              beta=255, norm_type=cv2.NORM_MINMAX)
        frame = frame.astype(np.uint8)

        frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

        try:
            self.predictor.reset_state()
        except:
            pass
        self.predictor.load_first_frame(frame)

        if prompt_type == "Points":
            # Get the positive and negative point layers
            point_layer_positive = self.prompt_layers['point_positive']
            point_layer_negative = self.prompt_layers['point_negative']

            if len(point_layer_positive.data) == 0:
                return
            # Get the data from the positive and negative point layers
            pos_points = point_layer_positive.data[:,
                                                   self._viewer.dims.order[::-1]]
            neg_points = point_layer_negative.data[:,
                                                   self._viewer.dims.order[::-1]]
            if len(frame.shape) == 3:
                pos_points = pos_points[pos_points[:, 2] ==
                                        self._viewer.dims.current_step[self._viewer.dims.order[0]]]
                neg_points = neg_points[neg_points[:, 2] ==
                                        self._viewer.dims.current_step[self._viewer.dims.order[0]]]
            if len(frame.shape) == 4:
                pos_points = pos_points[pos_points[:, 3] ==
                                        self._viewer.dims.current_step[self._viewer.dims.order[1]]]
                neg_points = neg_points[neg_points[:, 3] ==
                                        self._viewer.dims.current_step[self._viewer.dims.order[1]]]

            # Use only the first two dimensions (x, y)
            pos_points = pos_points[:, :2]
            neg_points = neg_points[:, :2]

            # combine
            point_prompts = np.concatenate(
                (pos_points, neg_points), axis=0)
            point_labels = np.concatenate(
                (np.ones(len(pos_points)), np.zeros(len(neg_points))), axis=0)

            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0, obj_id=0,
                points=point_prompts, labels=point_labels)
            out_mask_masks = (out_mask_logits > self.predictor.mask_threshold)
            out_mask_masks = out_mask_masks[0,
                                            0].cpu().numpy().astype(np.uint8)

        elif prompt_type == "BBox":
            bbox_layer = self.prompt_layers['bbox']
            if len(bbox_layer.data) == 0:
                return

            # Remove all but the last bbox
            with self.no_autopredict():
                bbox_layer.data = bbox_layer.data[-1:]
                bbox_layer.refresh()

            bbox_data = bbox_layer.data[-1][:, self._viewer.dims.order[::-1]]

            bbox_prompt = np.array([
                np.min(bbox_data[:, 0]), np.min(bbox_data[:, 1]),
                np.max(bbox_data[:, 0]), np.max(bbox_data[:, 1])
            ])

            bbox_prompt[0] = np.maximum(bbox_prompt[0], 0)
            bbox_prompt[1] = np.maximum(bbox_prompt[1], 0)
            bbox_prompt[2] = np.minimum(bbox_prompt[2], frame.shape[1])
            bbox_prompt[3] = np.minimum(bbox_prompt[3], frame.shape[0])

            _, out_obj_ids, out_mask_logits = self.predictor.add_new_prompt(
                frame_idx=0, obj_id=0,
                bbox=bbox_prompt)

            out_mask_masks = (out_mask_logits > self.predictor.mask_threshold)

            out_mask_masks = out_mask_masks[0,0].cpu().numpy().astype(np.uint8)

            # out_mask_masks, out_mask_scores, out_mask_logits = self.predictor.predict(
            #    box=bbox_prompt)  # XYXY format
        elif prompt_type == "Mask":
            show_warning("Mask prompt does not make much sense for 2D.")
            # use mask prompt as initialization
            mask_prompt_layer = self.prompt_layers['mask']
            if len(mask_prompt_layer.data) == 0:
                return

            prompt_frame = self._viewer.dims.current_step[self._viewer.dims.order[0]]
            mask_prompt = mask_prompt_layer.data[prompt_frame]
            out_mask_masks = mask_prompt
        else:
            # Manual Prompt 
            return
        out_mask = out_mask_masks > 0

        target_size = frame.shape[:2]
        out_mask = out_mask.astype(np.uint8)

        self.add_prediction_to_preview(
            out_mask, np.s_[self._viewer.dims.current_step[self._viewer.dims.order[0]]])
