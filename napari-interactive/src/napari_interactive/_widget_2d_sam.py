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
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QFileDialog,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls

from napari_promptable.controls.bbox_controls import CustomQtBBoxControls
from napari_promptable.controls.lasso_controls import CustomQtLassoControls
from napari_promptable.controls.point_controls import CustomQtPointsControls
from napari_promptable.controls.scribble_controls import CustomQtScribbleControls
from napari.layers import Shapes, Points, Labels

from .base_widget import InteractiveSegmentationWidget2DBase

class InteractiveSegmentationWidget2DSAM(InteractiveSegmentationWidget2DBase):
    def __init__(self, viewer: Viewer):
        super().__init__(viewer)
        set_value(self.run_ckbx, True) 

    @property
    def supported_prompt_types(self):
        return ["Points", "BBox", "Mask"]

    def load_model(self):
        from sam2.build_sam import build_sam2_camera_predictor,build_sam2
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        checkpoint = "/app/MedSAM2_latest.pt"
        if not os.path.exists("/app/MedSAM2_latest.pt"):
            base_path = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
            checkpoint = os.path.join(base_path, "MedSAM2_latest.pt")

        model_cfg= "configs/sam2.1_hiera_t512.yaml"
        self.predictor = SAM2ImagePredictor(build_sam2(model_cfg, checkpoint, device="cuda" if torch.cuda.is_available() else "cpu"))
        set_value(self.threshold_slider, self.predictor.mask_threshold)
        
    def setup_hyperparameter_gui(self, _layout):
        _ = setup_label(_layout, "Threshold:")
        self.threshold_slider = setup_editdoubleslider(
            _layout, 2, -3, 3.0, 0.5, function=lambda: self.update_hyperparameters(), include_buttons=False
        )
        pass

    def predict(self):
        try:
            prompt_type = get_value(self.prompt_type_select)[0]
            img_layer = get_value(self.layerselect_a)[1]
            self.predictor.mask_threshold = get_value(self.threshold_slider)

            current_frame_idx = self._viewer.dims.current_step[self._viewer.dims.order[0]]
            
            frame = np.transpose(self._viewer.layers[img_layer].data, self._viewer.dims.order)[current_frame_idx]
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            #self.predictor.reset_state()
            self.predictor.set_image(frame)

            if prompt_type == "Points":
                # Get the positive and negative point layers
                point_layer_positive = self.prompt_layers['point_positive']
                point_layer_negative = self.prompt_layers['point_negative']

                if len(point_layer_positive.data) == 0:
                    return
                # Get the data from the positive and negative point layers
                pos_points = point_layer_positive.data[:, self._viewer.dims.order[::-1]][:, :2]
                neg_points = point_layer_negative.data[:, self._viewer.dims.order[::-1]][:, :2]

                point_prompts = np.concatenate((pos_points, neg_points), axis=0)  # Use only the first two dimensions (x, y)
                point_labels = np.concatenate((np.ones(len(pos_points)), np.zeros(len(neg_points))), axis=0)

                #_, out_obj_ids, out_mask_logits = self.predictor.add_new_points(frame_idx=0, obj_id=0, points=point_prompts, labels=point_labels)
                out_mask_masks, out_mask_scores, out_mask_logits = self.predictor.predict(point_coords=point_prompts, point_labels=point_labels, multimask_output=get_value(self.multi_mask_ckbx))  # Use the multi-mask option if checked


            elif prompt_type == "BBox":
                bbox_layer = self.prompt_layers['bbox']
                if len(bbox_layer.data) == 0:
                    return
                bbox_data = bbox_layer.data[-1][:, self._viewer.dims.order][:,1:]

                bbox_prompt = np.array([
                    np.min(bbox_data[:, 1]), np.min(bbox_data[:, 0]),
                    np.max(bbox_data[:, 1]), np.max(bbox_data[:, 0])
                ])
                out_mask_masks, out_mask_scores, out_mask_logits = self.predictor.predict(box=bbox_prompt) # XYXY format
     
            #if get_value(self.multi_mask_ckbx) and get_value(self.scoring_ckbx):
            #    # If multi-mask and scoring are enabled, we will have multiple masks
            #    out_mask = out_mask_masks[np.argmax(out_mask_scores)]  # Select the mask with the highest score
            #else:
            
            out_mask = out_mask_masks[-1]
            # Apply connected component analysis to the mask
            #if get_value(self.connected_component_ckbx):
            #    num_labels, labels_im = cv2.connectedComponents(out_mask.astype(np.uint8), connectivity=8)
            #    out_mask = np.zeros_like(out_mask, dtype=np.uint8)
            #    if num_labels > 1:
            #        largest_label = np.argmax(np.bincount(labels_im.flat)[1:]) + 1  # Skip the background label (0)
            #        out_mask[labels_im == largest_label] = 1

            target_size = frame.shape[:2]
            
            out_mask = cv2.resize(out_mask.astype(np.uint8), target_size[::-1], interpolation=cv2.INTER_NEAREST)

            np.transpose(self.preview_layer.data, self._viewer.dims.order)[current_frame_idx] = out_mask
            self.preview_layer.refresh()   
        except Exception as e:
            print(f"Error in on_prompt_update_event: {e}")