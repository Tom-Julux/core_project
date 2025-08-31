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

from napari.layers import Shapes, Points, Labels

from .base_widget import InteractiveSegmentationWidget3DBase

class InteractiveSegmentationWidgetSAM2(InteractiveSegmentationWidget3DBase):
    def __init__(self, viewer: Viewer):
        super().__init__(viewer)
        
    def setup_hyperparameter_gui(self, _layout):
        _ = setup_label(_layout, "Threshold:")
        self.threshold_slider = setup_editdoubleslider(
            _layout, 2, -3, 3.0, 0.5, function=lambda: self.update_hyperparameters(), include_buttons=False
        )
        pass

    def setup_model_selection_gui(self, _scroll_layout):
        pass
        # log view
        _container, _layout = setup_vcollapsiblegroupbox(_scroll_layout, "Setup", True)

        #_container, _layout = setup_vgroupbox(_scroll_layout, "Metric mode:")
        _ = setup_label(_layout, "Select model:")
        
        model_options = ["MedSAM2-May2025"]#,"SAM2.1 tiny", "SAM2.1 small", "SAM2.1 large"]

        self.model_selection = setup_combobox(
            _layout, options=model_options, function=lambda: print("Model Selection Changed")
        )
      
        _ = setup_iconbutton(
            _layout, "Initialize", "new_labels", self._viewer.theme, lambda: self.on_model_change()
        )

    def on_model_change(self):
        self.reset_model()
        self.load_model()

    def load_model(self):
        try:
            from sam2.build_sam import build_sam2_camera_predictor,build_sam2
            from sam2.sam2_camera_predictor import SAM2CameraPredictor

            checkpoint = "/app/MedSAM2_latest.pt"
            model_cfg= "configs/sam2.1_hiera_t512.yaml"
            self.predictor = build_sam2_camera_predictor(model_cfg, checkpoint, device="cuda")
            set_value(self.threshold_slider, self.predictor.mask_threshold)
        except Exception as e:
            self.predictor = None

            show_warning(f"Failed to load model. {e}")
            show_warning("Empty preditions will be returned.")
            return

    def reset_model(self):
        pass

    def predict(self):
        try:
            prompt_type = get_value(self.prompt_type_select)[0]
            img_layer = get_value(self.layerselect_a)[1]

            mask_threshold = get_value(self.threshold_slider)

            img_data = self._viewer.layers[img_layer].data.astype(np.float32)
            # normalize the image data to 0-255 range if it's not already
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255
            img_data = img_data.astype(np.uint8)

            if prompt_type == "Mask":
                mask_prompt_layer = self.prompt_layers['mask']
                prompt_frames = self._viewer.dims.current_step
                print(prompt_frames)
                print(self.prompt_frame_index_view_1, self.prompt_frame_index_view_2, self.prompt_frame_index_view_3)
                prompt_frames = [self.prompt_frame_index_view_1, self.prompt_frame_index_view_2, self.prompt_frame_index_view_3]
                mask_1 = mask_prompt_layer.data[prompt_frames[0]]
                mask_2 = mask_prompt_layer.data[:,prompt_frames[1]]
                mask_3 = mask_prompt_layer.data[:,:,prompt_frames[2]]
                
                if self.predictor is None:
                    # expand the mask to match the shape of the image data
                    mask_1 = np.repeat(mask_1[None,...], img_data.shape[0], axis=0)
                    mask_2 = np.repeat(mask_2[:,None,:], img_data.shape[1], axis=1)
                    mask_3 = np.repeat(mask_3[:,:,None], img_data.shape[2], axis=2)
            
                    union_mask = np.logical_or(mask_1, mask_2)
                    union_mask = np.logical_or(union_mask, mask_3)
                    print(self.preview_layer.data.shape)
                    print(mask_1.shape, mask_2.shape, mask_3.shape, union_mask.shape)
                    out_mask_masks = union_mask.astype(np.uint8)  # Convert to uint8
                else: 
                    print("USING SAM2")
                    from napari_interactive.sam2_utils import propagate_along_path, merge_results, view_1_to_view_2, view_1_to_view_3, mask_view_2_to_view_1, center_of_mass, mask_view_3_to_view_1
                    volume_labels = mask_prompt_layer.data
                    volume_data = img_data
                    sam_mask_threshold = self.predictor.mask_threshold
                    center_slice = prompt_frames[0]

                    #initial_point = np.array([center_slice, point[0],point[1]], dtype=np.float32)
                    #point_prompts = np.concatenate([point_prompts, np.array([[volume_data.shape[1]/2, volume_data.shape[2]/2]], dtype=np.float32)], axis=0)
                    #point_prompts, point_labels = add_negative_point_prompts(point_prompts, point_labels, volume_labels, d=negative_point_distance)
                    #cha forward in space
                    results_1 = merge_results(  propagate_along_path(volume_data[:center_slice+1][::-1], self.predictor,threshold=sam_mask_threshold, keep_logits=True, reset_state=True, initialization="mask", mask_prompt=volume_labels[center_slice]),\
                                                propagate_along_path(volume_data[center_slice:], self.predictor,threshold=sam_mask_threshold, keep_logits=True, reset_state=True, initialization="mask", mask_prompt=volume_labels[center_slice]))
                    # do propagation
                    volume_data_view2 = view_1_to_view_2(volume_data)
                    volume_labels_view2 = view_1_to_view_2(volume_labels)
                    center_slice_2 = prompt_frames[1]
                    results_2 = merge_results(  propagate_along_path(volume_data_view2[:center_slice_2+1][::-1], self.predictor,threshold=sam_mask_threshold, keep_logits=True, reset_state=True, initialization="mask", mask_prompt=volume_labels_view2[center_slice_2]),\
                                                propagate_along_path(volume_data_view2[center_slice_2:], self.predictor,threshold=sam_mask_threshold, keep_logits=True, reset_state=True,  initialization="mask", mask_prompt=volume_labels_view2[center_slice_2]))
                    results_2["masks"] = mask_view_2_to_view_1(results_2["masks"])
                    # view 3
                    volume_data_view3 = view_1_to_view_3(volume_data)
                    volume_labels_view3 = view_1_to_view_3(volume_labels).copy()
                    center_slice_3 = prompt_frames[2]
                    results_3 = merge_results(  propagate_along_path(volume_data_view3[:center_slice_3+1][::-1], self.predictor,threshold=sam_mask_threshold, keep_logits=True, reset_state=True, initialization="mask", mask_prompt=volume_labels_view3[center_slice_3]),\
                                                propagate_along_path(volume_data_view3[center_slice_3:], self.predictor,threshold=sam_mask_threshold, keep_logits=True, reset_state=True, initialization="mask", mask_prompt=volume_labels_view3[center_slice_3]))
                    results_3["masks"] = mask_view_3_to_view_1(results_3["masks"])
                    m1 = results_1["masks"]
                    m2 = results_2["masks"]
                    m3 = results_3["masks"]
                    #m_threshold2 = (np.array([m1.sum(), m2.sum(), m3.sum()])>20).sum()
                    union = m1
                    #union = np.sum([m1, m2, m3], axis=0) >= (m_threshold-1 if m_threshold>1 else 1)
                    out_mask_masks = union.astype(np.uint8)  # Convert to uint8

                # apply the model

                out_mask_masks = out_mask_masks  # Convert to numpy array

            #if get_value(self.multi_mask_ckbx) and get_value(self.scoring_ckbx):
            #    # If multi-mask and scoring are enabled, we will have multiple masks
            #    out_mask = out_mask_masks[np.argmax(out_mask_scores)]  # Select the mask with the highest score
            #else:
            
            self.preview_layer.data = out_mask_masks#np.zeros_like(self.preview_layer.data, dtype=np.uint8)

            self.preview_layer.refresh()   
        except Exception as e:
            print(f"Error in on_prompt_update_event: {e}")
            print(f"Traceback: {traceback.format_exc()}") 