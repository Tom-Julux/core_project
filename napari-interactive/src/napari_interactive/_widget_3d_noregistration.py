import json
from pathlib import Path
import threading
import time
import torch
import os
import cv2
from magicgui import magicgui
from napari.layers import Image
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
import traceback

from .base_widget import InteractiveSegmentationWidget3DBase


class InteractiveSegmentationWidget3DNoRegistration(InteractiveSegmentationWidget3DBase):
    def __init__(self, viewer: Viewer):
        super().__init__(viewer)

    def setup_hyperparameter_gui(self, _layout):
        pass

    def setup_model_selection_gui(self, _scroll_layout):
        pass

    def on_model_change(self):
        self.reset_model()
        self.load_model()

    def load_model(self):
        pass

    def reset_model(self):
        pass

    def predict(self):
        try:
            prompt_type = get_value(self.prompt_type_select)[0]
            img_layer = get_value(self.layerselect_a)[1]

            mask_threshold = get_value(self.threshold_slider)

            img_data = self._viewer.layers[img_layer].data.astype(np.float32)
            # normalize the image data to 0-255 range if it's not already
            img_data = (img_data - np.min(img_data)) / \
                (np.max(img_data) - np.min(img_data)) * 255
            img_data = img_data.astype(np.uint8)

            if prompt_type == "Mask":
                mask_prompt_layer = self.prompt_layers['mask']
                prompt_frames = self._viewer.dims.current_step

                prompt_frames = [self.prompt_frame_index_view_1,
                                 self.prompt_frame_index_view_2, self.prompt_frame_index_view_3]
                scale_factors = self._viewer.layers[img_layer].scale

                prompt_frames = [img_data.shape[i] - 1 - prompt_frames[i]
                                 if scale_factors[i] < 0 else prompt_frames[i] for i in range(3)]

                def save_preview(img, mask, filename):
                    img = cv2.cvtColor(img, cv2.COLOR_GRAY2BGR)
                    contour, _ = cv2.findContours(mask.astype(
                        np.uint8), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    cv2.drawContours(img, contour, -1, (0, 255, 0), 2)
                    cv2.imwrite(filename, img)

                mask_1 = mask_prompt_layer.data[prompt_frames[0]]
                mask_2 = mask_prompt_layer.data[:, prompt_frames[1]]
                mask_3 = mask_prompt_layer.data[:, :, prompt_frames[2]]

                save_preview(
                    img_data[prompt_frames[0]], mask_prompt_layer.data[prompt_frames[0]], "view1.png")
                save_preview(img_data[:, prompt_frames[1]],
                             mask_prompt_layer.data[:, prompt_frames[1]], "view2.png")
                save_preview(img_data[:, :, prompt_frames[2]],
                             mask_prompt_layer.data[:, :, prompt_frames[2]], "view3.png")

                # expand the mask to match the shape of the image data
                mask_1 = np.repeat(mask_1[None, ...],
                                   img_data.shape[0], axis=0)
                mask_2 = np.repeat(
                    mask_2[:, None, :], img_data.shape[1], axis=1)
                mask_3 = np.repeat(
                    mask_3[:, :, None], img_data.shape[2], axis=2)

                union_mask = np.logical_or(mask_1, mask_2)
                union_mask = np.logical_or(union_mask, mask_3)
                print(self.preview_layer.data.shape)
                print(mask_1.shape, mask_2.shape,
                      mask_3.shape, union_mask.shape)
                out_mask_masks = union_mask.astype(
                    np.uint8)  # Convert to uint8

            # np.zeros_like(self.preview_layer.data, dtype=np.uint8)
            self.preview_layer.data = out_mask_masks
            self.preview_layer.refresh()
        except Exception as e:
            print(f"Error in on_prompt_update_event: {e}")
            print(f"Traceback: {traceback.format_exc()}")
