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
from napari.qt.threading import thread_worker

import traceback

from .base_widget import InteractiveSegmentationWidget3DBase


class InteractiveSegmentationWidget3DNNI(InteractiveSegmentationWidget3DBase):
    def __init__(self, viewer: Viewer):
        super().__init__(viewer)
        self.run_button.setEnabled(True)

    def setup_hyperparameter_gui(self, _layout):
        pass

    def setup_model_selection_gui(self, _scroll_layout):
        pass
        # log view
        _container, _layout = setup_vcollapsiblegroupbox(
            _scroll_layout, "Setup", True)

        # _container, _layout = setup_vgroupbox(_scroll_layout, "Metric mode:")
        _ = setup_label(_layout, "Select model:")

        # ,"SAM2.1 tiny", "SAM2.1 small", "SAM2.1 large"]
        model_options = ["nnInteractive_v1.0"]

        self.model_selection = setup_combobox(
            _layout, options=model_options, function=lambda: print(
                "Model Selection Changed")
        )

        _ = setup_iconbutton(
            _layout, "Initialize", "new_labels", self._viewer.theme, lambda: self.on_model_change()
        )

    def on_model_change(self):
        self.reset_model()
        # self.load_model()

    def load_model(self):
        try:
            # Install huggingface_hub if not already installed
            from huggingface_hub import snapshot_download
            REPO_ID = "nnInteractive/nnInteractive"
            MODEL_NAME = "nnInteractive_v1.0"  # Updated models may be available in the future
            DOWNLOAD_DIR = "./nnInteractive"  # Specify the download directory
            download_path = snapshot_download(
                repo_id=REPO_ID,
                allow_patterns=[f"{MODEL_NAME}/*"],
                local_dir=DOWNLOAD_DIR
            )
            from nnInteractive.inference.inference_session import nnInteractiveInferenceSession
            self.session = nnInteractiveInferenceSession(
                device=torch.device("cuda"),  # Set inference device
                use_torch_compile=False,  # Experimental: Not tested yet
                verbose=False,
                torch_n_threads=2,  # Use available CPU cores
                do_autozoom=True,  # Enables AutoZoom for better patching
                use_pinned_memory=True,  # Optimizes GPU memory transfers
            )

            # Load the trained model
            model_path = os.path.join(DOWNLOAD_DIR, MODEL_NAME)
            self.session.initialize_from_trained_model_folder(model_path)
            print(self.session)
        except Exception as e:
            self.session = None
            print(f"Failed to load model. {e}")

            show_warning(f"Failed to load model. {e}")
            show_warning("Empty preditions will be returned.")
            return

    def reset_model(self):
        self.session.reset_interactions()

    def predict(self):
        try:
            prompt_type = get_value(self.prompt_type_select)[0]
            img_layer = get_value(self.layerselect_a)[1]

            # mask_threshold = get_value(self.threshold_slider)

            img_data = self._viewer.layers[img_layer].data.astype(np.float32)
            # normalize the image data to 0-255 range if it's not already
            img_data = (img_data - np.min(img_data)) / \
                (np.max(img_data) - np.min(img_data)) * 255
            # img_data = img_data.astype(np.uint8)
            print(self.session)
            if prompt_type == "Mask":
                mask_prompt_layer = self.prompt_layers['mask']
                # prompt_frames = self._viewer.dims.current_step
                prompt_frames = [self.prompt_frame_index_view_1,
                                 self.prompt_frame_index_view_2, self.prompt_frame_index_view_3]
                scale_factors = self._viewer.layers[img_layer].scale
                # invert prompt frames where scale factor is negative to shape - frame
                # for i in range(3):
                #    if scale_factors[i] < 0:
                #        prompt_frames[i] = img_data.shape[i] - 1 - prompt_frames[i]
                # in a single line
                prompt_frames = [img_data.shape[i] - 1 - prompt_frames[i]
                                 if scale_factors[i] < 0 else prompt_frames[i] for i in range(3)]
                mask_1 = mask_prompt_layer.data[prompt_frames[0]]
                mask_2 = mask_prompt_layer.data[:, prompt_frames[1]]
                mask_3 = mask_prompt_layer.data[:, :, prompt_frames[2]]

                print("USING NNI")
                from scipy.ndimage import binary_dilation
                self.session.set_image(img_data[None])
                target_tensor = torch.zeros(
                    img_data.shape, dtype=torch.uint8)  # Must be 3D (x, y, z)
                self.session.set_target_buffer(target_tensor)
                if self.prompt_frame_index_view_1 != 0:
                    lasso = np.zeros(target_tensor.shape, dtype=np.uint8)
                    lasso[prompt_frames[0]] = binary_dilation(
                        mask_prompt_layer.data[prompt_frames[0]], iterations=1).astype(np.uint8)
                    self.session.add_lasso_interaction(
                        lasso, include_interaction=True)
                if self.prompt_frame_index_view_2 != 0:
                    lasso = np.zeros(target_tensor.shape, dtype=np.uint8)
                    lasso[:, prompt_frames[1]] = binary_dilation(
                        mask_prompt_layer.data[:, prompt_frames[1]], iterations=1).astype(np.uint8)
                    self.session.add_lasso_interaction(
                        lasso, include_interaction=True)
                if self.prompt_frame_index_view_3 != 0:
                    lasso = np.zeros(target_tensor.shape, dtype=np.uint8)
                    lasso[:, :, prompt_frames[2]] = binary_dilation(
                        mask_prompt_layer.data[:, :, prompt_frames[2]], iterations=1).astype(np.uint8)
                    self.session.add_lasso_interaction(
                        lasso, include_interaction=True)

                results = self.session.target_buffer.clone()
                self.session.reset_interactions()
                pred = results.cpu().numpy()
                out_mask_masks = pred.astype(np.uint8)  # Convert to uint8

            # if get_value(self.multi_mask_ckbx) and get_value(self.scoring_ckbx):
            #    # If multi-mask and scoring are enabled, we will have multiple masks
            #    out_mask = out_mask_masks[np.argmax(out_mask_scores)]  # Select the mask with the highest score
            # else:

            # np.zeros_like(self.preview_layer.data, dtype=np.uint8)
            self.preview_layer.data = out_mask_masks

            self.preview_layer.refresh()
        except Exception as e:
            print(f"Error in on_prompt_update_event: {e}")
            print(f"Traceback: {traceback.format_exc()}")
