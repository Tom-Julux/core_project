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
import traceback

from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls

from napari_promptable.controls.bbox_controls import CustomQtBBoxControls
from napari_promptable.controls.lasso_controls import CustomQtLassoControls
from napari_promptable.controls.point_controls import CustomQtPointsControls
from napari_promptable.controls.scribble_controls import CustomQtScribbleControls
from napari.layers import Shapes, Points, Labels

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
layer_to_controls[ContourPromptLayer] = CustomQtLassoControls

class InteractiveSegmentationWidgetSAM2(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        # layers managed by this widget
        self.prompt_layers = {}
        self.preview_layer = None  # Layer to show the preview of the segmentation
        self.preview_label_data = None  # Data Label for the preview layer
        

        self.propagating_lock = threading.Lock()

        self.build_gui()
        self.load_model()

    # GUI
    def build_gui(self):
       
        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)
        self._scroll_layout = _scroll_layout
        # log view
        _container, _layout = setup_vcollapsiblegroupbox(_scroll_layout, "Setup", True)

        #_container, _layout = setup_vgroupbox(_scroll_layout, "Metric mode:")
        _ = setup_label(_layout, "Select model:")
        
        model_options = ["MedSAM2-May2025","SAM2.1 tiny", "SAM2.1 small", "SAM2.1 large"]

        self.model_selection = setup_combobox(
            _layout, options=model_options, function=lambda: print("Model Selection Changed")
        )
      
        _ = setup_iconbutton(
            _layout, "Initialize", "new_labels", self._viewer.theme, self.load_model
        )


        _container, _layout = setup_vgroupbox(_scroll_layout, "Image Selection:")
        # layer select for image layer
        self.layerselect_a = setup_layerselect(
            _layout, self._viewer, Image, function=lambda: self.predict()
        )


        _container, _layout = setup_vgroupbox(_scroll_layout, "Prompt Selection:")

        self.prompt_type_select = setup_combobox(
            _layout, ["Points", "BBox", "Mask"],\
                 "QComboBox", function=lambda: self.update_prompt_type()
        )


        _container, _layout = setup_vgroupbox(_scroll_layout, "Hyperparameters:")
        _ = setup_label(_layout, "Threshold:")
        self.threshold_slider = setup_editdoubleslider(
            _layout, 2, -3, 3.0, 0.5, function=lambda: self.update_hyperparameters(), include_buttons=False
        )
        
        #self.scoring_ckbx = setup_checkbox(_layout, "Use Scoring", True, function=lambda: self.predict())
        #self.connected_component_ckbx = setup_checkbox(_layout, "Connected Component", True, function=lambda: self.predict())
        
        _group_box, _layout = setup_vcollapsiblegroupbox(_scroll_layout, text="Propagation:", collapsed=False)

        self.run_button = setup_iconbutton(
            _layout,
            "Initialize",
            "right_arrow",
            self._viewer.theme,
            function=lambda: self.run_predict_in_thread(),
            tooltips="Run the predict step",
        )
        self.run_ckbx = setup_checkbox(
            _layout,
            "Auto Run Prediction",
            False,
            tooltips="Run automatically after each interaction",
        )
        self.run_button = setup_iconbutton(
            _layout,
            "Step",
            "right_arrow",
            self._viewer.theme,
            function=lambda: self.run_propagate_in_thread(),
        )

        self.reset_button = setup_iconbutton(
            _layout,
            "Reset",
            "erase",
            self._viewer.theme,
            function=lambda: self.reset()
        )
        #self.run_ckbx = setup_checkbox(
        #    _layout,
        #    "Auto Run Prediction",
        #    False,
        #    tooltips="Run automatically after each interaction",
        #)

        _container, _layout = setup_vgroupbox(_scroll_layout, "Export")
        _ = setup_label(_layout, "Export the contents of the preview layer to a sperate layer.")
        _ = setup_iconbutton(
            _layout, "Export", "pop_out", self._viewer.theme, self.export_preview
        )
        self.update_prompt_type()
        
    def load_model(self):
        from sam2.build_sam import build_sam2_camera_predictor,build_sam2
        from sam2.sam2_camera_predictor import SAM2CameraPredictor

        checkpoint = "/Users/tomjulius/Developer/napari-plugins/napari-interactive/src/MedSAM2_latest.pt"
        model_cfg= "configs/sam2.1_hiera_t512.yaml"
        self.predictor = build_sam2_camera_predictor(model_cfg, checkpoint, device="mps")
        set_value(self.threshold_slider, self.predictor.mask_threshold)

    def prepare_preview_layer(self):
        img_layer = get_value(self.layerselect_a)[1]

        img_layer_shape = self._viewer.layers[img_layer].data.shape
        self.preview_label_data = np.zeros(img_layer_shape, dtype=np.uint8)
        self.preview_layer = Labels(name='Preview Layer', data=self.preview_label_data, opacity=0.5)

        self._viewer.add_layer(self.preview_layer)

    def clear_prompt_layers(self):
        # Remove all existing prompt layers
        for layer in self.prompt_layers.values():
            if layer in self._viewer.layers:
                self._viewer.layers.remove(layer)
        self.prompt_layers.clear()

    def update_prompt_type(self):
        prompt_type = get_value(self.prompt_type_select)[0]
        
        self.clear_prompt_layers()

        img_layer = get_value(self.layerselect_a)[1]
        print(f"Image Layer: {img_layer}")
        img_layer_shape = self._viewer.layers[img_layer].data.shape
        self.preview_label_data = np.zeros(img_layer_shape, dtype=np.uint8)
        self.preview_layer = Labels(name='Preview Layer', data=self.preview_label_data, opacity=0.5)
        self._viewer.add_layer(self.preview_layer)

        if prompt_type == "Points":
            point_layer_positive = PointPromptLayer(name='Point Point Layer (Positive)', ndim=3)
            point_layer_negative = PointPromptLayer(name='Point Point Layer (Negative)', ndim=3)
            self._viewer.add_layer(point_layer_positive)
            self._viewer.add_layer(point_layer_negative)
            self.prompt_layers['point_positive'] = point_layer_positive
            self.prompt_layers['point_negative'] = point_layer_negative

            point_layer_positive.events.data.connect(self.on_prompt_update_event)
            point_layer_negative.events.data.connect(self.on_prompt_update_event)
            
            # set active layer to positive point layer
            self._viewer.layers.selection.active = point_layer_positive
        elif prompt_type == "BBox":
            bbox_layer = BoxPromptLayer(name='BBox Prompt Layer', ndim=3)
            self._viewer.add_layer(bbox_layer)
            self.prompt_layers['bbox'] = bbox_layer

            bbox_layer.events.data.connect(self.on_prompt_update_event)
            # set active layer to bbox layer
            self._viewer.layers.selection.active = bbox_layer
   
    def reset(self):
        # Reset the predictor state
        self.predictor.reset_state()
       
        self.model_selection.setDisabled(False)
        self.layerselect_a.setDisabled(False)
        self.threshold_slider.setDisabled(False)
        self.prompt_type_select.setDisabled(False)
        
    def update_hyperparameters(self):
        if get_value(self.run_ckbx):
            self.run_predict_in_thread()
    def on_prompt_update_event(self, event):
        # Ignore in progress events like adding, removing, changing
        if hasattr(event, 'action') and event.action in ['adding', 'removing', 'changing']:
            return

        if get_value(self.run_ckbx):
            self.run_predict_in_thread()

    def run_predict_in_thread(self):
        @thread_worker
        def _worker():
            self.predict()
        _worker().start()

    def predict(self):
        try:
            
            prompt_type = get_value(self.prompt_type_select)[0]
            img_layer = get_value(self.layerselect_a)[1]

            self.predictor.mask_threshold = get_value(self.threshold_slider)

            current_frame_idx = self._viewer.dims.current_step[self._viewer.dims.order[0]]
            
            frame = np.transpose(self._viewer.layers[img_layer].data, self._viewer.dims.order)[current_frame_idx]
            # normalize the frame to 0-255 range if it's not already
            if frame.dtype != np.uint8:
                frame = cv2.normalize(frame, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2RGB)

            #self.predictor.reset_state()
            #self.predictor.set_image(frame)
            # reset the predictor state
            try:
                self.predictor.reset_state()
            except Exception as e:
                pass
            self.predictor.load_first_frame(frame)


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
                frame_idx, obj_ids, out_mask_logits = self.predictor.add_new_points(frame_idx=0, obj_id=0, points=point_prompts, labels=point_labels)  # Use the multi-mask option if checked
                out_mask_masks = out_mask_logits > self.predictor.mask_threshold
                out_mask_masks = out_mask_masks.squeeze().cpu().numpy()  # Convert to numpy array

            elif prompt_type == "BBox":
                bbox_layer = self.prompt_layers['bbox']
                if len(bbox_layer.data) == 0:
                    return
                bbox_data = bbox_layer.data[-1][:, self._viewer.dims.order][:,1:]

                bbox_prompt = np.array([
                    np.min(bbox_data[:, 1]), np.min(bbox_data[:, 0]),
                    np.max(bbox_data[:, 1]), np.max(bbox_data[:, 0])
                ])
                frame_idx, obj_ids, out_mask_logits = self.predictor.add_new_prompt(frame_idx=0, obj_id=0, bbox=bbox_prompt) # XYXY format
                out_mask_masks = out_mask_logits > self.predictor.mask_threshold
                out_mask_masks = out_mask_masks.squeeze().cpu().numpy()  # Convert to numpy array

            #if get_value(self.multi_mask_ckbx) and get_value(self.scoring_ckbx):
            #    # If multi-mask and scoring are enabled, we will have multiple masks
            #    out_mask = out_mask_masks[np.argmax(out_mask_scores)]  # Select the mask with the highest score
            #else:
            
            out_mask = out_mask_masks#[-1]
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
            print(f"Traceback: {traceback.format_exc()}") 

    def run_propagate_in_thread(self):
        @thread_worker
        def _worker():
            self.propagate()
        #if not self.propagating_lock.locked():
        _worker().start()

    def propagate(self):

        self.propagating_lock.acquire()
        # Once the propagation starts, disable the controls to prevent further changes to the initialization parameters
        # until the propagation is done.
        self.model_selection.setDisabled(True)
        self.layerselect_a.setDisabled(True)
        self.threshold_slider.setDisabled(True)
        self.prompt_type_select.setDisabled(True)

        
        img_layer = get_value(self.layerselect_a)[1]

        self.predictor.mask_threshold = get_value(self.threshold_slider)

        next_frame_idx = self._viewer.dims.current_step[self._viewer.dims.order[0]]+1
        if next_frame_idx >= self._viewer.layers[img_layer].data.shape[self._viewer.dims.order[0]]:
            show_warning("No more frames to propagate to.")
            return
        
        next_frame = np.transpose(self._viewer.layers[img_layer].data, self._viewer.dims.order)[next_frame_idx]
        next_frame = cv2.cvtColor(next_frame, cv2.COLOR_GRAY2RGB)

        out_obj_ids, out_mask_logits = self.predictor.track(next_frame)
        out_mask_masks = out_mask_logits>self.predictor.mask_threshold
        out_mask_masks = out_mask_masks.squeeze().cpu().numpy()
        
        out_mask = out_mask_masks#[-1]

        target_size = next_frame.shape[:2]
        
        out_mask = cv2.resize(out_mask.astype(np.uint8), target_size[::-1], interpolation=cv2.INTER_NEAREST)

        np.transpose(self.preview_layer.data, self._viewer.dims.order)[next_frame_idx] = out_mask
        self.preview_layer.refresh()   
        self._viewer.dims.set_current_step(self._viewer.dims.order[0], next_frame_idx)
        self.propagating_lock.release()


    def closeEvent(self, event):
        # Clean up any resources or connections
        for layer in self.prompt_layers.values():
            self._viewer.layers.remove(layer)
        self.prompt_layers.clear()
        if self.preview_layer:
            self._viewer.layers.remove(self.preview_layer)


    def export_preview(self):
       pass
