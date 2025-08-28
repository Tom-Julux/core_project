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

from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls

from napari_promptable.controls.bbox_controls import CustomQtBBoxControls
from napari_promptable.controls.lasso_controls import CustomQtLassoControls
from napari_promptable.controls.point_controls import CustomQtPointsControls
from napari_promptable.controls.scribble_controls import CustomQtScribbleControls
from napari._qt.layer_controls.qt_shapes_controls import QtShapesControls

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
layer_to_controls[ContourPromptLayer] = QtShapesControls

from .multiple_viewer_widget import MultipleViewerWidget

class InteractiveSegmentationWidget3DBase(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        # layers managed by this widget
        self.prompt_layers = {}
        self.preview_layer = None  # Layer to show the preview of the segmentation
        self.preview_label_data = None  # Data Label for the preview layer
        
        self.propagating_lock = threading.Lock()
        self.rerun_after_lock = False
        self.prompt_frame_set_view_1 = False
        self.prompt_frame_set_view_2 = False
        self.prompt_frame_set_view_3 = False

        self.prompt_frame_index_view_1 = 0
        self.prompt_frame_index_view_2 = 0
        self.prompt_frame_index_view_3 = 0

        self.build_gui()
        self.load_model()

        self.update_prompt_type()
        self.on_image_layer_scale_or_rotate()
        
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

    # GUI
    def build_gui(self):
       
        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)

        self.setup_model_selection_gui(_scroll_layout)
        
        _container, _layout = setup_vgroupbox(_scroll_layout, "Image Selection:")
        # layer select for image layer
        self.layerselect_a = setup_layerselect(
            _layout, self._viewer, Image, function=lambda: self.on_image_layer_change()
        )

        _container, _layout = setup_vgroupbox(_scroll_layout, "Prompt Selection:")

        self.prompt_type_select = setup_combobox(#, "Points", "BBox"],\
            _layout, ["Mask"], "QComboBox", function=lambda: self.update_prompt_type()
        )

        _container, _layout = setup_vgroupbox(_scroll_layout, "View Control:")
        self.view_select = setup_hswitch(_layout, ["View A", "View B", "View C"], default=0, function=lambda: self.set_view())
        
        # check box automatically set the prompt on view change
        #self.auto_set_prompt_ckbx = setup_checkbox(
        #    _layout,
        #    "Auto Set Prompt",
        #    True,
        #    tooltips="Automatically set the prompt layer when the view changes.",
        #    function=lambda: None
        #)

        # Progress indicatoin
        setup_label(_layout, "Progress:")
        self.progress_indicator_1 = setup_checkbox(_layout,"Contour in view 1", False)
        self.progress_indicator_1.setDisabled(True)
        self.progress_indicator_2 = setup_checkbox(_layout,"Contour in view 2", False)
        self.progress_indicator_2.setDisabled(True)
        self.progress_indicator_3 = setup_checkbox(_layout,"Contour in view 3", False)
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

        _container, _layout = setup_vgroupbox(_scroll_layout, "Hyperparameters:")

        self.setup_hyperparameter_gui(_layout)
        
        _group_box, _layout = setup_vgroupbox(_scroll_layout, text="Propagation:")

        self.run_button = setup_iconbutton(
            _layout,
            "Predict",
            "right_arrow",
            self._viewer.theme,
            function=lambda: self.run_predict_in_thread(),
            tooltips="Run the predict step",
        )
        self.run_button.setEnabled(False)

        self.run_ckbx = setup_checkbox(
            _layout,
            "Auto Run Prediction",
            False,
            tooltips="Run automatically after each interaction once all three view prompts are set.",
        )

        _container, _layout = setup_vgroupbox(_scroll_layout, "Export to layer:")
        _ = setup_label(_layout, "Export the contents of the preview layer to a sperate layer.")

        #self.export_to_new_layer_ckbx = setup_checkbox(
        #    _layout, "Accumulate",True, tooltips="If unchecked, the preview layer will be exported to the currently selected layer.")

        _ = setup_iconbutton(
            _layout, "Export to layer", "pop_out", self._viewer.theme, self.export_preview
        )
        _container, _layout = setup_vcollapsiblegroupbox(_scroll_layout, "Export to file:", collapsed=True)
        _ = setup_label(_layout, "Export the contents of the preview layer to a new file.")

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
        
        _ = setup_label(_layout, "Reset the widget and clear all prompt and preview layers.")

        _ = setup_iconbutton(
            _layout, "Reset", "erase", self._viewer.theme, self.on_image_layer_change
        )

    def set_view(self):
        # Set the current view based on the selected option in the view_select widget
        selected_view = get_value(self.view_select)[1]

        print(f"Selected View: {selected_view}")

        current_view = self._viewer.dims.order[0]
        print(f"Current Order: {current_view}")

        # Update the prompt type based on the current view
        #if get_value(self.auto_set_prompt_ckbx):
        #    self.set_current_view_prompt(view=selected_view)
        
        if selected_view == 0:
            self._viewer.dims.order = (0, 1, 2)  # Set the order of dimensions to A
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
        #prompt_frames = [self.prompt_frame_index_view_1, self.prompt_frame_index_view_2, self.prompt_frame_index_view_3]
        #mask_0 = np.take(mask_prompt_layer.data,prompt_frames[view], axis=view)  # Take the mask along the current view axis

        if view == 0:
            self.prompt_frame_set_view_1 = True
            self.prompt_frame_index_view_1 = prompt_frames[0]
            set_value(self.progress_indicator_1, self.prompt_frame_set_view_1)
            self.progress_indicator_1.setText(f"Contour in view 1 (slice {self.prompt_frame_index_view_1})")
        elif view == 1:
            self.prompt_frame_set_view_2 = True
            self.prompt_frame_index_view_2 = prompt_frames[1]
            set_value(self.progress_indicator_2, self.prompt_frame_set_view_2)
            self.progress_indicator_2.setText(f"Contour in view 2 (slice {self.prompt_frame_index_view_2})")
        elif view == 2:
            self.prompt_frame_set_view_3 = True
            self.prompt_frame_index_view_3 = prompt_frames[2]
            set_value(self.progress_indicator_3, self.prompt_frame_set_view_3)
            self.progress_indicator_3.setText(f"Contour in view 3 (slice {self.prompt_frame_index_view_3})")

        if self.prompt_frame_set_view_1 and self.prompt_frame_set_view_2 and self.prompt_frame_set_view_3:
            self.run_button.setEnabled(True)
            if get_value(self.run_ckbx):
                self.run_predict_in_thread()

    def clear_prompt_layers(self):
        # Remove all existing prompt layers
        for layer in self.prompt_layers.values():
            if layer in self._viewer.layers:
                self._viewer.layers.remove(layer)
        self.prompt_layers.clear()
    
    def on_image_layer_change(self, event=None):
        # This method is called when the image layer is changed

        self.closeEvent()  # Clean up previous layers
        img_layer = get_value(self.layerselect_a)[1]

        if img_layer is None or img_layer not in self._viewer.layers:
            show_warning("Please select a valid image layer.")
            self.run_button.setEnabled(False)
            return


        self.update_prompt_type()
        # connect the image layer's scale and rotation changes to the preview layer
        img_layer = self._viewer.layers[img_layer]
        img_layer.events.scale.connect(self.on_image_layer_scale_or_rotate)
        img_layer.events.rotate.connect(self.on_image_layer_scale_or_rotate)
        img_layer.events.translate.connect(self.on_image_layer_scale_or_rotate)

        self.on_image_layer_scale_or_rotate()

    def on_image_layer_scale_or_rotate(self, event=None):
        # This method is called when the image layer is scaled or rotated
        img_layer = get_value(self.layerselect_a)[1]

        if self.preview_layer is not None:
            self.preview_layer.scale = self._viewer.layers[img_layer].scale
            self.preview_layer.rotate = self._viewer.layers[img_layer].rotate
            self.preview_layer.translate = self._viewer.layers[img_layer].translate
        if self.prompt_layers:
            for layer in self.prompt_layers.values():
                layer.scale = self._viewer.layers[img_layer].scale
                layer.rotate = self._viewer.layers[img_layer].rotate
                layer.translate = self._viewer.layers[img_layer].translate

    def update_prompt_type(self):
        prompt_type = get_value(self.prompt_type_select)[0]
        
        self.clear_prompt_layers()

        img_layer = get_value(self.layerselect_a)[1]

        img_layer_shape = self._viewer.layers[img_layer].data.shape
        self.preview_label_data = np.zeros(img_layer_shape, dtype=np.uint8)
        self.preview_layer = Labels(name='Preview Layer', data=self.preview_label_data, opacity=0.5)
        
        self.preview_layer.scale = self._viewer.layers[img_layer].scale
        self.preview_layer.rotate = self._viewer.layers[img_layer].rotate
        self.preview_layer.translate = self._viewer.layers[img_layer].translate

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
        elif prompt_type == "Mask":
            data = np.zeros(img_layer_shape, dtype=np.uint8)
            mask_layer = ScribblePromptLayer(name='Mask Prompt Layer', data=data)
            self._viewer.add_layer(mask_layer)
            self.prompt_layers['mask'] = mask_layer

            mask_layer.events.data.connect(self.on_prompt_update_event)
            # set active layer to bbox layer
            self._viewer.layers.selection.active = mask_layer
        
    def update_hyperparameters(self):
        if get_value(self.run_ckbx):
            self.run_predict_in_thread()

    def on_prompt_update_event(self, event):
        # Ignore in progress events like adding, removing, changing
        if hasattr(event, 'action') and event.action in ['adding', 'removing', 'changing']:
            return
        print(event)

        if get_value(self.run_ckbx):
            self.run_predict_in_thread()

    def run_predict_in_thread(self):
        @thread_worker
        def _worker():
            if self.propagating_lock.locked():
                self.rerun_after_lock = True
                return
            with self.propagating_lock:
                self.rerun_after_lock = True
                while self.rerun_after_lock:
                    self.rerun_after_lock = False
                    self.predict()
        _worker().start()

    def closeEvent(self, event=None):
        # Clean up any resources or connections
        for layer in self.prompt_layers.values():
            self._viewer.layers.remove(layer)
        self.prompt_layers.clear()
        
        if self.preview_layer:
            self._viewer.layers.remove(self.preview_layer)

        # disconnect the image layer's scale and rotation changes
        img_layer = get_value(self.layerselect_a)[1]
        if img_layer is not None and img_layer in self._viewer.layers:
            img_layer = self._viewer.layers[img_layer]
            img_layer.events.scale.disconnect(self.on_image_layer_scale_or_rotate)
            img_layer.events.rotate.disconnect(self.on_image_layer_scale_or_rotate)
            img_layer.events.translate.disconnect(self.on_image_layer_scale_or_rotate)

        self.prompt_frame_set_view_1 = False
        self.prompt_frame_set_view_2 = False
        self.prompt_frame_set_view_3 = False
        self.run_button.setEnabled(False)

        self.reset_model()
        
    def export_preview(self):
        # Export the contents of the preview layer to a separate layer
        if self.preview_layer is None:
            show_warning("No preview layer to export.")
            return
        
        # Create a new Labels layer with the data from the preview layer
        new_layer = Labels(name='Exported Layer', data=self.preview_layer.data.copy())
        
        # Add the new layer to the viewer
        self._viewer.add_layer(new_layer)
        
        show_info("Preview layer exported successfully.")
    
    def export_to_file(self):
        if self.preview_layer is None:
            show_warning("No preview layer to export.")
            return
        
        file_path = get_value(self.export_file_select)
        if not file_path:
            show_warning("Please select a file path to export the preview layer.")
            return
        
        try:
            import SimpleITK as sitk
            # Convert the numpy array to a SimpleITK image
            sitk_image = sitk.GetImageFromArray(self.preview_layer.data.astype(np.uint8))
            # Save the image to the specified file path
            sitk.WriteImage(sitk_image, file_path, useCompression=True)
            show_info(f"Preview layer exported successfully to {file_path}.")
        except Exception as e:
            show_error(f"Failed to export preview layer: {e}")


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
                #prompt_frames = [self.prompt_frame_index_view_1, self.prompt_frame_index_view_2, self.prompt_frame_index_view_3]
                mask_1 = mask_prompt_layer.data[prompt_frames[0]]
                mask_2 = mask_prompt_layer.data[:,prompt_frames[1]]
                mask_3 = mask_prompt_layer.data[:,:,prompt_frames[2]]
                
                if self.predictor is None:
                    # expand the mask to match the shape of the image data
                    mask_1 = np.repeat(mask_1[None,...], img_data.shape[0], axis=0)
                    mask_2 = np.repeat(mask_2[:,None,:], img_data.shape[1], axis=1)
                    mask_3 = np.repeat(mask_3[:,:,None], img_data.shape[2], axis=2)
            
                    union_mask = np.logical_and(mask_1, mask_2)
                    union_mask = np.logical_and(union_mask, mask_3)
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
                    m_threshold = (np.array([m1.sum(), m2.sum(), m3.sum()])>20).sum()
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