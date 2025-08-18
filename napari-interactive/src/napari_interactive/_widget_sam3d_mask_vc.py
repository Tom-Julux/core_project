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
class InteractiveSegmentationWidgetSAM2(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer

        # layers managed by this widget
        self.prompt_layers = {}
        self.preview_layer = None  # Layer to show the preview of the segmentation
        self.preview_label_data = None  # Data Label for the preview layer
        
        self.propagating_lock = threading.Lock()

        self._multi_viwer_widget = MultipleViewerWidget(self._viewer)
        self._viewer.window.add_dock_widget(self._multi_viwer_widget, name='Linked Viewer', area='bottom')

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

        self.prompt_type_select = setup_combobox(#, "Points", "BBox"],\
            _layout, ["Mask"], "QComboBox", function=lambda: self.update_prompt_type()
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
            "Predict",
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
        elif prompt_type == "Mask":
            data = np.zeros(img_layer_shape, dtype=np.uint8)
            mask_layer = ScribblePromptLayer(name='Mask Prompt Layer', data=data)
            self._viewer.add_layer(mask_layer)
            self.prompt_layers['mask'] = mask_layer

            mask_layer.events.data.connect(self.on_prompt_update_event)
            # set active layer to bbox layer
            self._viewer.layers.selection.active = mask_layer
   
    def reset(self):
        # Reset the predictor state
        pass
        
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
            self.predict()
        _worker().start()

    def predict(self):
        try:
            
            prompt_type = get_value(self.prompt_type_select)[0]
            img_layer = get_value(self.layerselect_a)[1]

            mask_threshold = get_value(self.threshold_slider)

            prompt_frames = self._viewer.dims.current_step

            img_data =self._viewer.layers[img_layer].data.astype(np.float32)
            # normalize the image data to 0-255 range if it's not already
            img_data = (img_data - np.min(img_data)) / (np.max(img_data) - np.min(img_data)) * 255
            img_data = img_data.astype(np.uint8)

            if prompt_type == "Points":
                # Get the positive and negative point layers
                point_layer_positive = self.prompt_layers['point_positive']
                point_layer_negative = self.prompt_layers['point_negative']

                if len(point_layer_positive.data) == 0:
                    return
                
                print(point_layer_positive.data, point_layer_negative.data)
                # Get the data from the positive and negative point layers
                pos_points = point_layer_positive.data#point_layer_positive.data[:, self._viewer.dims.order[::-1]][:, :2]
                neg_points = point_layer_negative.data#.data[:, self._viewer.dims.order[::-1]][:, :2]

                point_prompts = np.concatenate((pos_points, neg_points), axis=0)  # Use only the first two dimensions (x, y)
                point_labels = np.concatenate((np.ones(len(pos_points)), np.zeros(len(neg_points))), axis=0)

                out_mask_logits = np.zeros_like(np.zeros_like(self.preview_layer.data, dtype=np.uint8))
                for point in point_prompts:
                    # Convert point coordinates to the format expected by the predictor
                    out_mask_logits[int(point[0]), int(point[1]), int(point[2])] = 1  # Assuming point is in (z, y, x) format
                # apply binary dilation to the mask logits
                from scipy.ndimage import binary_dilation
                out_mask_logits = binary_dilation(out_mask_logits, structure=np.ones((3,3,3)), iterations=10).astype(np.float32)
                print(out_mask_logits.sum())
                #_, out_obj_ids, out_mask_logits = self.predictor.add_new_points(frame_idx=0, obj_id=0, points=point_prompts, labels=point_labels)
                #frame_idx, obj_ids, out_mask_logits = self.predictor.add_new_points(frame_idx=0, obj_id=0, points=point_prompts, labels=point_labels)  # Use the multi-mask option if checked
                

                out_mask_masks = out_mask_logits > 0.5
                #out_mask_masks = out_mask_masks.squeeze().cpu().numpy()  # Convert to numpy array

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

            elif prompt_type == "Mask":
                mask_prompt_layer = self.prompt_layers['mask']
                prompt_frames = self._viewer.dims.current_step
                mask_1 = mask_prompt_layer.data[prompt_frames[0]]
                mask_2 = mask_prompt_layer.data[:,prompt_frames[1]]
                mask_3 = mask_prompt_layer.data[:,:,prompt_frames[2]]
                
                if False:
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
                    from sam2_utils import propagate_along_path, merge_results, view_1_to_view_2, view_1_to_view_3, mask_view_2_to_view_1, center_of_mass, add_negative_point_prompts, view_2_to_view_1, view_3_to_view_1, mask_view_3_to_view_1, mask_view_1_to_view_2, mask_view_1_to_view_3
                    volume_labels = mask_prompt_layer.data
                    volume_data = img_data
                    sam_mask_threshold = self.predictor.mask_threshold
                    center = np.array(center_of_mass(volume_labels), dtype=int)
                    center_slice = center[0] #np.argmax(np.sum(volume_labels, axis=(1,2)))

                    point = center_of_mass(volume_labels[center_slice])[::-1]
                    point_prompts = np.array([point], dtype=np.float32)
                    #initial_point = np.array([center_slice, point[0],point[1]], dtype=np.float32)
                    initial_point = np.array([center[0],center[2],center[1]], dtype=np.float32)
                    point_labels = np.array([1], dtype=np.int32)
                    #point_prompts = np.concatenate([point_prompts, np.array([[volume_data.shape[1]/2, volume_data.shape[2]/2]], dtype=np.float32)], axis=0)
                    #point_prompts, point_labels = add_negative_point_prompts(point_prompts, point_labels, volume_labels, d=negative_point_distance)
                    #cha forward in space
                    results_1 = merge_results(  propagate_along_path(volume_data[:center_slice+1][::-1], self.predictor,threshold=sam_mask_threshold, keep_logits=True, reset_state=True, initialization="mask", mask_prompt=volume_labels[center_slice]),\
                                                propagate_along_path(volume_data[center_slice:], self.predictor,threshold=sam_mask_threshold, keep_logits=True, reset_state=True, initialization="mask", mask_prompt=volume_labels[center_slice]))
                    # do propagation
                    volume_data_view2 = view_1_to_view_2(volume_data)
                    volume_labels_view2 = view_1_to_view_2(volume_labels)
                    center_slice_2 = volume_data_view2.shape[0] - initial_point[2].astype(np.int32)
                    results_2 = merge_results(  propagate_along_path(volume_data_view2[:center_slice_2+1][::-1], self.predictor,threshold=sam_mask_threshold, keep_logits=True, reset_state=True, initialization="mask", mask_prompt=volume_labels_view2[center_slice_2]),\
                                                propagate_along_path(volume_data_view2[center_slice_2:], self.predictor,threshold=sam_mask_threshold, keep_logits=True, reset_state=True,  initialization="mask", mask_prompt=volume_labels_view2[center_slice_2]))
                    results_2["masks"] = mask_view_2_to_view_1(results_2["masks"])
                    # view 3
                    volume_data_view3 = view_1_to_view_3(volume_data)
                    volume_labels_view3 = view_1_to_view_3(volume_labels).copy()
                    center_slice_3 = volume_data_view3.shape[0] - initial_point[1].astype(np.int32)
                    results_3 = merge_results(  propagate_along_path(volume_data_view3[:center_slice_3+1][::-1], self.predictor,threshold=sam_mask_threshold, keep_logits=True, reset_state=True, initialization="mask", mask_prompt=volume_labels_view3[center_slice_3]),\
                                                propagate_along_path(volume_data_view3[center_slice_3:], self.predictor,threshold=sam_mask_threshold, keep_logits=True, reset_state=True, initialization="mask", mask_prompt=volume_labels_view3[center_slice_3]))
                    results_3["masks"] = mask_view_3_to_view_1(results_3["masks"])
                    m1 = results_1["masks"]
                    m2 = results_2["masks"]
                    m3 = results_3["masks"]
                    m_threshold = (np.array([m1.sum(), m2.sum(), m3.sum()])>20).sum()
                    union = np.sum([m1, m2, m3], axis=0) >= (m_threshold-1 if m_threshold>1 else 1)
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

    def closeEvent(self, event):
        # Clean up any resources or connections
        for layer in self.prompt_layers.values():
            self._viewer.layers.remove(layer)
        self.prompt_layers.clear()
        if self.preview_layer:
            self._viewer.layers.remove(self.preview_layer)


    def export_preview(self):
       pass

class MedSAM2():
    def __init__(self):
        pass

    def predict(self, event):
        pass