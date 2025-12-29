import os
import numpy as np
from napari.utils.notifications import show_info, show_warning
from napari import Viewer
from napari_toolkit.containers.boxlayout import hstack
from napari_toolkit.utils.widget_getter import get_value
from napari_toolkit.widgets import setup_combobox, setup_iconbutton, setup_label
from qtpy.QtWidgets import QVBoxLayout, QWidget
import SimpleITK as sitk

class DemoWidget(QWidget):    
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer  # type: Viewer
        
        # list of all available demos
        # add new demos here and in the load_demo function below
        self.DEMOS = [
            "Select a demo...",
            "shifted labels",
            "Import",
            "Size Estimator",
            "QuickView",
            "QuickSize",
            "QuickSize3D",
            "QuickSize3DNNI",
            "napari-shape-based-interpolation",
            "---",
            "2D NoPredictor",
            "SAM2 2D",
            "---",
            "SAM2 2D+t CineMRI",
            "SAM2 3D (3D case with one 2d masks)",
            "Fetal Tracking SAM2 3D+t",
            "---"
            "3D NoPredictor",
            "SAM2 3D (3D case with 3 2d masks)",
            #"nnInteractive 3D NNI"
        ]

        self.active_widget = None

        main_layout = QVBoxLayout(self)
        # the demo selection does not need to scroll
        _scroll_layout = main_layout

        setup_label(_scroll_layout, "Select a demo to load:")        

        self.demo_select = setup_combobox(
            _scroll_layout, self.DEMOS, "QComboBox", function=lambda: None
        )

        # setup run and reset buttons
        self.run_button = setup_iconbutton(
            None,
            "Load",
            "right_arrow",
            self._viewer.theme,
            function=lambda: self.load_demo()
        )

        self.reset_button = setup_iconbutton(
            None,
            "Reset",
            "erase",
            self._viewer.theme,
            function=lambda: self.reset_viewer()
        )
        hstack(_scroll_layout, [self.run_button, self.reset_button])
        # initial reset
        self.reset_viewer()

    def load_demo(self, demo_id=None):
        # load the demo as specified in the demo_id, or from the dropdown if None
        if demo_id is None:
            demo_id = get_value(self.demo_select)[0]

        show_info(f"Loading demo: {demo_id}")

        self.reset_viewer()
        base_path = os.path.dirname(os.path.abspath(__file__))
        if demo_id == "Mask 3D NNI":
            if os.path.exists("/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"):
                img = sitk.ReadImage(
                    "/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"
                )
            else:
                 img = sitk.ReadImage(
                    f'{base_path}/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha'
                )

            img = sitk.GetArrayFromImage(img)

            image_layer = self._viewer.add_image(
                img,
                name='Example Image',
                colormap='gray'
            )
            image_layer.scale = [-2, 1, 1]
            image_layer.translate = np.array(image_layer.data.shape) * (image_layer.scale * (image_layer.scale !=1))
            self._viewer.dims.current_step = (img.shape[0]//2, img.shape[1]//2, img.shape[2]//2)

            from napari_promptable_nni._widget_3d_nni import PromptableSegmentationWidget3DNNI
            widget = PromptableSegmentationWidget3DNNI(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Promptable Segmentation", area="right"
            )
            self.active_widget = widget
        elif demo_id == "3D NoPredictor":
            if os.path.exists("/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"):
                img = sitk.ReadImage(
                    "/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"
                )
                mask = sitk.ReadImage(
                    "/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"
                )
            else:
                 img = sitk.ReadImage(
                    f'{base_path}/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha'
                )

            img = sitk.GetArrayFromImage(img)

            image_layer = self._viewer.add_image(
                img,
                name='Example Image',
                colormap='gray'
            )
            image_layer.scale = [-2, 1, 1]
            image_layer.translate = np.array(image_layer.data.shape) * (image_layer.scale * (image_layer.scale !=1))
            self._viewer.dims.current_step = (img.shape[0]//2, img.shape[1]//2, img.shape[2]//2)

            from napari_promptable._widget_3d_noregistration import PromptableSegmentationWidget3DNoRegistration
            widget = PromptableSegmentationWidget3DNoRegistration(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Promptable Segmentation", area="right"
            )
            self.active_widget = widget
        elif demo_id == "2D NoPredictor":
            if os.path.exists("/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"):
                img = sitk.ReadImage(
                    "/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"
                )
            else:
                 img = sitk.ReadImage(
                    f'{base_path}/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha'
                )

            img = sitk.GetArrayFromImage(img)
            img = img[img.shape[0]//2]
            image_layer = self._viewer.add_image(
                img,
                name='Example Image',
                colormap='gray'
            )

            from napari_promptable._widget_2d_noregistration import PromptableSegmentationWidget2DNoRegistration
            widget = PromptableSegmentationWidget2DNoRegistration(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Promptable Segmentation", area="right"
            )
            self.active_widget = widget
        elif demo_id == "SAM2 2D":
            if os.path.exists("/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"):
                img = sitk.ReadImage(
                    "/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"
                )
            else:
                 img = sitk.ReadImage(
                    f'{base_path}/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha'
                )

            img = sitk.GetArrayFromImage(img)
            img = img = img[img.shape[0]//2-16:img.shape[0]//2+32]
            image_layer = self._viewer.add_image(
                img,
                name='Example Image',
                colormap='gray'
            )

            from napari_promptable_sam2._widget_2d_sam import PromptableSegmentationWidget2DSAM
            widget = PromptableSegmentationWidget2DSAM(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Promptable Segmentation", area="right"
            )
            self.active_widget = widget

        elif demo_id == "SAM2 3D (3D case with one 2d masks)":
            if os.path.exists("/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"):
                img = sitk.ReadImage(
                    "/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"
                )
            else:
                 img = sitk.ReadImage(
                    f'{base_path}/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha'
                )

            img = sitk.GetArrayFromImage(img)
            img = img = img[img.shape[0]//2-16:img.shape[0]//2+32]
            image_layer = self._viewer.add_image(
                img,
                name='Example Image',
                colormap='gray'
            )

            from napari_promptable_sam2._widget_2dt_sam import PromptableSegmentationWidget2DTSAM
            widget = PromptableSegmentationWidget2DTSAM(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Promptable Segmentation", area="right"
            )
            self.active_widget = widget
        elif demo_id == "shifted labels":
            if os.path.exists("/app/example_data/2d+t_trackrad/A_003_frames_8bit.mha"):
                img = sitk.ReadImage(
                    "/app/example_data/2d+t_trackrad/A_003_frames_8bit.mha"
                )
            else:
                 img = sitk.ReadImage(
                    f'{base_path}/example_data/2d+t_trackrad/A_003_frames_8bit.mha'
                )

            img = sitk.GetArrayFromImage(img)

            img = np.rot90(img, k=1, axes=(1,2))

            image_layer = self._viewer.add_image(
                img,
                name='Example Image',
                colormap='gray'
            )

            labels_layer = self._viewer.add_labels(
                np.zeros_like(img, dtype=np.uint8),
                name='Example Label'
            )

            from napari_shifted_labels import ShiftedLabelsWidget
            widget = ShiftedLabelsWidget(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="ShiftedLabel", area="right"
            )
            self.active_widget = widget
        elif demo_id == "Size Estimator":
            if os.path.exists("/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"):
                img = sitk.ReadImage(
                    "/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"
                )
            else:
                 img = sitk.ReadImage(
                    f'{base_path}/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha'
                )

            img = sitk.GetArrayFromImage(img)
            img = img = img[img.shape[0]//2-16:img.shape[0]//2+32]
            image_layer = self._viewer.add_image(
                img,
                name='Example Image',
                colormap='gray'
            )

            from napari_size_estimator import SizeEstimatorWidget
            widget = SizeEstimatorWidget(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Size Estimator", area="right"
            )
            self.active_widget = widget
        elif demo_id == "QuickView":
           

            from napari_quick_view import QuickViewWidget
            widget = QuickViewWidget(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="QuickView", area="right"
            )
            self.active_widget = widget
        elif demo_id == "QuickSize":
            from napari_quick_view import QuickSizeWidget
            widget = QuickSizeWidget(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="QuickSizeWidget", area="right"
            )
            self.active_widget = widget
        elif demo_id == "QuickSize3D":
            from napari_quick_view import QuickSize3DWidget
            widget = QuickSize3DWidget(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="QuickSize3DWidget", area="right"
            )
            self.active_widget = widget
        elif demo_id == "QuickSize3DNNI":
            from napari_quick_view import QuickSize3DNNIWidget
            widget = QuickSize3DNNIWidget(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="QuickSize3DNNI", area="right"
            )
            self.active_widget = widget
        elif demo_id == "napari-shape-based-interpolation":
            if os.path.exists("/app/example_data/2d+t_trackrad/A_003_frames_8bit.mha"):
                img = sitk.ReadImage(
                    "/app/example_data/2d+t_trackrad/A_003_frames_8bit.mha"
                )
            else:
                img = sitk.ReadImage(
                    f'{base_path}/example_data/2d+t_trackrad/A_003_frames_8bit.mha'
                )

            img = sitk.GetArrayFromImage(img)

            img = np.rot90(img, k=1, axes=(1,2))

            image_layer = self._viewer.add_image(
                img,
                name='Example Image',
                colormap='gray'
            )

            labels_layer = self._viewer.add_labels(
                np.zeros_like(img, dtype=np.uint8),
                name='Example Label'
            )

            from napari_shape_based_interpolation import ShapeBasedInterpolationWidget
            widget = ShapeBasedInterpolationWidget(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="napari-shape-based-interpolation", area="right"
            )
            self.active_widget = widget 
                
        elif demo_id == "Import":
            if os.path.exists("/app/example_data/2d+t_trackrad/A_003_frames_8bit.mha"):
                img = sitk.ReadImage(
                    "/app/example_data/2d+t_trackrad/A_003_frames_8bit.mha"
                )
            else:
                 img = sitk.ReadImage(
                    f'{base_path}/example_data/2d+t_trackrad/A_003_frames_8bit.mha'
                )
            
            img = sitk.GetArrayFromImage(img)

            img = np.rot90(img, k=1, axes=(1,2))

            image_layer = self._viewer.add_image(
                img,
                name='Example Image',
                colormap='gray'
            )

            from napari_promptable._widget_2d_noregistration import PromptableSegmentationWidget2DNoRegistration
            widget = PromptableSegmentationWidget2DNoRegistration(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Promptable Segmentation", area="right"
            )
            self.active_widget = widget
            self._viewer.dims.current_step = (34,0,0)

        elif demo_id == "SAM2 2D+t CineMRI":
            if os.path.exists("/app/example_data/2d+t_trackrad/A_003_frames_8bit.mha"):
                img = sitk.ReadImage(
                    "/app/example_data/2d+t_trackrad/A_003_frames_8bit.mha"
                )
            else:
                 img = sitk.ReadImage(
                    f'{base_path}/example_data/2d+t_trackrad/A_003_frames_8bit.mha'
                )
            
            img = sitk.GetArrayFromImage(img)

            img = np.rot90(img, k=1, axes=(1,2))

            image_layer = self._viewer.add_image(
                img,
                name='Example Image',
                colormap='gray'
            )

            from napari_promptable_sam2._widget_2dt_sam import PromptableSegmentationWidget2DTSAM
            widget = PromptableSegmentationWidget2DTSAM(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Promptable Segmentation", area="right"
            )
            self._viewer.dims.current_step = (34,0,0)
            self.active_widget = widget
            
        elif demo_id == "SAM2 3D (3D case with 3 2d masks)":
            if os.path.exists("/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"):
                img = sitk.ReadImage(
                    "/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"
                )
            else:
                 img = sitk.ReadImage(
                    f'{base_path}/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha'
                )

            img = sitk.GetArrayFromImage(img)
            img = img[img.shape[0]//2-16:img.shape[0]//2+32]
            image_layer = self._viewer.add_image(
                img,
                name='Example Image',
                colormap='gray'
            )

            from napari_promptable_sam2._widget_3d_sam import PromptableSegmentationWidget3DSAM
            widget = PromptableSegmentationWidget3DSAM(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Promptable Segmentation", area="right"
            )
            self.active_widget = widget
        
        elif demo_id == "Fetal Tracking SAM2 3D+t":
            if os.path.exists("/app/example_data/3d+t fetal tracking/fm0116_3_e1.nii.gz"):
                img = sitk.ReadImage(
                    "/app/example_data/3d+t fetal tracking/fm0116_3_e1.nii.gz"
                )
            else:
                 img = sitk.ReadImage(
                    f'{base_path}/example_data/3d+t fetal tracking/fm0116_3_e1.nii.gz'
                )

            img = sitk.GetArrayFromImage(img)
            #img = img[img.shape[0]//2:img.shape[0]//2+16]
            image_layer = self._viewer.add_image(
                img,
                name='Example Image',
                colormap='gray'
            )

            from napari_promptable_sam2._widget_2dt_sam import PromptableSegmentationWidget2DTSAM
            widget = PromptableSegmentationWidget2DTSAM(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Promptable Segmentation", area="right"
            )
            self.active_widget = widget
            
        elif demo_id == "Select a demo...":
            pass
        else:
            # No such demo exists
            show_warning(f"Demo '{demo_id}' not found.")


    def reset_viewer(self):
        # this function tries to remove all layers and widgets from the viewer
        # this is needed to free memory and GPU resources when loading new demos
        # it also might fail, in which case napari might need to be restarted to free all resources
        # this might look like a crash, but is actually the intended behavior ;)

        # remove all layers except the image layer
        for layer in self._viewer.layers:
            try:
                self._viewer.layers.remove(layer)
            except:
                pass
        
        # remove active widget
        if self.active_widget is not None:
            try:
                # this should call the closeEvent of the widget
                self._viewer.window.remove_dock_widget(self.active_widget)
                # manually call close event in case the above does not work
                self.active_widget.close()
                # mannually call close event in case the above does not work
                self.active_widget.closeEvent()
            except:
                pass

            # hopefully the widget removes all its resources in the close event
            # if not, garbage collection maybe takes care of it    
            self.active_widget = None
            # if this also does not work, napari might close to free all resources
            # this might look like a crash, but is actually intended ;)
            # otherwise memory leaks might occur, especially with machine learning models
        
        # unload all dock widgets except the demo loader
        for name, widget in list(self._viewer.window.dock_widgets.items()):
            try:
                if widget is not self and widget.widget() is not self:
                    self._viewer.window.remove_dock_widget(widget)
            except:
                pass
