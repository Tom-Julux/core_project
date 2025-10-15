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
        
        self.DEMOS = [
            "Select a demo...",
            "shifted labels",
            "---"
            "2D NoPredictor",
            "SAM2 2D",
            "---"
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

        _scroll_layout = main_layout#setup_vscrollarea(main_layout)

        setup_label(_scroll_layout, "Select a demo to load:")

        
        # layer select for image layer
        self.demo_select = setup_combobox(#, "Points", "BBox"],\
            _scroll_layout, self.DEMOS, "QComboBox", function=lambda: None
        )

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
        self.reset_viewer()

    def load_demo(self, demo_id=None):
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

            from napari_interactive._widget_3d_nni import InteractiveSegmentationWidget3DNNI
            widget = InteractiveSegmentationWidget3DNNI(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Interactive Segmentation", area="right"
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

            from napari_interactive._widget_3d_noregistration import InteractiveSegmentationWidget3DNoRegistration
            widget = InteractiveSegmentationWidget3DNoRegistration(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Interactive Segmentation", area="right"
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

            from napari_interactive._widget_2d_noregistration import InteractiveSegmentationWidget2DNoRegistration
            widget = InteractiveSegmentationWidget2DNoRegistration(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Interactive Segmentation", area="right"
            )
            self.active_widget = widget
        elif demo_id == "SAM2 2D":
            if os.path.exists("/app/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"):
                img = sitk.ReadImage(
                    "/app//Users/tomjulius/Developer/core_project/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha"
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

            from napari_interactive._widget_2d_sam import InteractiveSegmentationWidget2DSAM
            widget = InteractiveSegmentationWidget2DSAM(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Interactive Segmentation", area="right"
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

            from napari_interactive._widget_2dt_sam import InteractiveSegmentationWidget2DTSAM
            widget = InteractiveSegmentationWidget2DTSAM(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Interactive Segmentation", area="right"
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

            from napari_interactive._widget_2dt_sam import InteractiveSegmentationWidget2DTSAM
            widget = InteractiveSegmentationWidget2DTSAM(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Interactive Segmentation", area="right"
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

            from napari_interactive._widget_3d_sam import InteractiveSegmentationWidget3DSAM
            widget = InteractiveSegmentationWidget3DSAM(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Interactive Segmentation", area="right"
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

            from napari_interactive._widget_2dt_sam import InteractiveSegmentationWidget2DTSAM
            widget = InteractiveSegmentationWidget2DTSAM(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="Interactive Segmentation", area="right"
            )
            self.active_widget = widget
            
        elif demo_id == "Select a demo...":
            pass
        else:
            show_warning(f"Demo '{demo_id}' not found.")


    def reset_viewer(self):
        # remove all layers except the image layer
        for layer in self._viewer.layers:
            try:
                self._viewer.layers.remove(layer)
            except:
                pass
        if self.active_widget is not None:
            try:
                self._viewer.window.remove_dock_widget(self.active_widget)
                self.active_widget.close()
                self.active_widget.closeEvent()
            except:
                pass
            self.active_widget = None
        # unload all dock widgets except this one
        for name, widget in list(self._viewer.window._dock_widgets.items()):
            try:
                if widget is not self and widget.widget() is not self:
                    self._viewer.window.remove_dock_widget(widget)
            except:
                pass
