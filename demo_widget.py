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
        
        main_layout = QVBoxLayout(self)

        _scroll_layout = main_layout#setup_vscrollarea(main_layout)

        setup_label(_scroll_layout, "Select a demo to load:")

        # layer select for image layer
        self.demo_select = setup_combobox(#, "Points", "BBox"],\
            _scroll_layout, ["Mask"], "QComboBox", function=lambda: None
        )

        self.run_button = setup_iconbutton(
            None,
            "Load",
            "right_arrow",
            self._viewer.theme,
            function=self.load_demo,
        )

        self.reset_button = setup_iconbutton(
            None,
            "Reset",
            "erase",
            self._viewer.theme,
            function=self.reset_viewer,
        )
        hstack(_scroll_layout, [self.run_button, self.reset_button])
        self.reset_viewer()

    def load_demo(self, demo_id=None):
        if demo_id is None:
            demo_id = get_value(self.demo_select)[0]
            
        show_info(f"Loading demo: {demo_id}")

        self.reset_viewer()

        if demo_id == "Mask":
            img = sitk.ReadImage(
                #'/Users/tomjulius/Developer/napari-plugins/example_data/A_005_frames_8bit.mha'
                '/Users/tomjulius/Developer/core_project/example_data/3d mrlinac/aumc_lung_patient026__GTV.mha'
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

            from napari_interactive import InteractiveSegmentationWidgetSAM2_3D
            widget = InteractiveSegmentationWidgetSAM2_3D(self._viewer)
            self._viewer.window.add_dock_widget(
                widget, name="SAM2 3D", area="right"
            )
            show_info("Demo loaded. Please select the image layer and set the prompt layer in the widget.")
        else:
            show_warning(f"Demo '{demo_id}' not found.")


    def reset_viewer(self):
        # remove all layers except the image layer
        for layer in self._viewer.layers:
            try:
                self._viewer.layers.remove(layer)
            except:
                pass
        # unload all dock widgets except this one
        for name, widget in list(self._viewer.window._dock_widgets.items()):
            try:
                if widget is not self and widget.widget() is not self:
                    self._viewer.window.remove_dock_widget(widget)
            except:
                pass
