from napari.utils.notifications import show_info, show_warning, show_error, show_console_notification
from napari import Viewer
from .base_widget import InteractiveSegmentationWidget2DBase

class InteractiveSegmentationWidget2DNoRegistration(InteractiveSegmentationWidget2DBase):
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
        show_info("NoRegistration model used. Empty preview mask")
        pass
    