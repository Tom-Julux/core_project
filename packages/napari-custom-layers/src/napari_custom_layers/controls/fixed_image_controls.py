from napari._qt.layer_controls.qt_image_controls import QtImageControls
from napari.utils.action_manager import action_manager
import napari
from packaging.version import Version

class CustomQtFixedImageControls(QtImageControls):
    """Custom Qt controls for labels layer used for previewing labels.
    
    Hides unnecessary controls and buttons since the layer is non-editable.

    Args:
        layer (Labels): The labels layer associated with this control panel.
    """

    def __init__(self, layer):
        super().__init__(layer)

        # We don't need this Fields -> Hide them
        self._projection_mode_control.projection_combobox.setHidden(True)
        self._projection_mode_control.projection_combobox_label.setHidden(True)

        # We don't need all these button -> hide and disable tem + remove key binding
        buttons_to_hide = [
            {"button": self.transform_button, "shortcut": "napari:activate_image_transform_mode"},
        ]

        for button in buttons_to_hide:
            button["button"].setDisabled(True)
            button["button"].hide()
            action_manager.unbind_shortcut(button["shortcut"])

        # Reorder the remaining buttons to not have a sparse layout
        #self.button_grid.addWidget(self.delete_button, 0, 1)
        #self.button_grid.addWidget(self.polygon_lasso_button, 0, 2)

