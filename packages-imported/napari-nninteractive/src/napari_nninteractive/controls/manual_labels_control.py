from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari.utils.action_manager import action_manager
import napari
from packaging.version import Version

class CustomQtManualLabelsControls(QtLabelsControls):
    """Custom Qt controls for labels layer used for previewing labels.
    
    Hides unnecessary controls and buttons since the layer is non-editable.

    Args:
        layer (Labels): The labels layer associated with this control panel.
    """

    def __init__(self, layer):
        super().__init__(layer)

        # We don't need this Fields -> Hide them
        self._ndim_spinbox_control.ndim_spinbox.setHidden(True)
        self._ndim_spinbox_control.ndim_spinbox_label.setHidden(True)

        self._contiguous_checkbox_control.contiguous_checkbox.setHidden(True)
        self._contiguous_checkbox_control.contiguous_checkbox_label.setHidden(True)

        self._preserve_labels_checkbox_control.preserve_labels_checkbox.setHidden(True)
        self._preserve_labels_checkbox_control.preserve_labels_checkbox_label.setHidden(True)

        self._display_selected_label_checkbox_control.selected_color_checkbox.setHidden(True)
        self._display_selected_label_checkbox_control.selected_color_checkbox_label.setHidden(True)

        self._colormode_combobox_control.color_mode_combobox.setHidden(True)
        self._colormode_combobox_control.color_mode_combobox_label.setHidden(True)

        self._label_control.selection_spinbox.setHidden(False)
        self._label_control.selection_spinbox.setDisabled(True)

        # We don't need all these button -> hide and disable tem + remove key binding
        buttons_to_hide = [
            {"button": self.pick_button, "shortcut": "napari:activate_labels_picker_mode"},
            {"button": self.fill_button, "shortcut": "napari:activate_labels_fill_mode"},
            {"button": self.transform_button, "shortcut": "napari:activate_labels_transform_mode"},
        ]

        for button in buttons_to_hide:
            button["button"].setDisabled(True)
            button["button"].hide()
            action_manager.unbind_shortcut(button["shortcut"])

        # Reorder the remaining buttons to not have a sparse layout
        #self.button_grid.addWidget(self.delete_button, 0, 1)
        #self.button_grid.addWidget(self.polygon_lasso_button, 0, 2)

