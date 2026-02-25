from napari._qt.layer_controls.qt_labels_controls import QtLabelsControls
from napari.utils.action_manager import action_manager
import napari
from packaging.version import Version

import numpy as np
from scipy.ndimage import binary_fill_holes

from qtpy.QtCore import Qt
from qtpy.QtWidgets import (
    QWidget,
    QCheckBox
)
from superqt import QLargeIntSpinBox

from napari._qt.layer_controls.widgets.qt_widget_controls_base import (
    QtWidgetControlsBase,
    QtWrappedLabel,
)
from napari._qt.utils import attr_to_settr
from napari.layers import Labels
from napari.layers.labels._labels_utils import get_dtype
from napari.utils._dtype import get_dtype_limits
from napari.utils.translations import trans
from napari._qt.utils import attr_to_settr, checked_to_bool
from napari.utils.events.event_utils import connect_setattr

class QtShapeBasedInterpolationControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer contour
    thickness attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari.layers.Labels
        An instance of a napari Labels layer.

    Attributes
    ----------
    contour_spinbox : superqt.QLargeSpinBox
        Spinbox to control the layer contour thickness.
    contour_spinbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the layer contour thickness chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Labels) -> None:
        super().__init__(parent, layer)
        # Setup widgets
        self.contour_spinbox = QLargeIntSpinBox()
        dtype_lims = get_dtype_limits(get_dtype(layer))
        self.contour_spinbox.setRange(0, dtype_lims[1])
        self.contour_spinbox.setValue(self._layer.contour)
        self.contour_spinbox.valueChanged.connect(self.change_contour)
        self.contour_spinbox.setKeyboardTracking(False)
        self.contour_spinbox.setAlignment(Qt.AlignmentFlag.AlignCenter)


        self.contour_spinbox_label = QtWrappedLabel("shape based interp.:")

    def change_contour(self, value: int) -> None:
        """Change contour thickness.
        Parameters
        ----------
        value : int
            Thickness of contour.
        """
        self._layer.contour = value
        self.contour_spinbox.clearFocus()
        self.parent().setFocus()

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.contour_spinbox_label, self.contour_spinbox)]


class QtAutofillCheckBoxControl(QtWidgetControlsBase):
    """
    Class that wraps the connection of events/signals between the layer autofill
    attribute and Qt widgets.

    Parameters
    ----------
    parent: qtpy.QtWidgets.QWidget
        An instance of QWidget that will be used as widgets parent
    layer : napari_custom_layers.ManualLabelsLayer
        An instance of a napari_custom_layers ManualLabelsLayer layer.

    Attributes
    ----------
    autofill_checkbox : qtpy.QtWidgets.QCheckBox
        Checkbox to control if label layer is autofilled.
    autofill_checkbox_label : napari._qt.layer_controls.widgets.qt_widget_controls_base.QtWrappedLabel
        Label for the autofill model chooser widget.
    """

    def __init__(self, parent: QWidget, layer: Labels) -> None:
        super().__init__(parent, layer)
        # Setup widgets
        self.autofill_checkbox = QCheckBox()
        self.autofill_checkbox.setChecked(False)
        self.autofill_checkbox.setToolTip("Fill connected components after painting (only in paint mode)")
        self._callbacks.append(
            attr_to_settr(
                self._layer,
                'autofill',
                self.autofill_checkbox,
                'setChecked',
            )
        )
        connect_setattr(
            self.autofill_checkbox.stateChanged,
            layer,
            'autofill',
            convert_fun=checked_to_bool,
        )
        self.autofill_checkbox_label = QtWrappedLabel('autofill:')

    def get_widget_controls(self) -> list[tuple[QtWrappedLabel, QWidget]]:
        return [(self.autofill_checkbox_label, self.autofill_checkbox)]


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

        # TODO
        # shape based interpolation
        # start button
        # apply button

        # quickly switch to erase mode when holding shift and switch back to paint mode when releasing shift
        @layer.bind_key('Shift')
        def quick_switch_tool(viewer):
            print("quick_switch_tool")
            if layer.mode != "paint":
                return
            # on press
            # switch paint/erase mode
            layer.mode = 'erase'
            yield
            # on release
            # switch paint/erase mode back
            # only switch back to paint if we are still in erase mode, otherwise keep the current mode (e.g. if user switched to another mode while holding shift)
            if layer.mode == 'erase': 
                layer.mode = 'paint'

        # shortcuts for increasing/decreasing brush size with shift + plus/minus
        @layer.bind_key('Plus', overwrite=True)
        def increase_brush_size(viewer):
            # on press
            if layer.brush_size < 30:
                layer.brush_size += 1
        
        @layer.bind_key('-', overwrite=True)
        def decrease_brush_size(viewer):
            # on press
            if layer.brush_size > 1:
                layer.brush_size -= 1

        self._autofill_checkbox_control = QtAutofillCheckBoxControl(self, layer)
        self._add_widget_controls(self._autofill_checkbox_control)

        def autofill(event):
            if not layer.autofill:
                return
                
            viewer = napari.current_viewer()
            # get current view
            transposed_layer_data = np.transpose(layer.data.copy(), viewer.dims.order)
            # get current slice in that view
            current_slice = (transposed_layer_data.shape[0] - 1 - viewer.dims.current_step[viewer.dims.order[0]]) \
                if layer.scale[viewer.dims.order[0]] < 0 else \
                viewer.dims.current_step[viewer.dims.order[0]]
            current_slice_data = transposed_layer_data[current_slice]

            # fill holes using 
            filled_slice = binary_fill_holes(current_slice_data == layer.selected_label)   

            # set the filled slice back to the layer data
            transposed_layer_data[current_slice] = filled_slice.astype(layer.data.dtype) * layer.selected_label
            # set the layer data back to the layer         
            with layer.events.blocker():
                np.copyto(layer.data, np.transpose(transposed_layer_data, viewer.dims.order))
                layer.refresh()  # refresh the layer to update the view

        layer.events.paint.connect(autofill)
        
