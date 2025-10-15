from typing import Callable, Optional, List, Type, Union

from napari.layers import Layer
from napari.viewer import Viewer
from qtpy.QtWidgets import QComboBox, QLayout, QWidget

from napari_toolkit.utils.utils import connect_widget

def _isinstance_of_layer_type(layer, layer_types):
    return any(isinstance(layer, t) for t in layer_types if t is not None)

class QLayerSelect(QComboBox):
    """
    A QComboBox widget that dynamically updates with the names of layers in a Napari viewer.

    Args:
        parent (Optional[QWidget]): The parent widget.
        layer_type (Optional[Union[Type[Layer],List[Type[Layer]]]]): One or more Napari layer types to filter by (e.g., Image, Labels).
    """

    def __init__(self, parent=None, layer_type=Layer, filter_function: Optional[Callable[[Layer, str], bool]] = None):
        super().__init__(parent)
        # Accept a single type or a list/tuple of types
        if isinstance(layer_type, (list, tuple)):
            self.layer_types = tuple(layer_type)
        else:
            self.layer_types = (layer_type,)
        self.filter_function = filter_function
        self.value = self.currentText()
        self.currentIndexChanged.connect(self.update_tooltip)
        self.update_tooltip()
        self.layer_names = {}

    def _update(self, event):
        """
        Updates the combo box with the names of layers in the viewer that match the specified layer type(s).

        Args:
            event: The Napari event triggered by a layer being added or removed.
        """
        # print(event.type)
        # for attr in dir(event):
        #     if not attr.startswith("_"):  # Skip private attributes
        #         print(f"{attr}: {getattr(event, attr)}")

        if event.type == "removed":
            layer = event.value
            if _isinstance_of_layer_type(layer, self.layer_types):

                item_index = self.findText(layer.name)
                if item_index != -1:
                    self.removeItem(item_index)
                    del self.layer_names[layer]
        elif event.type == "inserted":
            layer = event.value
            if _isinstance_of_layer_type(layer, self.layer_types):
                self.addItem(layer.name)
                self.layer_names[layer] = layer.name
                layer.events.name.connect(self._update)
        elif event.type == "name":
            layer = event.source  # The layer that triggered the event
            old_name = self.layer_names.get(layer)
            new_name = layer.name
            self.layer_names[layer] = new_name

            item_index = self.findText(old_name)
            if item_index != -1:  # If the old name exists
                self.setItemText(item_index, new_name)
            else:
                self.addItem(new_name)

    def connect(self, viewer: Viewer):
        """
        Connects the widget to the Napari viewer's layer events to update the combo box
        when layers are added or removed.

        Args:
            viewer (Viewer): The Napari viewer instance to connect to.
        """

        viewer.layers.events.inserted.connect(self._update)
        viewer.layers.events.removed.connect(self._update)
        self.layer_names = {
            layer: layer.name
            for layer in viewer.layers
            if (None in self.layer_types) or _isinstance_of_layer_type(layer, self.layer_types)
        }
        if self.filter_function is not None:
            self.layer_names = {
                layer: name
                for layer, name in self.layer_names.items()
                if self.filter_function(layer, name)
            }
        for layer in self.layer_names:
            layer.events.name.connect(self._update)

        self.addItems(list(self.layer_names.values()))

    def update_tooltip(self):
        # Set the tooltip to the current item’s text
        self.setToolTip(self.currentText())


def setup_layerselect(
    layout: QLayout,
    viewer: Optional[Viewer] = None,
    layer_type: Optional[Layer] = Layer,
    function: Optional[Callable[[str], None]] = None,
    tooltips: Optional[str] = None,
    shortcut: Optional[str] = None,
    stretch: int = 1,
) -> QWidget:
    """
    Adds a LayerSelectionWidget to a layout with optional configurations, including connecting
    a function, setting a tooltip, and adding a keyboard shortcut.

    Args:
        layout (QLayout): The layout to add the LayerSelectionWidget to.
        viewer (Optional[Viewer], optional): The Napari viewer instance to connect the widget to. Defaults to None.
        layer_type (Optional[Union[Type[Layer],List[Type[Layer]]]], optional): A specific Napari layer type to filter by in the selection widget.
        function (Optional[Callable[[str], None]], optional): The function to call when the selection changes.
        tooltips (Optional[str], optional): Tooltip text for the widget. Defaults to None.
        shortcut (Optional[str], optional): A keyboard shortcut to trigger the function. Defaults to None.
        stretch (int, optional): The stretch factor for the spinbox in the layout. Defaults to 1.

    Returns:
        QWidget: The configured LayerSelectionWidget added to the layout.
    """
    _widget = QLayerSelect(layer_type=layer_type)
    if viewer:
        _widget.connect(viewer)
    _widget.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
    return connect_widget(
        layout,
        _widget,
        widget_event=_widget.currentTextChanged,
        function=function,
        shortcut=shortcut,
        tooltips=tooltips,
        stretch=stretch,
    )