from typing import Any, List

import numpy as np
from napari.layers import Shapes
from napari.layers.base._base_constants import ActionType
from napari.layers.shapes._shape_list import ShapeList

from napari_nninteractive.layers.abstract_layer import BaseLayerClass


class CustomShapeList(ShapeList):
    """Custom list of shapes that disallows rotation, ensuring bounding boxes
    are always aligned with the grid.

    Attributes:
        ndisplay (int): The display dimensionality of the shapes.
    """

    def rotate(self, index, angle, center=None):
        """Override rotation to prevent bounding boxes from rotating."""


class BBoxLayer(BaseLayerClass, Shapes):
    """
    A bounding box layer class that extends `BaseLayerClass` and `Shapes` with specific color
    management and interaction handling. This class manages the addition, removal, and color
    updating of bounding boxes and restricts rotation.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._data_view = CustomShapeList(ndisplay=self._slice_input.ndisplay)
        self._data_view.slice_key = np.array(self._data_slice.point)[
            self._slice_input.not_displayed
        ]

    def replace_color(self, _color: List[float]) -> None:
        """
        Replaces the color of the last added shape in the layer.

        Args:
            _color (List[float]): The RGBA color to apply to the edge and face of the last shape.
        """
        self._data_view.update_edge_color(len(self.data) - 1, _color)
        self._data_view.update_face_color(len(self.data) - 1, _color)

    def remove_last(self) -> None:
        """
        Removes the last shape in the layer along with associated colors and metadata.
        """
        index = [len(self.data) - 1]
        to_remove = sorted(index, reverse=True)
        self.events.data(
            value=self.data,
            action=ActionType.REMOVING,
            data_indices=tuple(
                index,
            ),
            vertex_indices=((),),
        )
        for ind in to_remove:
            self._data_view.remove(ind)

        self._feature_table.remove(index)
        self.text.remove(index)
        self._data_view._edge_color = np.delete(self._data_view._edge_color, index, axis=0)
        self._data_view._face_color = np.delete(self._data_view._face_color, index, axis=0)

    def run(self) -> None:
        """
        Finalizes the current operation, resets selected data, and completes drawing.
        """
        super().run()
        self.selected_data.clear()
        self._finish_drawing()

    def _add(self, data: Any, *args, **kwargs) -> None:
        """
        Adds a shape with specified colors, utilizing the current prompt's color settings.

        Args:
            data (Any): The shape data to add.
        """
        Shapes.add(
            self,
            data,
            *args,
            edge_color=self.colors[self.prompt_index],
            face_color=self.colors[self.prompt_index],
            **kwargs,
        )

    def remove_selected(self) -> None:
        """Removes selected points if any."""
        index = list(self.selected_data)
        if not self._is_free and len(index) == 1 and index[0] == (len(self.data) - 1):
            super().remove_selected()
            self._is_free = True

    def _rotate_box(self, angle, center=(0, 0)) -> None:
        """Disable rotation by overriding the rotation function."""

    @Shapes.selected_data.setter
    def selected_data(self, selected_data: set) -> None:
        """
        Sets the selected data, restricting selection to the last added shape when the layer is occupied.

        Args:
            selected_data (set): The set of selected shapes' indices.
        """
        index = list(selected_data)
        if len(index) == 1 and index[0] == (len(self.data) - 1) and not self._is_free:
            Shapes.selected_data.fset(self, selected_data)
        else:
            Shapes.selected_data.fset(self, set())

    def get_last(self) -> Any:
        """
        Retrieves the last shape data added to the layer.

        Returns:
            Any: The data of the last added shape.
        """
        return self.data[-1]
