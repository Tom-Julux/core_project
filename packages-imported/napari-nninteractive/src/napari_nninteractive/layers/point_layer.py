from typing import Any, List, Sequence

import numpy as np
from napari.layers import Points

from napari_nninteractive.layers.abstract_layer import BaseLayerClass


class SinglePointLayer(BaseLayerClass, Points):
    """
    A layer class for managing single points with prompt-based color updates, extending
    `BaseLayerClass` and `Points`. This class provides methods to add, remove, and manage
    colors for individual points in the layer.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.point_colors = []

    def remove_color(self) -> None:
        """
        Removes the color associated with the last point, updating the face and border colors.
        """
        self.point_colors = self.point_colors[:-1]
        self.face_color = np.array(self.point_colors)
        self.border_color = np.array(self.point_colors)

    def replace_color(self, _color: List[float]) -> None:
        """
        Replaces the color of the last point with a new color.

        Args:
            _color (List[float]): The new RGBA color to apply to the last point.
        """
        self.point_colors[-1] = _color
        self.face_color = np.array(self.point_colors)
        self.border_color = np.array(self.point_colors)

    def remove_last(self) -> None:
        """
        Removes the last point from the layer along with its color.
        """
        self.data = self.data[:-1]
        self.remove_color()

    def _add(self, data: Any, *arg, **kwargs) -> None:
        """
        Adds a new point to the layer with the current prompt color.

        Args:
            data (Any): The point data to add.
        """
        Points.add(self, data, *arg, **kwargs)
        _color = self.colors[self.prompt_index]
        self.point_colors.append(_color)
        self.face_color = np.array(self.point_colors)
        self.border_color = np.array(self.point_colors)

    def remove_selected(self) -> None:
        """Removes selected points if any."""
        index = list(self.selected_data)
        if not self._is_free and len(index) == 1 and index[0] == (len(self.data) - 1):
            super().remove_selected()
            self._is_free = True
            self.remove_color()

    def _move(
        self,
        selection_indices: Sequence[int],
        position: Sequence[int | float],
    ) -> None:
        """Moves the last edited point if it's the only selected point and the layer is occupied."""
        # Only the last edited one should be able to be moved
        if (
            not self._is_free
            and len(selection_indices) == 1
            and list(selection_indices)[0] == (len(self.data) - 1)
        ):
            super()._move(selection_indices, position)

    def get_last(self) -> Any:
        """
        Retrieves the data of the last added point.

        Returns:
            Any: Data of the last added point.
        """
        return self.data[-1]
