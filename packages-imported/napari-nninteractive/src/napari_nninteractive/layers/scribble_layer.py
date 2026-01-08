import numpy as np
from napari.layers import Labels
from napari.layers.base._base_constants import ActionType

from napari_nninteractive.layers.abstract_layer import BaseLayerClass


class ScribbleLayer(BaseLayerClass, Labels):
    """
    A scribble layer class that extends `BaseLayerClass` and `Labels`, with prompt-based color
    adjustments and custom drawing interactions. This class handles color management, adding
    scribble data, and executing the drawing interactions.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.colormap = {
            None: None,
            1: self.colors[self.prompt_index],
            2: self.colors_set[0],
            3: self.colors_set[1],
        }

        self._is_free = False
        self.mouse_drag_callbacks.append(self.on_draw)

    def replace_color(self, _color) -> None:
        """
        Replaces the color of the scribble label for the current prompt index.

        Args:
            _color (List[float]): The new RGBA color to apply to the current prompt.
        """
        self.colormap = {
            None: None,
            1: self.colors[self.prompt_index],
            2: self.colors_set[0],
            3: self.colors_set[1],
        }
        self.refresh()

    def _add(self, data, *arg, **kwargs) -> None:
        """We dont need this function here"""

    def run(self) -> None:
        """
        Finalizes the current scribble interaction, updating the label index and marking the layer as free.
        """
        self.data[self.data == 1] = self.prompt_index + 2
        self._is_free = True
        self.refresh()

    def remove_last(self) -> None:
        """
        Undoes the last action, reverting the most recent scribble interaction.
        """
        self.undo()

    def _commit_staged_history(self) -> None:
        """
        Commits the current staged history for the layer and marks the action as finished.
        """
        super()._commit_staged_history()
        self._is_free = False
        self.events.finished(action=ActionType.ADDED, value="ADD")

    def on_draw(self, *args, **kwargs) -> None:
        """Handles the drawing interaction for the layer. Removes previous scribbles if the layer is occupied."""
        if not self._is_free:
            self.remove_last()
            self.colormap = {
                None: None,
                1: self.colors[self.prompt_index],
                2: self.colors_set[0],
                3: self.colors_set[1],
            }
            self._is_free = True
            self.refresh()
        self.colormap = {
            None: None,
            1: self.colors[self.prompt_index],
            2: self.colors_set[0],
            3: self.colors_set[1],
        }

    def get_last(self) -> None:
        """
        Retrieves a binary mask of the last scribble interaction.

        Returns:
            np.ndarray: A binary array where 1 indicates the last scribble interaction.
        """
        return (self.data == 1).astype(np.uint8)
