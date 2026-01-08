from abc import ABC, abstractmethod
from typing import Any, List

from napari.layers.base._base_constants import ActionType
from napari.utils.events import Event


class BaseLayerClass(ABC):
    """
    An abstract base class for layers that manage prompt-based interactions and color updates.

    This class provides mechanisms to set prompt indices, manage layer colors based on prompts,
    add or remove data, and track the occupancy status of the layer (whether it is free or occupied).
    Subclasses must implement the `replace_color`, `remove_last`, and `_add` methods.

    Args:
        prompt_index (int): The index of the current prompt, affecting the color of the layer.
    """

    def __init__(self, prompt_index: int = 0, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._is_free = True
        self.prompt_index = prompt_index
        self.colors = {
            0: [0.0, 0.694, 0.047, 1],  # [0, 177, 12]
            1: [0.827, 0.027, 0.0, 1],  # [211, 7, 0]
        }
        self.colors_set = {
            0: [0.0, 0.427, 0.027, 1],  # [0, 109, 7]
            1: [0.561, 0.02, 0.0, 1],  # [143, 5, 0]
        }
        self.events.add(finished=Event)

    def set_prompt(self, index: int) -> None:
        """
        Sets the prompt index and updates the layer color if the layer is occupied.

        Args:
            index (int): The index of the prompt to set.
        """
        self.prompt_index = index
        if not self._is_free:
            self.replace_color(self.colors[self.prompt_index])

    def add(self, data: Any, *arg, **kwargs) -> None:
        """
        Adds data to the layer. If the layer is occupied, it removes the last item before adding the new data.

        Args:
            data (Any): The data to add to the layer.
        """
        if self._is_free:
            self._is_free = False
            self._add(data, *arg, **kwargs)
        else:
            self.remove_last()
            self._add(data, *arg, **kwargs)

        self.events.finished(action=ActionType.ADDED, value="ADD")

    def run(self) -> None:
        """
        Marks the layer as free and updates its color based on the current prompt index.
        """
        if self._is_free:
            return

        self._is_free = True
        self.replace_color(self.colors_set[self.prompt_index])

    def is_free(self) -> bool:
        """
        Checks if the layer is currently free.
        """
        return self._is_free

    @abstractmethod
    def replace_color(self, _color: List[float]) -> None:
        """
        Abstract method to replace the color of the layer.

        Args:
            _color (List[float]): The new RGBA color to apply.
        """

    @abstractmethod
    def remove_last(self) -> None:
        """
        Abstract method to remove the last item from the layer.
        """

    @abstractmethod
    def _add(self, data: Any, *arg, **kwargs) -> None:
        """
        Abstract method to add data to the layer.

        Args:
            data (Any): The data to add.
        """
