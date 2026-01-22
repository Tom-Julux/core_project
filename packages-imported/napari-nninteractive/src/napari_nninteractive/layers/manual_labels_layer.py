from typing import Any, List

import napari
import numpy as np
from napari.layers import Labels
from napari.layers.base._base_constants import ActionType

class ManualLabelsLayer(Labels):
    """
    A bounding box layer class that extends `BaseLayerClass` and `Shapes` with specific color
    management and interaction handling. This class manages the addition, removal, and color
    updating of bounding boxes and restricts rotation.
    """

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
