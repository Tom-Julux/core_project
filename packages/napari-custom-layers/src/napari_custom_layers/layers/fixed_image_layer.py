from typing import Any, List

import napari
import numpy as np
from napari.layers import Image
from napari.layers.base._base_constants import ActionType

from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls
from napari_custom_layers.controls.fixed_image_controls import CustomQtFixedImageControls

class FixedImageLayer(Image):
    """
    A bounding box layer class that extends `BaseLayerClass` and `Shapes` with specific color
    management and interaction handling. This class manages the addition, removal, and color
    updating of bounding boxes and restricts rotation.
    """

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)

# register the custom layer controls
layer_to_controls[FixedImageLayer] = CustomQtFixedImageControls
