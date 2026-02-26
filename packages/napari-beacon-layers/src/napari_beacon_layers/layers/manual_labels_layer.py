from typing import Any, List

import napari
import numpy as np
from napari.layers import Labels
from napari.layers.base._base_constants import ActionType
from napari.utils.events import EmitterGroup, Event

from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls
from napari_beacon_layers.controls.manual_labels_control import CustomQtManualLabelsControls


class ManualLabelsLayer(Labels):
    """
    A bounding box layer class that extends `BaseLayerClass` and `Shapes` with specific color
    management and interaction handling. This class manages the addition, removal, and color
    updating of bounding boxes and restricts rotation.
    """

    def __init__(self, data, *args, **kwargs):
        super().__init__(data, *args, **kwargs)
        self.mode = 'paint'  # default mode is paint
        self._autofill = True  # whether to autofill connected components after painting
        self.events.add(autofill=Event) 
    

    @property
    def autofill(self):
        """bool: fill bucket changes only connected pixels of same label."""
        return self._autofill

    @autofill.setter
    def autofill(self, autofill):
        self._autofill = autofill
        self.events.autofill()

# register the custom layer controls
layer_to_controls[ManualLabelsLayer] = CustomQtManualLabelsControls
