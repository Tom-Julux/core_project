from typing import Optional

import numpy as np

import napari
from napari.layers import Image, Labels, Shapes
from napari.viewer import Viewer

from napari._qt.layer_controls.qt_layer_controls_container import layer_to_controls

from napari_promptable.controls.bbox_controls import CustomQtBBoxControls
from napari_promptable.controls.lasso_controls import CustomQtLassoControls
from napari_promptable.controls.point_controls import CustomQtPointsControls
from napari_promptable.controls.scribble_controls import CustomQtScribbleControls
from napari.layers import Shapes, Points, Labels

class BoxPromptLayer(Shapes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
class PointPromptLayer(Points):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
class ContourPromptLayer(Shapes):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
class ScribblePromptLayer(Labels):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        

layer_to_controls[PointPromptLayer] = CustomQtPointsControls
layer_to_controls[BoxPromptLayer] = CustomQtBBoxControls
layer_to_controls[ScribblePromptLayer] = CustomQtScribbleControls
layer_to_controls[ContourPromptLayer] = CustomQtLassoControls

import SimpleITK as sitk

def show_promptable_demo() -> None:
    """Launches a Napari viewer and displays the widget gallery.

    This function creates a new Napari viewer, initializes the `GalleryWidget`,
    and adds it as a dock widget to the right side of the viewer window.

    The function then starts the Napari event loop.
    """

    example_img = sitk.GetArrayFromImage(
        sitk.ReadImage(
            '/Users/tomjulius/Developer/napari-plugins/example_data/A_005_frames_8bit.mha'
        )
    )
    example_labels = sitk.GetArrayFromImage(
        sitk.ReadImage(
            '/Users/tomjulius/Developer/napari-plugins/example_data/A_005_labels.mha'
        )
    )

    viewer = napari.Viewer()

    # Add an image layer
    image_layer = viewer.add_image(
        example_img,
        name='Example Image',
        colormap='gray'
    )
    # Add a labels layer
    labels_layer = viewer.add_labels(
        example_labels,
        name='Example Labels',
    )

    # rotate viewer
    viewer.dims.order = (2,0,1)

    # Add a BBox layer
    bbox_layer = BoxPromptLayer(name='BBox Layer', ndim=3)
    viewer.add_layer(bbox_layer)
    # Add a Lasso layer
    lasso_layer = ContourPromptLayer(name='Lasso Layer', ndim=3)
    viewer.add_layer(lasso_layer)
    # Add a Point layer
    point_layer = PointPromptLayer(name='Point Layer', ndim=3)
    viewer.add_layer(point_layer)
    # Add a Scribble layer
    data_ = np.zeros_like(example_img, dtype=np.uint8)
    scribble_layer = ScribblePromptLayer(
        data=data_,
        name='Scribble Layer'
    )
    viewer.add_layer(scribble_layer)


    napari.run()


if __name__ == "__main__":
    show_promptable_demo()
