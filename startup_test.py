import numpy as np
from skimage import data, measure
from qtpy.QtCore import Qt
import napari

np.random.seed(1)
viewer = napari.Viewer()
import SimpleITK as sitk
img_sitk = sitk.ReadImage("/Users/tomjulius/Developer/core_project/example_data/3d mrlinac/aumc_lung_patient031__GTV.mha")
img_array = sitk.GetArrayFromImage(img_sitk)
img_layer = viewer.add_image(img_array, name='GTV')

def inverted_scrolling(viewer, event):
    if "Control" in event.modifiers:
        # do normal zooming
        viewer.window._qt_viewer.canvas._scene_canvas.__class__.__bases__[0]._process_mouse_event(
            viewer.window._qt_viewer.canvas._scene_canvas, event)
        event.handled = True
    else:
        if event.native.inverted():
            viewer.dims._scroll_progress += event.delta[1]
        else:
            viewer.dims._scroll_progress -= event.delta[1]
        while abs(viewer.dims._scroll_progress) >= 1:
            if viewer.dims._scroll_progress < 0:
                viewer.dims._increment_dims_left()
                viewer.dims._scroll_progress += 1
            else:
                viewer.dims._increment_dims_right()
                viewer.dims._scroll_progress -= 1
        event.handled = True
viewer.mouse_wheel_callbacks.append(inverted_scrolling)
from napari.components._viewer_mouse_bindings import dims_scroll
viewer.mouse_wheel_callbacks.remove(dims_scroll)

if __name__ == '__main__':
    napari.run()
