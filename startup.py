input_image = "/app/example_data/2d+t_trackrad/A_003_frames_8bit.mha"

from napari import Viewer
import napari

import SimpleITK as sitk

viewer = Viewer()
example_img = sitk.GetArrayFromImage(sitk.ReadImage(input_image))
image_layer = viewer.add_image(example_img, name='Example Image', colormap='gray')

from napari_interactive import InteractiveSegmentationWidget

viewer.window.add_plugin_dock_widget(
    "napari-interactive",
    "Interactive Segmentation",
)
#viewer.dims.order = (2,0,1)
napari.run()
