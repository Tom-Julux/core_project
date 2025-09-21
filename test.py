# %%
#%load_ext autoreload
#%autoreload 2

# %%
from napari import Viewer
import SimpleITK as sitk

# %%
viewer = Viewer()

# %%
example_img = sitk.GetArrayFromImage(
    sitk.ReadImage(
        #'/Users/tomjulius/Developer/napari-plugins/example_data/A_005_frames_8bit.mha'
        '/Users/tomjulius/Developer/core_project/example_data/3d mrlinac/aumc_lung_patient026__GTV.mha'
    )
)
# Add an image layer
image_layer = viewer.add_image(
    example_img,
    name='Example Image',
    colormap='gray'
)
# %%
#example_labels = sitk.GetArrayFromImage(
#    sitk.ReadImage(
#        '/Users/tomjulius/Developer/napari-plugins/example_data/A_005_labels.mha'
#    )
#)

# Add a labels layer
#labels_layer = viewer.add_labels(
#    example_labels,
#    name='Example Labels',
#)

# rotate viewer
#viewer.dims.order = (2,0,1)

# %%
from napari_interactive import InteractiveSegmentationWidget

#viewer.window.add_plugin_dock_widget(
#    "napari-interactive",
#    "Interactive Segmentation",
#)

viewer.window.add_plugin_dock_widget(
    "napari-interactive",
    "SAM2 3D",
)
import napari
napari.run()
