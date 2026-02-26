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


if __name__ == '__main__':
    napari.run()
