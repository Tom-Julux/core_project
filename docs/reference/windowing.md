# Windowing

Depending on the type of image loaded, you might want to adjust the windowing of the image. This is especially useful for CT images, where different window settings can highlight different tissues (e.g., bone, soft tissue, lung), or for x-ray images to enhance contrast.

To adjust the windowing, we recommend using the [napari-brightness-contrast](https://github.com/haesleinhuepf/napari-brightness-contrast) plugin. This plugin allows you to set custom window levels and widths for your images and includes a brightness histogram for better visualization.


A custom windowing plugin for the core-tool is planned
If desired, we can also implement a custom 