# napari-edit-log

A plugin for recording edits made in napari. It was developed to track changes in label layers, shapes, and points layers within the napari image viewer, to enable a comprehensive analysis of the usage of the tool and to 

> **Note**: This plugin is currently in an early development stage and is not yet ready for production use. It is intended for testing and feedback purposes only.


## Features

- **Recording Edits**: The plugin captures edits made to label, shapes, and points layers, such as adding, removing, or modifying them. It also tracks the creation and deletion of new layers.  
- **History Functionality**: The plugin maintains a history of changes, enabling users to go back to previous states of the label layer.
- **Export Log**: The edit log can be exported to a json file, allowing users to save their edit history for future analysis.
- **Compression**: The plugin supports compression of the edit log to reduce file size.


## Installation

You can install `napari-labels` via [pip]:

```
pip install napari-edit-log
```

## Seaborn Color palettes

You can use all color palette names which are valid for `seaborn.color_palette()`.
An overview can be found here:

- https://r02b.github.io/seaborn_palettes/
- https://www.practicalpythonfordatascience.com/ap_seaborn_palette

## License

Distributed under the terms of the [Apache Software License 2.0] license,
"napari-labels" is free and open source software

## Acknowledgments

<p align="left">
  <img src="https://github.com/MIC-DKFZ/napari-labels/raw/main/imgs/Logos/HI_Logo.png" width="150"> &nbsp;&nbsp;&nbsp;&nbsp;
  <img src="https://github.com/MIC-DKFZ/napari-labels/raw/main/imgs/Logos/DKFZ_Logo.png" width="500">
</p>

This repository is developed and maintained by the Applied Computer Vision Lab (ACVL)
of [Helmholtz Imaging](https://www.helmholtz-imaging.de/) and the
[Division of Medical Image Computing](https://www.dkfz.de/en/medical-image-computing) at DKFZ.

This [napari] plugin was generated with [copier] using the [napari-plugin-template].

[apache software license 2.0]: http://www.apache.org/licenses/LICENSE-2.0
[copier]: https://copier.readthedocs.io/en/stable/
[napari]: https://github.com/napari/napari
[napari-plugin-template]: https://github.com/napari/napari-plugin-template
[pip]: https://pypi.org/project/pip/
