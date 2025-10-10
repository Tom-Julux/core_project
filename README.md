# CoreTool

<img src="images/Screenshot 2025-09-24 at 12.21.15.png"/>

## What is the CoreTool?

It consists of a number of indivitual napari plugins:

- **napari-interactive** is a core-plugin of the core-tool. It implements an interactive segmentation widget for multidimensional images. The plugin supports multi-object segmentation, and can be used as a base for other plugins. The following plugins are derived versions of this plugin for specific models and use-cases.

- **napari-interactive-2d-sam** uses the base-plugin to implement image segmentation based on SAM2 in 2D planes.

- **napari-interactive-2dt-sam** extends the 2D SAM plugin with the ability to propagate masks to adjacent frames in 2D+t or 3D images.

- **napari-interactive-3d-sam** extends the base-plugin to support segmentation in 3D images, based on up to three orthogonal planes. For this purpose, a view control section is implemented, to select the planes in the viewer.

- **napari-interactive-{2d,3d}-noregistration** are placeholder plugins and do not contain any machine learning model. They can be used as a base for implementing new models or for testing of the UI.

------ 

The following are additional, mostly self-contained plugins that enhance the interactive segmentation experience:

- **napari-edit-log** is plugin that logs every user interaction and stores it in a log-file. This log-file can then be used to analyize to usage of the tool.

- **napari-shifted-preview** allows you to display masks across frames/layers, to ensure a more consistent segmentation experience.

## Installation

#### Prerequisites

The tool supports any device on which napari can run. However, the models powering the interactive segmentation might require specific hardware (GPUs with sufficient memory) or operation systems.

The tool is tested by the developers on MacOS and Linux. (If you are a windows user, please let us know of any issues you encounter.)

### Local Installation

For the local installation, you clone the repository and run `pip install -e` for all included plugin folders. If you now start napari, the plugins will appear in the plugins menu.

```bash
# Clone the repository

git clone

# Install the plugins
for plugin in /app/napari-*; do
    if [ -d "$plugin" ]; then
        echo "Installing plugin: $plugin"
        pip install -qq -e "$plugin"
    else
        echo "Skipping non-directory: $plugin"
    fi
done
``` 

### Installing only some plugins

In case you only need to use some specific plugin, already have napari installed, do not need/what them demo-widget, and do not need a development setup, you can install the plugins individually using pip.

```bash
# napari-interactive
pip install git+https://git.repo/some_repo.git#egg=$NAME_OF_PACKAGE&subdirectory=$SUBDIR_IN_REPO
```

### Alternatively using Docker-based

Alternatively, ou can use Docker to use the tool without installing it locally. Note that you might need to update the mounted paths, such that you can access your imaging files. The container image is quite large and cannot easily be used for local development.

>

## Getting Started / Overview

Use one of these three options to start napari and activate the core tool plugins you want to use. For the start we recommend loading the demo widget, to get to know the napari and plugin widget UI. To load the demo widget run the following command in your CLI.

```python
python3 startup.py
```

We recommend opening the 2D SAM2 demo while reading along the Usage guide below.

## Usage

This section explains the overall princples of using the interactive segmentation plugins. The exact workflow might differ slightly between the different plugins, but the overall princples are the same.

0. [Open an image in napari](https://napari.org/stable/tutorials/fundamentals/quick_start.html#open-an-image). This can be done by drag-and-drop, or via the `File -> Open` menu. 

> Tip: Napari can open most image formats, with plugins available for more exotic formats. This leads to you maybe beeing asked with a dialog to select a plugin for opening your image. In most cases the default option (napari builtins) should work fine.

1. Select the plugin you want to use from the plugins menu. The plugin will open a control panel on the right side of the napari viewer.

2. At the top of this control panel you can select the **image layer** you want to segment. Changing the desired image layer will reset the plugin state, so make sure to select the correct layer first.

3. Select the **prompt type** you want to use. Based on the selected prompt type, different prompt layers will appear in the __layerlist__ (typically on the left). For example, if you select the `Points` prompt type, two new layers will be created, one for foreground points and one for background points. For the `Box` prompt type, only a single layer for the bounding box will be created. For the `Mask` prompt type, a single layer for the input mask will be created.

4. When a `prompt layer` is selected, you can use the respective napari tool to add prompts to the image. For this purpose, use the layer control panel (typically on the top left) to input the desired prompt.

5. (optional) You can change the `hyperparameters` of the underlying model using the control panel. The available hyperparameters depend on the specific model and plugin you are using.

6. (optional) Depending on the widget type, the widget might require certain additional steps. For example the 3D segmentation widget requires you to select up to three orthogonal planes in the viewer. This is controlled/informed via the `View Control` section of the control panel located above the `Propagation` panel. Other widgets currently do not require any additional steps.

7. If the `AutoRun` option is enabled, a new prediction will be generated anytime you change the prompt or any hyperparameter. If it is disabled, you can use the `Predict` button to generate a new prediction.

8. (optional) If you want to segment multiple objects, use the `Multi Object` section of the interactive segmentation plugin. Here you can switch between object ids to segment different objects.

> Tip: You can change the color and opacity of each label using the [napari-labels](https://github.com/MIC-DKFZ/napari-labels) plugin.

> Tip: Currently only non-overlapping objects are supported.

9. (optional) Some plugins support additional features, such as `propagation` of masks to adjacent frames (for 2D+t or 3D images). These features are controlled via additional control panels, typically located below the main interactive segmentation plugin panel.

10. Use the `Export` options to export the current segmentation mask to a new layer or to a file.

11. Use the `Reset` button to clear the current prompt and prediction, and reset the underlying model.


### 2D+t/3D propagation
TODO

### 3D segmentation from 2D planes
TODO

## Roadmap

The goal for the core tool was to create a shared, highly usable tool for researchers to view arbitrary types of images and make use of promptable segmentation model for interactive segmentation.

This goal was reached with the v0-beta version of the tool in October 2025. 

The next step is to roll out the tool to researchers involved with the [BZKF lighthouse on local therapies](https://bzkf.de/f/forschung/leuchttuerme/lokale-therapien/). Additonally, we will work on a study to evaluate AI-assisted segmentation of GTVs in MRgRT in a clinical setting. Finally, there are a number of additonal features that are planned to be added to the tool.

- Better documentation, including a developer guide for adding new models
- Automatic loading of models from huggingface
- A custom windowing plugin
- Integration of more machine learning models, including models for specific organs or pathologies
- A custom plugin for better label management
- Extension of the edit log plugin with an replay function and automated analysis tools

To tool is open-source and for the time being any PRs are welcome.

## Reference

The following section contains detailed information on how to use the tool. Feel free to use it as a reference when using the core tool, or to.

### Importing/Opening images

See the napari documentation on [how to open images](https://napari.org/stable/tutorials/fundamentals/quick_start.html#open-an-image) for more information.

> Tip: Napari can open most image formats, with plugins available for more exotic formats. This leads to you maybe beeing asked with a dialog to select a plugin for opening your image. In most cases the default option (napari builtins) should work fine.

### Prediction

The predict control panel contains a single button to start the prediction of a segmentation mask, based on the currently provided prompt.


<img src="images/Screenshot 2025-09-24 at 12.22.23.png"/>

> By default, the autorun option is enabled and a new prediction is generated anytime the prompt or  the hyperparameters change. This is great for near instantanious feedback, but can be irretating if the prediction takes more than a few moments. For this case the autorun option can be disabled using the checkbox.

### Exporting masks

Masks generated by the tool can be exported in two ways. 

The `Export to layer` button copies the current state of the preview layer onto a seperate layer. This layer can then be used for other purposes or exported using the default `Save` option from napari.

The `Export to file` option allows for the direct export to the file system, under the provided path.

In any case the shape/dimensionality of the image is preserved and the dtype set to uint8.

<img src="images/Screenshot 2025-09-24 at 12.22.34.png"/>

### Reset

The `Reset` button clear the preview layer, the prompt layers and resets the underlying segmentation model.

<img src="images/Screenshot 2025-09-24 at 12.22.41.png"/>

### Asthetics

> Tip: You can change the color and opacity of each label using the [napari-labels](https://github.com/MIC-DKFZ/napari-labels) plugin.

### Windowing

To apply windowing to the image in the napari viewer (or to adjust the contrast), we recommend using the [napari-brightness-contrast](https://github.com/haesleinhuepf/napari-brightness-contrast) plugin. This plugin allows you to set custom window levels and widths for your images and includes a brightness histogram for better visualization.

<img src="images/Screenshot 2025-10-09 at 18.22.01.png"/>

### Multi Object Mode

The interactive 

<img src="images/Screenshot 2025-09-24 at 12.22.06.png"/>

### Propagaton

The propagation control plane exists for the 2D+t plugin. It allows for the propagation of the masks present in the current frame to be propagated to the next frame. 
The dimension in which the propagation occurs can be set with the spin-box. The direction of the propagation can be reversed with the reverse checkbox. The overwrite existing checkbox controlls if any existing delineations in the next frame should be overwritten by the propagation.

Multiple objects (with the same label index) can be propagated in parallel.

The step button is used to propagate from the current frame to the next frame, and make the new frame visible.
The run button starts a continous propagation (similar to repetedly clicking the step button), until it is pressed again.

<img src="images/Screenshot 2025-09-24 at 12.22.15.png"/>

> Tip: To continously segment the same target use the run option, as (at least for SAM2) only the run option uses the full 8-frame memory bank. Contounous clicking the Step button can also work, but only uses the last mask as prompt, thus providing less high quailty segmentation.

### The Edit Log

### Keyboard control

Napari (and the plugins of the core-tool) support a variety of keyboard shortcuts for controlling the viewer. They are mostly also tied to UI elements and can be discoverd by hovering with the pointer over any given UI element.

The most useful viewer shortcuts (for the purpose of interactive segmentation) include:

- **Switching between the tools**: "Number keys" (1-9)
- **Moving between frames**: "Arrow keys" (left / right)
- ...

The most usefull shortcuts from the core project plugins include:

- **Add new label**: `N`
- ...

> Tip: You can also customize these shortcuts to your liking in the [napari preferences menu](https://napari.org/dev/guides/preferences.html#shortcuts-settings).

### DemoWidget

The Demo Widget was developed to enable an easy showcase the core tool. It loads example imaging data and a correspoding interactive segmentation plugin.

<img src="images/demo_widget.png"/>

> Tip: Adding an new plugin/extended widget to the demo allows you to quickly reload the widget after changes to its code.

## Contributing

This plugins in this repository can be easily extended with support for new segmentation models.

PRs and other contributions from third parties are considered if they fit the overall goal of the core project.

### Development setup

1. Follow the local setup guide above.

2. Run the `development.ipynb` notebook to start napari and load the demo plugin. This way, hot-reloading is enabled. Alternatively, you can use the `startup.py` file to check the setup without hot-reloading

3. Make any desired changes to the source code.

### Adding support for new models

1. Follow the local setup guide above.

2. Select what mode/dimensionality your model/plugin should support. The core tool currently supports segmentation in one 2D plane, propagation from one 2D plane to an adjacent one (for 3D or 2D+t segmenation), and segmentation in 3D based on up to three prompts in orthogonal planes. Other modes can be implemented by extending the __view_control__ section of the base widget, see the [3D segmentataion](napari-interactive/src/napari_interactive/_widget_3d_sam.py) widget as an example.

3. Select a comperable widget and clone the respective file in `napari-interactive/src/napari_interactive/`.

4. Rename the copied file, for example `_widget_<mode>_<model>.py`.

5. Rename the widget class in that file.

6. Add the newly created widget class to the `__init__.py` and `napari.yaml` files.

7. (optionally) Create a demo to quickly load your new plugin, by extending the `demo_widget.py` file.

8. (optionally) use the `development.ipynb` notebook to start napari and enable hot-reloading of your plugin.

9. Start developing.

10. For loading model checkpoints we recomment hosting them on huggingface and downloading them on-demand via `huggingface_hub`. Alternatively, you can store them somewhere local.

### Using napari-interactive as a library

Alternatively, you can import the napari-interactive plugin class and create a new pip package for your model. See [here](https://napari.org/dev/plugins/building_a_plugin/first_plugin.html#your-first-plugin) on how to create a new napari-plugin.

1. Create a new plugin.

2. Add `napari-interactive` as a dependencie to your `requirements.txt`.

3. Import whatever widget class you want to use as parent and implement the plugin for your own model.

## License

This repository is currently not licensed. It will be licensed under an open-source license in the near future upon release.

______________________________________________________________________

## Acknowledgments

This repository is developed and maintained by the LMU Adaptive Radiation Therapy Lab (LMU ART Lab)
of the [Department of Radiation Oncology, LMU University Hospital](https://www.lmu-klinikum.de/strahlentherapie-und-radioonkologie/forschung/physikalische-forschung/5e34c41a1e300c37), Munich, Germany, in the context of the [BZKF Lighthouse on Local Therapies](https://bzkf.de/f/forschung/leuchttuerme/lokale-therapien/).

[napari]: https://github.com/napari/napari

[napari_toolkit]: https://github.com/MIC-DKFZ/napari_toolkit