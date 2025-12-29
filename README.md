# CoreTool

<img src="images/Screenshot 2025-09-24 at 12.21.15.png" loading="lazy" alt="CoreTool screenshot" />

## What is this CoreTool?

CoreTool is a collection of small, documented napari plugins that provide an extensible interactive segmentation workflow for multidimensional images. The repository groups a set of core components and plugins so researchers and developers can prototype, extend, and evaluate promptable segmentation models (for example SAM-based models).

Main plugin(s) included in this repository:

- **napari-interactive** — the core plugin: an interactive segmentation widget that supports multi-object workflows and serves as the base class for model-specific plugins.
- **napari-interactive-2d-sam** — 2D plane segmentation using a SAM2-based model.
- **napari-interactive-2dt-sam** — 2D+t (or 3D) propagation: extends the 2D plugin with mask propagation between adjacent frames.
- **napari-interactive-3d-sam** — 3D segmentation using up to three orthogonal planes and a simple view-control UI.
- **napari-interactive-{2d,3d}-noregistration** — lightweight example plugins without ML models; useful for UI testing and as implementation templates.

Additional helper plugins:

- [**napari-edit-log**](./napari-edit-log/README.md) — logs user interactions to a file for replay and analysis.
- [**napari-shifted-labels**](./napari-shifted-labels/README.md)  — visualizes masks across frames to provide a more consistent segmentation experience.

The documentation for these plugins is available in their respective folders.

## Table of contents

- [Quick start](#quick-start)
- [Installation](#installation)
- [Development & contributing](#development--contributing)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Prerequisites

In principle, the CoreTool only requires a working napari installation and could be installed from the plugins menus. Some model-backed plugins (for example SAM2 variants) will require a CUDA-capable GPU and the matching PyTorch + CUDA runtime for best performance. The project is tested on macOS and Linux. Windows may work as well but may encounter platform-specific issues. Please report any problems.

### Local installation

For the local installation, clone the repository and install plugin packages in editable mode. We recommend [uv]() for this purpose. The script was tested on linux and macos.


```bash
# clone the repository
git clone <repo-url> core_project
cd core_project


# install uv
# curl -LsSf https://astral.sh/uv/install.sh | sh

# sync dependencies
uv sync

# run the startup script
uv run startup.py

# alternatively you can activate the venv created by uv and start the script directly
source .venv/bin/activate
python3 startup.py

# finally you can start napari and load the plugins from the plugins menu (inside the venv)
napari
```


### Docker

We also provide Docker images. The containers are large and meant for reproducible runs and CI, not lightweight local development. If you use Docker, make sure to mount any datasets and (when needed) model checkpoints.


```bash
# clone the repository
docker build -f ./Dockerfile -t napari_core_project .

# run the container with access to gpus, the files you want to access, and display devices
docker run --rm -it --gpus=all -v /project_data/:/project_data/ --device=/dev/dri:/dev/dri -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro napari_core_project

```

<details>
 <summary>Using the tool in the browser:</summary>
  
  You can also use [xpra](https://github.com/Xpra-org/xpra?tab=readme-ov-file#usage) to use the tool in your browser. Althouhgh you will only be able to load files available to the docker container.

  ```bash
  docker build -f ./Dockerfile.web -t napari_core_web .

  docker run --rm -it --gpus=all -v /project_data/:/project_data/ -p 9876:9876 napari_core_web

  ```
</details>

## Usage overview

The plugins share a common interaction model:

1. [Open an image](https://napari.org/stable/tutorials/fundamentals/quick_start.html#open-an-image) (drag & drop or `File → Open`).


2. Choose a plugin from the Plugins menu. The control panel appears on the right.

3. Select the image layer to segment. Changing the image layer will reset plugin state.

4. Choose a prompt type (Points, Box, Mask, etc.).

> Tip: Based on the selected prompt type, different prompt layers will appear in the __layerlist__ (typically on the left). For example, if you select the `Points` prompt type, two new layers will be created, one for foreground points and one for background points. For the `Box` prompt type, only a single layer for the bounding box will be created. For the `Mask` prompt type, a single layer for the input mask will be created.

5. With such a `prompt layer` selected, use the respective napari tool to add prompts to the image. For this purpose, use the layer control panel (typically on the top left) to input the desired prompt.

6. Optionally adjust `hyperparameters` in the control panel.

7. Optionally use the `View Control` section of the control panel if the widget requires certain additional steps. For example the 3D segmentation widget requires you to select up to three orthogonal planes in the viewer.

8. Use `AutoRun` (default) for live updates, or click `Predict` to run on demand.

9. For multi-object workflows, use the Multi Object controls to switch and label different targets.
> Tip: You can change the color and opacity of each label using the [napari-labels](https://github.com/MIC-DKFZ/napari-labels) plugin.

> Tip: Currently only non-overlapping objects are supported.

10. Optionally use post-prediction features, such as `Propagation` of masks to adjacent frames (for 2D+t or 3D images). These features are controlled via additional control panels, typically located below the `Predict` button.

11. Export results using the Export options (see below) or reset the session with `Reset`.

Tip: You can customize label colors and opacity with the `napari-labels` plugin.

## Development & contributing

We welcome any contributions that align with the goals of the project.

### Development setup

1. Follow the Local installation steps above.

2. Use the `development.ipynb` notebook for (semi-)hot-reloading during UI development or run `uv run startup.py`.

3. Make any desired changes to the source code.


### Adding support for new promptable models

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


<details>
  <summary>Tips for model authors:</summary>
  
  - Pick a comparable widget and copy its structure.
  - Implement `load_model`, `predict`, and `reset_model` following the existing patterns.
  - Prefer on-demand model download via `huggingface_hub` for large checkpoints, or document where to place local checkpoints. To store checkpoints locally, please use the `checkpoints/` folder.
</details>

### Using napari-interactive as a library

Alternatively, you can import the napari-interactive plugin class and create a new pip package for your model. See [here](https://napari.org/dev/plugins/building_a_plugin/first_plugin.html#your-first-plugin) on how to create a new napari-plugin.

1. Create a new plugin.

2. Add `napari-interactive` as a dependency to your project.

3. Import whatever widget class you want to use as parent and implement the plugin for your own model.

## Roadmap

Planned short- and mid-term improvements:

- Better developer documentation and examples.
- Automatic model loading from Hugging Face.
- Additional windowing and visualization plugins.
- Integration of more specialized segmentation models.
- Advanced label-management tools and edit-log replay/analysis.

## License

This repository is currently not licensed. It will be licensed under an open-source license in the near future upon release.

The UI is based upon the [napari-toolkit](https://github.com/MIC-DKFZ/napari_toolkit) project. It is licensed under the terms of the [Apache Software License 2.0](https://github.com/MIC-DKFZ/napari_toolkit/blob/master/LICENSE) license.
The toolkit is typically imported as a library, but some files are copied and adapted for use in this repository. These files are marked with a header comment containing the original license information.

## Acknowledgments

This project is developed and maintained by the [LMU Adaptive Radiation Therapy Lab](https://lmu-art-lab.userweb.mwn.de/) (LMU ART Lab) at the  [Department of Radiation Oncology, LMU University Hospital](https://www.lmu-klinikum.de/strahlentherapie-und-radioonkologie/forschung/physikalische-forschung/5e34c41a1e300c37), Munich, Germany, in the context of the [BZKF Lighthouse on Local Therapies](https://bzkf.de/f/forschung/leuchttuerme/lokale-therapien/).


For more information about napari and related toolkits see:

[napari]: https://github.com/napari/napari
[napari_toolkit]: https://github.com/MIC-DKFZ/napari_toolkit