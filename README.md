
![](images/logos.png)

# BEACON - **B**ZKF int**E**ractive **A**i-based **CON**touring

A collection of napari plugins for interactive segmentation of multidimensional images, with a focus on medical imaging applications and high usability for non-technical users. It also contains a number of applications built on top of these plugins, which implement specific workflows for different use cases.


<details>
  <summary>Why "BEACON"?</summary>

  > The name "BEACON" reflects the project's origin as core project of the [BZKF Lighthouse on Local Therapies](https://bzkf.de/fileadmin/local-therapies.pdf), which is focused on improving local therapies through the use of AI-based tools.
</details>

<img src="images/Screenshot 2025-09-24 at 12.21.15.png" loading="lazy" alt="BEACON screenshot" width="500"/>


## Overview

This repository contains a number of napari plugins.


- [**napari-promptable**](./packages/napari-promptable/) — the core plugin: an interactive segmentation widget that supports multi-object workflows and serves as the base class for model-specific plugins.
Based on this core plugin, several model-backed segmentation plugins are provided:
- [**napari-promptable-sam2**](packages/napari-promptable-sam2/) — segmentation in 2D, 2D+t, or 3D using a [sam2](https://github.com/facebookresearch/sam2)-based model
- [**napari-promptable-nnI**](packages/napari-promptable-nni/) — segmentation in 3D using a [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive)-based model


- [**napari-nninteractive-minimal**](./packages/napari-nninteractive-minimal/) — a simplified version of the excelent [napari-nninteractive](https://github.com/MIC-DKFZ/napari-nninteractive) plugin for interactive segmentation using nnInteractive. This plugin removes some of the more advanced features of the original plugin to provide a more streamlined experience for non-technical users. It also prevents users from accidentally breaking out of the intended workflow, for example by accidentally loading a different image or modifying the napari viewer in a way that breaks the plugin.

- [**napari-manual-segmentation**](./packages/napari-manual-segmentation/) — a plugin for manual segmentation, with an workflow and UI aligned to the simplified napari-nninteractive-minimal plugin.

- [**napari-beacon-layers**](./packages/napari-beacon-layers/) — custom napari layers for visualizing and editing segmentations.

It also contains some utility plugins:

- [**napari-edit-log**](./packages/napari-edit-log/) — logs user interactions to a file for replay and analysis.
- [**napari-shifted-labels**](./packages/napari-shifted-labels/) — visualizes masks across frames to provide a more consistent segmentation experience.
- [**napari-inverted-scrolling**](./packages/napari-inverted-scrolling/) — inverts the scrolling behaviour in napari to match other software commonly used in medical imaging (scolling through frames with the mouse wheel instead of zooming).
- [**napari-size-estimator**](./packages/napari-size-estimator/) — computes the volume of segmented objects in physical units.
- [**napari-shape-based-interpolation**](./packages/napari-shape-based-interpolation/) — shape based interpolation of labels between keyframes. (As an alternative to AI-based methods.)
- [**napari-quick-view**](./packages/napari-quick-view/) — quickly cycle through different images.


## Table of contents

- [Installation](#installation)
- [Development & contributing](#development--contributing)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Requirements

BEACON can be used on any device that supports napari (Windows/Macos). However, some plugins may have additional requirements (for example a CUDA-capable GPU for model inference) for optimal experience. The project is tested on macOS and Linux.

### Local installation

For a local installation, clone the repository and install plugin packages in editable mode. We recommend [uv](https://github.com/astral-sh/uv) for this purpose.

```bash
# clone the repository and navigate into it
git clone <repo-url> core_project && cd core_project

# sync dependencies and install the plugins in editable mode
uv sync
# run the startup script. This opens napari with a demo study.
uv run startup.py
# alternativly start napari
uv run napari
```
<details>
  <summary>Alternativly using `pip` and `venv`</summary>

  ```bash
  python3 -m venv .venv
  source .venv/bin/activate 
  for d in ./packages/napari-*/; do pip install -e "$d"; done
  pip install -e ./apps/artist_study_app/
  python3 startup.py

  ```
</details>

### In napari

In theory, you can also install the plugins directly from napari's plugin manager. However, this was not yet tested and will only be made available once the plugins are published to the napari hub.

### Docker

As an alternative to the local installation, napari can be run inside a container. Note to make sure to mount any datasets and (when needed) model checkpoints into the container. The example below mounts the `/project_data/` folder from the host into the container. You also need to run the container with access to display devices (X11 or similar) to see the napari GUI.

```bash
# clone the repository
docker build -f ./Dockerfile -t napari_core_project .

# run the container with access to gpus, the files you want to access, and display devices
docker run --rm -it --gpus=all -v /project_data/:/project_data/ -v ./artist_study:/app/artist_study --device=/dev/dri:/dev/dri -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro napari_core_project

```

<details>
 <summary>Using the tool in the browser:</summary>
  
  Alternatively, for example if no display devices are available, [xpra](https://github.com/Xpra-org/xpra?tab=readme-ov-file#usage) can be used to stream the application to a browser.

  ```bash
  docker build -f ./Dockerfile.web -t napari_core_web .
  docker run --rm -it --gpus=all -v /project_data/:/project_data/ -p 9876:9876 napari_core_web
  ```
</details>

## Usage overview

To learn how to use napari and the various plugins provided in this repository, please make use of the demos provided in the demo widget (loaded to the bottom left when starting the tool with `startup.py`).


Additionally please refer to the [Guide](./Guide.md) for video tutorials covering installation and usage of the tool. Finally, the individual plugins also contain documentation on their usage in their respective folders.

## Development & contributing

We welcome any contributions that align with the goals of the project.

### Development setup

1. Follow the [local installation](#local-installation) steps above.

2. Make any desired changes to the source code, for example adding a new model/plugin.

3. (optionally) add a demo for your new plugin in `demo_widget.py`.

4. Use the `startup.ipynb` notebook for (semi-)hot-reloading. With the notebook you can re-run cells to reload your plugin/demo without restarting napari. (best-effort basis might fail for complex changes/plugins)

5. Alternatively, you can also start napari manually with `uv run startup_test.py` and load the plugin you are working on from the plugin menu. (Tipp: modify the `startup_test.py` to load your plugin of interest on startup.)

## Roadmap

Planned short- and mid-term improvements:


- Packaging and release to napari hub/pypi.

## License

This repository is licensed under MIT license.

The UI is partly based upon the [napari-toolkit](https://github.com/MIC-DKFZ/napari_toolkit) project. It is licensed under the terms of the [Apache Software License 2.0](https://github.com/MIC-DKFZ/napari_toolkit/blob/master/LICENSE) license.
The toolkit is typically imported as a library, but some files are copied and adapted for use in this repository. These files are marked with a header comment containing the original license information.

## Acknowledgments

This project is developed and maintained by the [LMU Adaptive Radiation Therapy Lab](https://lmu-art-lab.userweb.mwn.de/) (LMU ART Lab) at the  [Department of Radiation Oncology, LMU University Hospital](https://www.lmu-klinikum.de/strahlentherapie-und-radioonkologie/forschung/physikalische-forschung/5e34c41a1e300c37), Munich, Germany, in the context of the [BZKF Lighthouse on Local Therapies](https://bzkf.de/f/forschung/leuchttuerme/lokale-therapien/).

![](images/logos.png)

For more information about napari and the ui toolkit used:

- **napari**: https://github.com/napari/napari
- **napari_toolkit**: https://github.com/MIC-DKFZ/napari_toolkit

For more information on the models used in the model-backed plugins:
- **nnInteractive**:
- **napari-nninteractive**: