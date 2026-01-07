# CoreTool

<img src="images/Screenshot 2025-09-24 at 12.21.15.png" loading="lazy" alt="CoreTool screenshot" width="500"/>

## What is this "CoreTool"?

CoreTool is a collection of napari plugins that provide an extensible interactive segmentation workflow for multidimensional images.

<details>
  <summary>Why "CoreTool" as a name?</summary>

  > The name "CoreTool" reflects the project's position at the core of the [BZKF Lighthouse on Local Therapies](https://bzkf.de/fileadmin/local-therapies.pdf) research project. 
  > The project aims to use promptable foundation models and other AI based tools to improve local therapies. For this purpose, the CoreTool provides a modular and extensible base for segmenting medical images using prompts.
</details>

## Overview
This repository contains a number of napari plugins centered around interactive segmentation:

- [**napari-promptable**](./packages/napari-promptable/) — the core plugin: an interactive segmentation widget that supports multi-object workflows and serves as the base class for model-specific plugins.

Based on this core plugin, several model-backed segmentation plugins are provided:
- [**napari-promptable-sam2**](packages/napari-promptable-sam2/) — segmentation in 2D, 2D+t, or 3D using a [sam2](https://github.com/facebookresearch/sam2)-based model
- [**napari-promptable-nnI**](packages/napari-promptable-nni/) — segmentation in 3D using a [nnInteractive](https://github.com/MIC-DKFZ/nnInteractive)-based model

It also contains some utility plugins:

- [**napari-edit-log**](./packages/napari-edit-log/) — logs user interactions to a file for replay and analysis.
- [**napari-shifted-labels**](./packages/napari-shifted-labels/) — visualizes masks across frames to provide a more consistent segmentation experience.
- [**napari-size-estimator**](./packages/napari-size-estimator/) — computes the volume of segmented objects in physical units.
- [**napari-shape-based-interpolation**](./packages/napari-shape-based-interpolation/) — shape based interpolation of labels between keyframes. (As an alternative to AI-based methods.)
- [**napari-quick-view**](./packages/napari-quick-view/) — quickly cycle through different images.

These pulgins are designed to work together for different applications. Examples are provided as *core_tool_apps* in the `core_tool_apps/` folder.

- [**core-tool-apps**](./core_tool_apps/)

The documentation for these plugins is currently work in progress.

## Table of contents

- [Installation](#installation)
- [Development & contributing](#development--contributing)
- [Roadmap](#roadmap)
- [License](#license)
- [Acknowledgments](#acknowledgments)

## Installation

### Requirements

The tool should in principle work wherever napari works. However, some plugins may have additional requirements (for example a CUDA-capable GPU for model inference). The project is also only tested on macOS and Linux.

### Local installation

For a local installation, clone the repository and install plugin packages in editable mode. We recommend [uv](https://github.com/astral-sh/uv) for this purpose.

```bash
# clone the repository
git clone <repo-url> core_project
cd core_project


# alternative 1: using uv (recommended)
# install uv
# curl -LsSf https://astral.sh/uv/install.sh | sh
# sync dependencies 
uv sync
# run the startup script. This opens napari with a demo loading plugin on the bottom left.
uv run startup.py

# alternative 2: using venv and pip:
# python3 -m venv .venv
# source .venv/bin/activate 
# for d in ./packages/napari-*/; do pip install -e "$d"; done
# pip install -e core_tool_apps/
# python3 startup.py

# alternative 3: starting napari manually and load the plugins from the plugins menu
napari
```

### In napari

In theory, you can also install the plugins directly from napari's plugin manager. However, this was not yet tested and will only be made available once the plugins are published to the napari hub.

### Docker

As an alternative to the local installation, the tool can be run inside a container. Note to make sure to mount any datasets and (when needed) model checkpoints into the container. The example below mounts a `/project_data/` folder from the host into the container. You also need to run the container with access to display devices (X11 or similar) to see the napari GUI.

```bash
# clone the repository
docker build -f ./Dockerfile -t napari_core_project .

# run the container with access to gpus, the files you want to access, and display devices
docker run --rm -it --gpus=all -v /project_data/:/project_data/ --device=/dev/dri:/dev/dri -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro napari_core_project

```

<details>
 <summary>Using the tool in the browser:</summary>
  
  Alternatively, for example if no display devices are available, [xpra](https://github.com/Xpra-org/xpra?tab=readme-ov-file#usage) can be used to strean the application to a browser.

  ```bash
  docker build -f ./Dockerfile.web -t napari_core_web .
  docker run --rm -it --gpus=all -v /project_data/:/project_data/ -p 9876:9876 napari_core_web
  ```
</details>

## Usage overview

Work in progress...

## Development & contributing

We welcome any contributions that align with the goals of the project.

### Development setup

1. Follow the [local installation](#local-installation) steps above.

2. Make any desired changes to the source code, for example adding a new model/plugin.

3. (optionally) add a demo for your new plugin in `demo_widget.py`.

4. Use the `startup.ipynb` notebook for (semi-)hot-reloading or run `uv run startup.py`. In both cases, napari will start and you can load your plugin from the plugins menu or the demo widget. With the notebook you can re-run cells to reload your code without restarting napari (best-effort basis).

## Roadmap

Planned short- and mid-term improvements:

- Better developer documentation and examples.
- Additional windowing and visualization plugins.
- Integration of more segmentation models.
- Advanced label-management tools and edit-log replay/analysis.
- Improved testing and CI/CD.
- Packaging and release to napari hub/pypi.

## License

This repository is licensed under MIT license.

The UI is partly based upon the [napari-toolkit](https://github.com/MIC-DKFZ/napari_toolkit) project. It is licensed under the terms of the [Apache Software License 2.0](https://github.com/MIC-DKFZ/napari_toolkit/blob/master/LICENSE) license.
The toolkit is typically imported as a library, but some files are copied and adapted for use in this repository. These files are marked with a header comment containing the original license information.

## Acknowledgments

This project is developed and maintained by the [LMU Adaptive Radiation Therapy Lab](https://lmu-art-lab.userweb.mwn.de/) (LMU ART Lab) at the  [Department of Radiation Oncology, LMU University Hospital](https://www.lmu-klinikum.de/strahlentherapie-und-radioonkologie/forschung/physikalische-forschung/5e34c41a1e300c37), Munich, Germany, in the context of the [BZKF Lighthouse on Local Therapies](https://bzkf.de/f/forschung/leuchttuerme/lokale-therapien/).


For more information about napari and related toolkits see:

- **napari**: https://github.com/napari/napari
- **napari_toolkit**: https://github.com/MIC-DKFZ/napari_toolkit