#!/bin/sh

echo "Building Docker image 'napari_core'..."
docker build -f ./installation/Dockerfile -t napari_core .

abspath() {
  case "$1" in
    /*) echo "$1" ;;
    *) echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" ;;
  esac
}

docker run --rm -it --gpus=all -v /project_data_2/:/project_data_2/\
    -v /project_data/:/project_data/\
    -v  "$(abspath ./example_data)":/app/example_data\
    -v  "$(abspath ./startup.py)":/app/startup.py\
    -v  "$(abspath ./demo_widget.py)":/app/demo_widget.py\
    --device=/dev/dri:/dev/dri -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro napari_core
