#!/bin/sh

# docker build if image not present
#if ! docker images | grep -q napari_core_web; then
echo "Building Docker image 'napari_core_web'..."
docker build -f ./installation/Dockerfile.web -t napari_core_web .
#fi

abspath() {
  case "$1" in
    /*) echo "$1" ;;
    *) echo "$(cd "$(dirname "$1")" && pwd)/$(basename "$1")" ;;
  esac
}

docker run --rm -it --gpus=all -v /project_data_2/:/project_data_2/\
    -v /project_data/:/project_data/\
    -v  "$(abspath ../example_data)":/app/example_data\
    -v  "$(abspath ../startup.py)":/app/startup.py\
    -v  "$(abspath ../demo_widget.py)":/app/demo_widget.py\
    -p 9876:9876 napari_core_web
