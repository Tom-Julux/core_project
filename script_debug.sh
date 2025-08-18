#!/bin/sh

# docker build if image not present
if ! docker images | grep -q napari_core_debug; then
    docker build -f ./Dockerfile.debug -t napari_core_debug .
fi

docker run --rm -it --gpus=all -v /project_data_2/:/project_data_2/ -v /project_data/:/project_data/ -v /project_data/mridian/mridian_tracking/tbloecker/code/core_project:/app --device=/dev/dri:/dev/dri -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro napari_core_debug
