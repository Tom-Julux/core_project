#!/bin/sh

# docker build if image not present
if ! docker images | grep -q napari_core_web_debug; then
    docker build -f ./Dockerfile.web-debug -t napari_core_web_debug .
fi

docker run --rm -it --gpus=all -v /project_data_2/:/project_data_2/ -v /project_data/:/project_data/ -v /project_data/mridian/mridian_tracking/tbloecker/code/core_project:/app -p 9876:9876 napari_core_web_debug
