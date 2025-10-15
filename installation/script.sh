#!/bin/sh

# docker build if image not present
#if ! docker images | grep -q napari_core; then
    echo "Building Docker image 'napari_core'..."
    docker build ./ -t napari_core
#fi

docker run --rm -it --gpus=all -v /project_data_2/:/project_data_2/\
    -v /project_data/:/project_data/\
    -v /project_data/mridian/mridian_tracking/tbloecker/code/core_project/MedSAM2_latest.pt:/app/MedSAM2_latest.pt\
    -v /project_data/mridian/mridian_tracking/tbloecker/code/core_project/example_data:/app/example_data\
    -v /project_data/mridian/mridian_tracking/tbloecker/code/core_project/startup.py:/app/startup.py\
    -v /project_data/mridian/mridian_tracking/tbloecker/code/core_project/demo_widget.py:/app/demo_widget.py\
    --device=/dev/dri:/dev/dri -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro napari_core
