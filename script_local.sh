#!/bin/sh

# docker build if image not present
#if ! docker images | grep -q napari_core; then
echo "Building Docker image 'napari_core'..."
docker build ./ -t napari_core
#fi

docker run --rm -it --gpus=all\
    -v ./MedSAM2_latest.pt:/app/MedSAM2_latest.pt\
    -v ./example_data:/app/example_data\
    -v ./startup.py:/app/startup.py\
    -v ./demo_widget.py:/app/demo_widget.py\
    --device=/dev/dri:/dev/dri -e DISPLAY=$DISPLAY -v /tmp/.X11-unix:/tmp/.X11-unix:ro napari_core
