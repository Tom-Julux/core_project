#!/bin/sh

# docker build if image not present
#if ! docker images | grep -q napari_core_web; then
    echo "Building Docker image 'napari_core_web'..."
    docker build -f ./Dockerfile.web -t napari_core_web .
#fi

docker run --rm -it --gpus=all -v /project_data_2/:/project_data_2/\
    -v /project_data/:/project_data/\
    -v /project_data/mridian/mridian_tracking/tbloecker/code/core_project/MedSAM2_latest.pt:/app/MedSAM2_latest.pt\
    -v /project_data/mridian/mridian_tracking/tbloecker/code/core_project/example_data:/app/example_data\
    -v /project_data/mridian/mridian_tracking/tbloecker/code/core_project/startup.py:/app/startup.py\
    -v /project_data/mridian/mridian_tracking/tbloecker/code/core_project/demo_widget.py:/app/demo_widget.py\
    -p 9876:9876 napari_core_web
