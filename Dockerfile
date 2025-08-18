# Dockerfile for project

# inherit from python image
FROM ghcr.io/napari/napari:sha-77fc0af

# update to latest pip version 
RUN pip uninstall opencv-python
RUN pip install opencv-python-headless
RUN pip install simpleitk

RUN pip install torch torchvision hydra-core iopath

WORKDIR /app

COPY napari-interactive /app/napari-interactive
RUN pip install -e /app/napari-interactive

COPY start.sh /app/start.sh

ENTRYPOINT [ "/bin/bash", "start.sh" ]