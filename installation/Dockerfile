# Dockerfile for project

# inherit from python image
FROM ghcr.io/napari/napari:sha-77fc0af

# update to latest pip version 
RUN pip uninstall opencv-python
RUN pip install opencv-python-headless
RUN pip install simpleitk

RUN pip install torch torchvision hydra-core iopath
RUN pip install nninteractive

WORKDIR /app

COPY ./napari-interactive /app/napari-interactive
RUN pip install -e /app/napari-interactive
COPY ./napari-interactive /app/napari-interactive
RUN pip install -e /app/napari-interactive
COPY ./napari-shifted-labels /app/napari-shifted-labels
RUN pip install -e /app/napari-shifted-labels
COPY ./napari-edit-log /app/napari-edit-log
RUN pip install -e /app/napari-edit-log

ENTRYPOINT [ "python3", "/app/startup.py" ]