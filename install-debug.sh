#!/bin/sh

pip install napari[all]
pip install opencv2-python-headless
pip install SimpleITK
pip install -e napari-edit-log
pip install -e napari-interactive
pip install -e napari-label-metrics
#pip install -e napari-spacing-adjust