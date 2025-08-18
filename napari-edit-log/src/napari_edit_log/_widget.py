import json
from pathlib import Path

import threading
import time
import cv2
from magicgui import magicgui
from napari.layers import Image
from typing import TYPE_CHECKING
from functools import partial
import numpy as np
from napari.utils.notifications import show_info, show_warning, show_error, show_console_notification
from napari import Viewer
from napari.layers import Labels
from napari_toolkit.containers import setup_scrollarea, setup_vcollapsiblegroupbox, setup_vgroupbox, setup_vscrollarea
from napari_toolkit.containers.boxlayout import hstack
from napari_toolkit.utils import set_value
from napari_toolkit.data_structs import setup_list
from napari_toolkit.utils.widget_getter import get_value
from napari_toolkit.widgets import (
    setup_checkbox,
    setup_combobox,
    setup_editcolorpicker,
    setup_editdoubleslider,
    setup_iconbutton,
    setup_label,
    setup_layerselect,
    setup_lineedit,
    setup_labeledslider,
    setup_pushbutton,
    setup_radiobutton,
    setup_savefileselect,
    setup_dirselect,
    setup_spinbox,
)
from napari.qt.threading import thread_worker
from qtpy.QtWidgets import (
    QFileDialog,
    QSizePolicy,
    QVBoxLayout,
    QWidget,
)

class EditLogWidget(QWidget):
    def __init__(self, viewer: Viewer):
        super().__init__()
        self._viewer = viewer


        self.metadata = {}
        self.edit_log = []
        self.recording = False

        self.build_gui()

        # All event listeners are always connected, but only log events when recording is True
        self.connect_layerlist_signals()
        for layer in self._viewer.layers:
            self.connect_layer_signals(layer)
        self._viewer.layers.events.inserted.connect(self.on_layer_added)
        self._viewer.layers.events.removed.connect(self.on_layer_removed)

    # GUI
    def build_gui(self):
        main_layout = QVBoxLayout(self)

        _scroll_widget, _scroll_layout = setup_vscrollarea(main_layout)
        # log view
        _container, _layout = setup_vcollapsiblegroupbox(_scroll_layout, "Log", False)


        self.past_state_list = setup_list(_layout, ["A"], True, function=lambda: print("QListWidget"))
        _ = setup_iconbutton(
            _layout, "Clear Log", "erase", self._viewer.theme, self.clear_log
        )
        # buttons for recording and exporting
        _container, _layout = setup_vgroupbox(_scroll_layout, "Edit Log Controls")

        self.toogle_recording_btn = setup_pushbutton(None, "Start Recording", function=self.toogle_recording)
        _ = hstack(_layout, [self.toogle_recording_btn])
        

        _container, _layout = setup_vcollapsiblegroupbox(_scroll_layout, "Edit log settings", False)

        self.clear_log_on_export_ckbx = setup_checkbox(_layout, "Clear log on export", True)
        self.continue_recording_after_export_ckbx  = setup_checkbox(_layout, "Continue recording after export", True)
    
        _container, _layout = setup_vgroupbox(_scroll_layout, "")
        _ = setup_iconbutton(
            _layout, "Export", "pop_out", self._viewer.theme, self.export_log
        )

    def export_log(self):
        print("Exporting edit log...")

        if self.edit_log and not self.edit_log[-1]['event_type'] == 'labels_update':
            # TODO: If the last event was labels update, the log does not contain the latest changes.
            # Pull the latest data from the last layer 
            self.edit_log[-1]["data"] = None
            pass

        file_path, _ = QFileDialog.getSaveFileName(self, "Save Edit Log", "", "JSON Files (*.json)")

        if not file_path:
            return

        with open(file_path, 'w') as f:
            json.dump(self.edit_log, f, indent=4)
        
        show_info(f"Edit log saved to {file_path}")

        if not self.continue_recording_after_export_ckbx.isChecked():
            self.toogle_recording(force_stop=True)

        if self.clear_log_on_export_ckbx.isChecked():
            self.clear_log()

    def clear_log(self):
        self.edit_log = []
        self.past_state_list.clear()
        print("Edit log cleared")

    def toogle_recording(self, force_stop=False):
        if self.recording or force_stop is True:
            self.toogle_recording_btn.setText("Start Recording")
            self.recording = False
        else:
            self.toogle_recording_btn.setText("Pause Recording")
            self.recording = True

    # Setting up event listeners
    def connect_layerlist_signals(self):
        # Connect signals to the viewer
        self._viewer.layers.events.inserted.connect(self.on_layer_event)
        self._viewer.layers.events.moved.connect(self.on_layer_event)
        self._viewer.layers.events.removed.connect(self.on_layer_event)
        self._viewer.layers.events.reordered.connect(self.on_layer_event)
        self._viewer.layers.events.changed.connect(self.on_layer_event)
        self._viewer.layers.selection.events.active.connect(self.on_layer_event)
        self._viewer.layers.selection.events.changed.connect(self.on_layer_event)

    def connect_layer_signals(self, layer):
        layer.events.data.connect(self.on_data_event)
        if isinstance(layer, Labels):
            # Labels layer has a specific event for label updates
            layer.events.labels_update.connect(self.on_labels_update_event)

    def disconnect_layer_signals(self, layer):
        layer.events.data.disconnect(self.on_data_event)
        if isinstance(layer, Labels):
            # Disconnect the specific label update event
            layer.events.labels_update.disconnect(self.on_labels_update_event)

    def on_layer_added(self, event):
        self.connect_layer_signals(event.value)

    def on_layer_removed(self, event):
        self.disconnect_layer_signals(event.value)
        
    # Event handlers that log events
    def on_layer_event(self, event):
        if not self.recording:
            return
        #print(f"Layer Event: {event.type}, Data: {event}")
        if hasattr(event, 'value'):
            self.past_state_list.addItem(f"Layer Event: {event.type}, Data: {event.value}")
        else:
            self.past_state_list.addItem(f"Layer Event: {event.type}, Data: {event}")
        print(f"Layer Event: {event.type}, Data: {event}")
        print(event.__dict__.keys())
        self.edit_log.append({
            'event_group': 'layer',
            'event_type': event.type,
            'data': str(event),
            'timestamp': time.time()
        })

    def on_labels_update_event(self, event):
        if not self.recording:
            return

        if event.type == "labels_update" and self.edit_log and self.edit_log[-1]['event_type'] == 'labels_update':
            self.edit_log[-1]["count"] = self.edit_log[-1].get("count", 0) + 1
            self.edit_log[-1]["last_event"] = time.time()
            print(f"Skipping layer update event: {event.type}, Data: {event}")
            return
            # Skip logging if the last event was also a layer update

        self.past_state_list.addItem(f"Edit Event: {event.type}, {event}")
        self.edit_log.append({
            'event_group': 'edit',
            'event_type': "labels_update",
            'count': 1,  # Count of updates
            'last_event': time.time(),
            'data': str(event),
            'timestamp': time.time()
        })

    def on_data_event(self, event):
        if not self.recording:
            return

        # Ignore in progress events like adding, removing, changing
        if hasattr(event, 'action') and event.action in ['adding', 'removing', 'changing']:
            return

        # Preview all attributes of the event
        for key, value in event.__dict__.items():
            if key.startswith('_'):
                continue
            print(f"{key}: {value}")

        
        if hasattr(event, 'action') and event.action in ['added', 'removed', 'changed']:
            self.past_state_list.addItem(f"Edit Event: {event.type}, {event.action}")
            self.edit_log.append({
                'event_group': 'edit',
                'event_type': "data",
                'action': event.action,
                'value': event.value,
                'timestamp': time.time()
            })
            return
        
        #print(f"Edit Event: {event.type}, Data: {event}")
        self.past_state_list.addItem(f"Edit Event: {event.type}, {event}")
        self.edit_log.append({
            'event_group': 'edit',
            'event_type': event.type,
            'data': str(event),
            'timestamp': time.time()
        })
