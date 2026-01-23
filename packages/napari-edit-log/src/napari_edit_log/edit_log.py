import time
import base64
import zlib
from typing import TYPE_CHECKING
import numpy as np
from napari import Viewer
from napari.layers import Labels

from napari_edit_log.utils import encode_labels_event_data, decode_labels_event_data
from napari.utils.events import EventEmitter, EmitterGroup, Event, EventedList
import fnmatch

class NapariEditLog():
    def __init__(self, viewer: Viewer):
        self._viewer = viewer

        self._log = EventedList()
        self._is_recording = False
        self._has_active_edit_series = False

        # Define events
        self.events = EmitterGroup(self,
            recorded = Event,
            updated = Event,
            cleared = Event,
            started = Event,
            stopped = Event
            )  
    
        # All event listeners are always connected, but only log events when recording is True
        self.connect_layerlist_signals()
        for layer in self._viewer.layers:
            self.connect_layer_signals(layer)
        self.connect_dims_signals()

    # Cleanup on delete
    def __del__(self):
        try:
            self.disconnect_layerlist_signals()
            for layer in self._viewer.layers:
                self.disconnect_layer_signals(layer)
            self.disconnect_dims_signals()
        except Exception as e:
            pass

    # Properties    
    @property
    def log(self) -> list:
        return self._log
        
    @property
    def is_recording(self) -> bool:
        return self._is_recording

    # Public methods
    def clear(self):
        self._log = []
        self._has_active_edit_series = False
        self.events.cleared()

    def start(self):
        self._is_recording = True
        self._has_active_edit_series = False
        self.events.started()

    def stop(self):
        self._is_recording = False
        self._has_active_edit_series = False
        self.events.stopped()

    def toggle(self, force_stop=False):
        if self._is_recording or force_stop is True:
            self.stop()
        else:
            self.start()

    def record(self, event_dict):
        if not self._is_recording:
            return
        
        self._log.append(event_dict)
        self.events.recorded()

    # Setting up event listeners
    def connect_dims_signals(self):
        # Connect signals to the viewer
        self._viewer.dims.events.order.connect(self.on_dims_event)
        self._viewer.dims.events.point.connect(self.on_dims_event)
    
    def disconnect_dims_signals(self):
        # Connect signals to the viewer
        self._viewer.dims.events.order.disconnect(self.on_dims_event)
        self._viewer.dims.events.point.disconnect(self.on_dims_event)
    
    # Event listener setup
    def connect_layerlist_signals(self):
        # Connect signals to the viewer
        self._viewer.layers.events.inserted.connect(self.on_layer_event)
        self._viewer.layers.events.moved.connect(self.on_layer_event)
        self._viewer.layers.events.removed.connect(self.on_layer_event)
        self._viewer.layers.events.reordered.connect(self.on_layer_event)
        self._viewer.layers.selection.events.active.connect(self.on_layer_event)

    def disconnect_layerlist_signals(self):
        # Connect signals to the viewer
        self._viewer.layers.events.inserted.disconnect(self.on_layer_event)
        self._viewer.layers.events.moved.disconnect(self.on_layer_event)
        self._viewer.layers.events.removed.disconnect(self.on_layer_event)
        self._viewer.layers.events.reordered.disconnect(self.on_layer_event)
        self._viewer.layers.selection.events.active.disconnect(self.on_layer_event)

    def connect_layer_signals(self, layer):
        layer.events.data.connect(self.on_data_event)
        if isinstance(layer, Labels):
            # Labels layer has a specific event for label updates
            layer.events.labels_update.connect(self.on_labels_update_event)
            layer.events.selected_label.connect(self.on_labels_tool_event)
            layer.events.mode.connect(self.on_labels_tool_event)

    def disconnect_layer_signals(self, layer):
        layer.events.data.disconnect(self.on_data_event)
        if isinstance(layer, Labels):
            # Disconnect the specific label update event
            layer.events.labels_update.disconnect(self.on_labels_update_event)
            layer.events.selected_label.disconnect(self.on_labels_tool_event)
            layer.events.mode.disconnect(self.on_labels_tool_event)

    def on_layer_added(self, event):
        self.connect_layer_signals(event.value)

    def on_layer_removed(self, event):
        self.disconnect_layer_signals(event.value)
    
    # Event handlers that log events
    def on_dims_event(self, event):
        self._has_active_edit_series = False
        if not self._is_recording:
            return
        
        self.record({
            'event_group': 'dims',
            'event_type': event.type,
            'order': self._viewer.dims.order,
            'current_step': self._viewer.dims.current_step,
            'timestamp': time.time()
        })

        self.events.recorded()

    def on_labels_tool_event(self, event):
        # Changing the active layer ends any current edit series
        self._has_active_edit_series = False
            
        if not self._is_recording:
            return

        if event.type == "selected_label" and len(self._log) > 1 and self._log[-1]['event_type'] == 'selected_label':
            # when repeatedly changing selected label - update last event
            self._log[-1]['data'] = str(event)
            self._log[-1]['timestamp'] = time.time()
            self._log[-1]['selected_label'] = event._sources[0].selected_label
            self._log[-1]['mode'] = event._sources[0].mode

            self.events.updated()
            return
        
        self.record({
            'event_group': 'labels_tool',
            'event_type': event.type,
            'layer_name': event._sources[0].name,
            'selected_label': event._sources[0].selected_label,
            'mode': event._sources[0].mode,
            'data': str(event),
            'timestamp': time.time()
        })

    # Event handlers that log events
    def on_layer_event(self, event):
        self._has_active_edit_series = False

        self.record({
            'event_group': 'layer',
            'event_type': event.type,
            'data': str(event),
            'timestamp': time.time()
        })


    def on_labels_update_event(self, event):
        if not self._is_recording:
            return        
        
        # add current time to the event for logging
        event.time = time.time()
        
        # add event to the widget for debugging -> access via napari console etc.
        # self.labels_update_event = event

        if self._has_active_edit_series:
            #event.type == "labels_update" and self._log and self._log[-1]['event_type'] == 'labels_update':
            # skip logging if the last event was also a label layer update
            # add new edit
            self._log[-1]['data_edits'].append(encode_labels_event_data(event, base64=True))
            # store time of final labels update in this series of edits
            self._log[-1]["last_event"] = time.time()

            self.events.updated()
            return

        # compute state before edits    
        source_layer = event._sources[0]

        transposed_step = np.array(self._viewer.dims.current_step)[np.array(self._viewer.dims.order)]
        transposed_data = self._viewer.layers[1].data.transpose(self._viewer.dims.order)
        current_view = transposed_data[tuple(transposed_step)[:-2]]

        changed = np.zeros_like(current_view)
        changed[event.offset[0]:event.offset[0]+event.data.shape[0],
                event.offset[1]:event.offset[1]+event.data.shape[1]] = event.data

        before_change = np.logical_xor(current_view, changed)

        self.record({
            'event_group': 'edit',
            'event_type': "labels_update",
            'timestamp': time.time(), # time of the first labels update in a series of edits
            'last_event': time.time(), # time of the last labels update in a series of edits
            'layer_name': source_layer.name, # name of the layer being edited
            'layer_shape': source_layer.data.shape, # shape of the layer being edited
            "viewer_dims_order": self._viewer.dims.order, # current viewer dims order
            "viewer_dims_current_step": self._viewer.dims.current_step, # current viewer dims step
            'data_edits': [encode_labels_event_data(event, base64=True)], # list of encoded edits
            'data_initial': before_change.dtype.char + base64.b64encode(zlib.compress(before_change.tobytes())).decode('utf-8')
        })

    def on_data_event(self, event):
        # Event for when the data of any layer changes
        if not self._is_recording:
            return

        # Ignore in progress events like adding, removing, changing
        if hasattr(event, 'action') and event.action in ['adding', 'removing', 'changing']:
            return

        # Preview all attributes of the event
        #for key, value in event.__dict__.items():
        #    if key.startswith('_'):
        #        continue
        #    print(f"{key}: {value}")
        
        if hasattr(event, 'action') and event.action in ['added', 'removed', 'changed']:
            self.record({
                'event_group': 'edit',
                'event_type': "data",
                'action': event.action,
                'value': event.value,
                'timestamp': time.time()
            })
        else:
            self.record({
                'event_group': 'edit',
                'event_type': event.type,
                'data': str(event),
                'timestamp': time.time()
            })
