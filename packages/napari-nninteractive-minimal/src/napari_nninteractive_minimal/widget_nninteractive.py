from qtpy.QtWidgets import QGroupBox

from napari_nninteractive import nnInteractiveWidget
from napari_nninteractive.layers.point_layer import SinglePointLayer

from napari_custom_layers import ManualLabelsLayer, PreviewLabelsLayer, FixedImageLayer

class nnInteractiveWidgetMinimal(nnInteractiveWidget):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._width = 250

        self.layout().setContentsMargins(0,0,0,0)

        # Modify the UI
        self.model_selection.parent().setHidden(True)
        self.image_selection.parent().setHidden(True)
        self.instance_aggregation_ckbx.setHidden(True)
        self.auto_refine.setHidden(True)
        self.auto_refine.parent().setHidden(True)
        self.propagate_ckbx.setHidden(True)
        self.run_ckbx.parent().setHidden(True)
        self.export_button.parent().setHidden(True)


    def add_preview_label_layer(self, data, name) -> None:
        """
        Check if a layer with the layer_name already exists. If yes rename this by adding an index
        and afterward create the layer
        :return:
        :rtype:
        """
        label_layer = PreviewLabelsLayer(
            data,
            name=name,
            opacity=0.9,
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            # colormap=self.colormap[index],
            metadata=self.session_cfg["metadata"],
        )
        label_layer.contour = 1
        label_layer.editable = False
    
        label_layer._source = self.session_cfg["source"]

        self._viewer.add_layer(label_layer)

    def add_label_layer(self, data, name) -> None:
        """
        Check if a layer with the layer_name already exists. If yes rename this by adding an index
        and afterward create the layer
        :return:
        :rtype:
        """
        if name == self.label_layer_name:
            self.add_preview_label_layer(data, name)
            return

        label_layer = ManualLabelsLayer(
            data,
            # self._data_result,
            name=name,
            opacity=0.9,
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            # colormap=self.colormap[index],
            metadata=self.session_cfg["metadata"],
        )
        label_layer.contour = 1
        label_layer._source = self.session_cfg["source"]
        label_layer.colormap = self.colormap[self.object_index]


        self._viewer.add_layer(label_layer)


    def add_point_layer(self) -> None:
        """Adds a single point layer to the viewer."""
        point_layer = SinglePointLayer(
            name=self.point_layer_name,
            ndim=self.session_cfg["ndim"],
            affine=self.session_cfg["affine"],
            scale=self.session_cfg["scale"],
            translate=self.session_cfg["translate"],
            rotate=self.session_cfg["rotate"],
            shear=self.session_cfg["shear"],
            metadata=self.session_cfg["metadata"],
            opacity=0.7,
            size=2,
            prompt_index=self.prompt_button.index,
        )

        # point_layer.size = 0.2
        point_layer.events.finished.connect(self.on_interaction)
        self._viewer.add_layer(point_layer)