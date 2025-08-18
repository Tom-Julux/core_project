from napari._qt.layer_controls.qt_points_controls import QtPointsControls


class CustomQtPointsControls(QtPointsControls):
    def __init__(self, layer):
        super().__init__(layer)
        fields_to_hide = [
            self.faceColorEdit,
            self.borderColorEdit,
            self.symbolComboBox,
            self.textDispCheckBox,
            self.outOfSliceCheckBox,
            # self.sizeSlider,
        ]

        for field in fields_to_hide:
            label_item = self.layout().labelForField(field)
            field.hide()
            label_item.hide()
        
        self.addition_button.setChecked(True)
