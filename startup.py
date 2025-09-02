import napari
from napari import Viewer

viewer = Viewer()

from demo_widget import DemoWidget

widget = DemoWidget(viewer)

viewer.window.add_dock_widget(
    widget, name="core tool demo", area="left"
)

widget.load_demo("Mask")

napari.run()