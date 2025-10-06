# UI development

The UIs provided by plugins are QT-widgets. They can be created either manually or semi-automatically using magicgui. Since the plugins of the core-tool are generally more complex and extendable, the manual approach is used here. 

For usage of magicgui, please refer to the [napari-magicgui documentation](https://napari.org/magicgui/).

To make the development of new UIs easier, a fork of the [napari_toolkit](https://github.com/MIC-DKFZ/napari_toolkit) is used.

## Napari Toolkit

The napari toolkit provides a set of utility functions to make the development of napari plugins easier, originally by the [Applied Computer Vision Lab (ACVL) of Helmholtz Imaging](https://github.com/MIC-DKFZ/napari_toolkit).


### Layouts

The napari-widget 

### Widget Value Handling

Easily get and set values for QWidgets in your Napari plugin. Note: These functions work for many widgets but are not guaranteed to support all

```python
from napari_toolkit.utils.widget_getter import get_value
from napari_toolkit.utils.widget_setter import set_value

set_value(<QWidget>,<value>)        # Sets the value of widget
_ = get_value(<QWidget>,<value>)    # Retrieves the value of a widget
```

### Individual Widget examples


## Tips and Tricks

During development keep the following things in mind:

- QT-widgets are detachable windows, so if your plugin does not fit within the single main napari window, you can simply drag it out and work with it in a separate window.

## Hot Reloading
If you setup the development environment as described in the [development setup guide](development/setup.md), you can use an notebook to force-reload your plugin after making changes. This is much faster than restarting napari each time.