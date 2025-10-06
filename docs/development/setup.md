# Development Setup

To set up a development environment for creating and testing napari plugins, follow these steps:

1. **Clone the Repository**: Start by cloning the repository containing the core-tool and its plugins.

   ```bash
   git clone

    cd core-tool
    ```
2. **Create a Virtual Environment**: It's recommended to use a virtual environment to manage dependencies.
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows use `venv\Scripts\activate`
    ```
3. **Install Dependencies**: Install the required dependencies using pip. Make sure to include development dependencies if available.
    ```bash
    pip install -e .[dev]
    ```
4. **Install Napari**: If napari is not included in the dependencies, install it separately.
    ```bash
    pip install napari
    ```
5. **Run Napari**: You can now run napari and test the plugins.
    ```bash
    napari
    ```

## Hot Reloading

If you setup the development environment as described above, you can use a notebook to force-reload your plugin after making changes. This is much faster than restarting napari each time.

```python
import importlib
import your_plugin_module  # Replace with your actual plugin module name
importlib.reload(your_plugin_module)
```

See test.ipynb for an example of how to do this.

We recomment you starting by creating a demo, this way the reload notebook can be used to test your changes quickly. 
