# Extending the Plugin with New Models

The plugin is designed to be extensible, allowing you to integrate new segmentation models with minimal effort.

## Steps to Add a New Model

1. **Prepare the Model**:
   - Ensure the model is compatible with the plugin's input and output formats.
   - Save the model weights in a supported format (e.g., PyTorch `.pt` files).

2. **Implement the Model Loader**:
   - Create a new Python file in the `src/napari_interactive/` directory.
   - Define a function to load the model and return a predictor object.

3. **Update the Widget**:
   - Modify the `load_model` method in the widget class to include the new model.
   - Add any required hyperparameter controls in the `setup_hyperparameter_gui` method.

4. **Test the Integration**:
   - Load the plugin in Napari and verify that the new model works as expected.
   - Check for GPU compatibility and performance.

5. **Document the Model**:
   - Update the user guide to include instructions for using the new model.

## Example
```python
from my_model_library import MyModel

def load_my_model(checkpoint_path, device):
    model = MyModel()
    model.load_weights(checkpoint_path)
    model.to(device)
    return model
```

## Placeholder for Images
- Add images showing the integration process and results.