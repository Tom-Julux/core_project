The goal of this file is to give AI coding agents immediate, practical context for working in this repository.

1) Quick architecture summary
- This repo contains multiple napari plugins and tools under top-level folders named `napari-*` (for example `napari-interactive`, `napari-label-metrics`, `napari_toolkit`).
- `napari-interactive` hosts an interactive segmentation UI and a bundled MedSAM2 integration (see `src/MedSAM2/` and `src/napari_interactive/`). The MedSAM2 code uses heavy GPU workloads and a checkpoint `MedSAM2_latest.pt` placed at the repo root or mounted into the container.
- Plugins are standard Python packages (see `pyproject.toml` in each plugin source) and are intended to be installed editable during development.

2) Essential developer workflows (concrete commands)
- Install dev dependencies and local plugins (recommended during local development):
  - See `install-debug.sh` — it runs `pip install napari[all]`, OpenCV/SimpleITK, and `pip install -e` for the plugin folders.
- Run the GUI inside Docker (X11 + GPU):
  - Use `./script.sh` to build `napari_core` and run a container with DISPLAY and GPU forwarded (the script mounts example data and startup.py into the container). Update volume mounts in the script to match your host paths.
  - For a debug image that mounts the repository into the container, use `./script_debug.sh` (builds `napari_core_debug` from `Dockerfile.debug`).
- Web/headless mode: set `USE_WEB` when running the container. `start.sh` checks `USE_WEB` and will start xpra/html mode instead of launching Napari directly.

3) Important environment flags and runtime notes
- USE_DEBUG: when set inside container `start.sh` iterates `/app/napari-*` and runs `pip install -e` for each napari plugin directory. Useful when you mount the workspace into the container for live edit cycles.
- USE_WEB, XPRA_*: `start.sh` expects xpra-related env vars to run the web-accessible front-end; see `start.sh` for exact options (ports, DISPLAY, XPRA_XVFB_SCREEN).
- GPU / CUDA: MedSAM2 expects CUDA. `src/MedSAM2/app.py` sets `os.environ["TORCH_CUDNN_SDPA_ENABLED"] = "1"`, uses autocast/bfloat16, and checks for compute capability >=8 for TF32 fallbacks — target machines with modern CUDA GPUs.
- Large files: `MedSAM2_latest.pt` is required for many experiments; either keep it at repo root or mount it into `/app/MedSAM2_latest.pt` when running the container (see `script.sh` volume example).

4) Project-specific code patterns and examples agents should know
- Napari plugin entry points: each plugin is a normal Python package under `src/<package_name>` with top-level `pyproject.toml`. Look for widget classes in `src/napari_interactive/` (e.g. `InteractiveSegmentationWidget2DSAM`) to find UI entrypoints.
- MedSAM2 runtime pattern: predictor state is commonly passed as a tuple (predictor, inference_state, image_predictor). Temporary frame outputs are written under `/tmp/output_frames/<session_id>/` and masks to `/tmp/output_masks/<session_id>/`.
- Long-running processes: MedSAM2 spawns threads and may use ffmpeg/moviepy; expect CPU/GPU and I/O heavy operations in `src/MedSAM2/app.py`.
- Packaging notes: some subpackages include compiled extensions or custom `setup.py` (see `src/MedSAM2/setup.py`) — editable installs (`pip install -e`) may trigger builds.

5) Tests, CI and release notes
- Each plugin contains a `.github/workflows/test_and_deploy` workflow and uses `tox`-style platform tests. To run local tests inspect `tox.ini` in the plugin folders and mimic the GitHub Actions runner environment when necessary.

6) External dependencies / system packages to be aware of
- FFmpeg (used by MedSAM2 for frame extraction), xvfb/xpra (web mode), display/X11 libraries (for GUI), and GPU drivers. If running locally, ensure these are installed or use the provided Docker images which package them.

7) Fast pointers for common AI dev tasks
- Add a new napari widget: put the code under `napari-interactive/src/napari_interactive/widgets` (follow existing widget naming patterns), update exports in `src/napari_interactive/__init__.py`, and verify `pyproject.toml` for package metadata.
- Debugging GPU issues: reproduce inside the Docker image (`./script_debug.sh`) which mounts your repo and exposes DISPLAY and GPUs. This is the closest environment to CI and includes system deps.

8) When in doubt (useful files to open first)
- `start.sh`, `install-debug.sh`, `script.sh`, `script_debug.sh` — runtime and developer scripts.
- `src/MedSAM2/app.py` — heavy runtime logic & model usage patterns (gradio requirement and ffmpeg usage noted in header).
- `src/MedSAM2/setup.py` — build instructions for native/compiled extensions.
- `src/napari_interactive/` — main plugin implementation and widget classes.

If any of the above sections are unclear or you want more examples (e.g., a minimal reproduce Docker command, or an example run for MedSAM2 inference), tell me which part to expand and I'll iterate.

9) Focus: `napari-interactive` widgets (what to change here)
- Location: `napari-interactive/src/napari_interactive/` contains the widget implementations. Key files:
  - `base_widget.py` — contains `InteractiveSegmentationWidgetBase` (widget lifecycle, prompt/preview layer management, threading helpers). New widget classes should subclass the project-specific base.
  - `_widget_2d_sam.py`, `_widget_3d_sam.py`, `_widget_3d_nni.py`, `_widget_3d_noregistration.py`, `_widget_2d_noregistration.py` — concrete widgets demonstrating how to implement `load_model`, `predict`, `reset_model`, and GUI hyperparameter hooks.
  - `multiple_viewer_widget.py` — example of multi-view synchronization; useful if you need multiple viewer instances.
  - `napari.yaml` — plugin manifest that registers commands and widgets shown in Napari. Use the existing entries as a template when adding new widgets.

- Widget contract (common methods & data flows):
  - __init__(viewer):__ call super(), build GUI, call `load_model()` and set any default GUI state.
  - load_model(self): load model weights / predictors. Use the existing pattern (see `_widget_2d_sam.py`) where a checkpoint path is resolved relative to `/app/MedSAM2_latest.pt` or repository root. Respect CPU/GPU availability via `torch.cuda.is_available()`.
  - predict(self): gather prompt layers (see `self.prompt_layers` keys like `'point_positive'`, `'point_negative'`, `'mask'`), prepare frame/volume data from the selected image layer, call the predictor API, and call `self.add_prediction_to_preview(new_mask, indices)` to update the preview layer.
  - reset_model(self): clear predictor state and any cached in-memory objects.
  - setup_hyperparameter_gui(self, _layout): expose any model thresholds / sliders; call `self.on_hyperparameter_update()` to trigger auto-run if `self.autorun_ckbx` is checked.

- Prompt layer conventions:
  - Prompt layers are created and tracked in `self.prompt_layers` using wrapper classes declared in `base_widget.py` (e.g., `PointPromptLayer`, `ScribblePromptLayer`). Use these classes instead of raw napari layer types to ensure correct custom Qt controls are used.
  - Supported prompt types are declared via `supported_prompt_types` property in each widget subclass. `base_widget` will create appropriate prompt layers and connect `events.data` to `on_prompt_update_event`.

- Preview and export patterns:
  - `setup_preview_layer()` creates a `Labels` layer named `'Preview Layer'` and stores data in `self.preview_label_data`. Use `add_prediction_to_preview(new_mask, indices)` to merge predicted mask(s) into that array.
  - To export: call `self._viewer.add_layer(...)` or use provided export helpers (there is an "Export to layer" button wired in `build_gui`). Respect `self.overwrite_existing_mm_ckbx` when merging.

- Adding a new widget (step-by-step):
  1. Add `_widget_<name>.py` under `napari-interactive/src/napari_interactive/` and implement a class that subclasses `InteractiveSegmentationWidgetBase` (or a 2D/3D base if present).
  2. Implement `supported_prompt_types`, `load_model`, `predict`, and `setup_hyperparameter_gui` at minimum. Follow patterns in `_widget_2d_sam.py`.
  3. Export the class in `src/napari_interactive/__init__.py` (add the class name to `__all__` and import it).
  4. Register the widget in `src/napari_interactive/napari.yaml` under `contributions.widgets` and add a matching `commands` entry. Use the existing `python_name: "napari_interactive:InteractiveSegmentationWidget2DSAM"` format.
  5. Install the plugin editable (`pip install -e napari-interactive`) or enable `USE_DEBUG` in container so changes are picked up.

- Common pitfalls and tips:
  - Predictor loading may allocate GPU memory. Use `torch.cuda.is_available()` and safe reset patterns. See how `_widget_2d_sam.py` resolves the checkpoint path and sets `self.predictor.mask_threshold`.
  - Be careful with dimension ordering — widgets re-order frames with `self._viewer.dims.order` before calling model code.
  - When adding new prompt layer types, wire `layer.events.data.connect(self.on_prompt_update_event)` to trigger predictions.
  - Keep UI updates on the main thread; use `@thread_worker` (already used in `base_widget.py`) for background prediction loops.

Requirements coverage: this change only updates documentation in `.github/copilot-instructions.md` and does not change runtime code.

If you'd like, I can now scaffold a new example widget file (`_widget_example.py`) that follows the exact contract above and wire it into `__init__.py` and `napari.yaml` so you can see a working template — shall I create that?
