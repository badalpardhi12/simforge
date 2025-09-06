# Simforge

Genesis-powered robot simulator with joint and Cartesian control, a simple GUI, and YAML-based environment configs.

## Quick Start

- Python: 3.10 or 3.11 recommended
- Install PyTorch first per your backend (CUDA/MPS/CPU). See https://pytorch.org/get-started/locally/
- Install Genesis: `pip install genesis-world`

Install Simforge (editable):

```
pip install -e .
```

Run the GUI with a template config:

```
simforge run --config env_configs/ur5e_env.yaml
```

If you need a fresh set of config templates:

```
simforge init
```

## Notes
- Backend auto-selects GPU if available. Override in YAML via `scene.backend: "cpu"|"gpu"|"cuda"`.
- Joint limits for the GUI are parsed from URDF directly, so sliders respect model constraints.
- Optional motion planning and IK integrate with Genesis/OMPL if installed.

## Structure

- `simforge/`: package with CLI, simulator, controller, GUI
- `env_configs/`: example YAML config files
- `assets/`: provided robot URDFs and meshes (already present)

## Optional dependencies
- OMPL for motion planning: `pip install ompl`
- Qt GUI (alternative to Tkinter): `pip install PySide6`

## Troubleshooting
- If the Genesis viewer doesn’t open in headless environments, set `scene.show_viewer: false`.
- On macOS, ensure your Python has Tkinter support or use the Qt extra.
- For CUDA, ensure NVIDIA drivers are installed; verify with `python -c "import torch; print(torch.cuda.is_available())"`.

