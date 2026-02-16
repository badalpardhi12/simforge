# simforge_genesis

Genesis-powered robot simulator with joint and Cartesian control, inverse kinematics, motion planning, and collision checking.

## Features

- **Multi-Robot Support**: Configure multiple robots in a single simulation
- **Joint & Cartesian Control**: Switch between joint-space and Cartesian control modes
- **Inverse Kinematics**: Drake-powered IK solver with robust seed generation
- **Motion Planning**: OMPL-based RRT-Connect planner with collision avoidance
- **Collision Detection**: FCL-based collision checking with mesh support
- **GUI Control**: wxPython-based control interface with real-time sliders
- **YAML Configuration**: Clean configuration system with inheritance
- **Genesis Integration**: Built on Genesis physics simulation engine

## Package Structure

```
simforge_genesis/
├── assets/            # Robot URDFs and meshes
├── control/           # Session, joint/Cartesian controllers
├── core/              # Config schema, transforms
├── demo/              # Demo scripts (face robot, etc.)
├── environment/       # Presets and YAML loader
├── infrastructure/    # Pinocchio cache and utilities
├── interfaces/        # CLI entry points
├── logging/           # Coloured logging setup
├── services/          # IK, planning, collision, real-robot drivers
└── tests/             # Unit and integration tests
```

## Prerequisites

- Python 3.10 or 3.11
- PyTorch (install based on your backend: CUDA / MPS / CPU)
- Genesis: `pip install genesis-world`

## Installation

From the **repository root**:

```bash
# Install in development mode
pip install -e .

# Optional dependency groups
pip install -e ".[planning]"  # OMPL motion planning
pip install -e ".[gui]"       # wxPython GUI
pip install -e ".[drake]"     # Advanced IK solving
pip install -e ".[dev]"       # pytest + ruff
```

## Usage

```bash
# Run with GUI using a configuration file
simforge_genesis run --config simforge_genesis/environment/presets/ur5e_env.yaml

# Launch the prototype protocol simulation workspace
simforge_genesis proto_sim --config simforge_genesis/environment/presets/face_robot.yaml

# Create template configuration files
simforge_genesis init
```

## Configuration

YAML configuration files define simulation environments. Example:

```yaml
scene:
  dt: 0.01
  gravity: [0.0, 0.0, -9.81]
  backend: gpu
  show_viewer: true

robots:
  - name: ur5e
    urdf: assets/ur5e/ur5e.urdf
    base_position: [0.0, 0.0, 0.0]
    end_effector_link: wrist_3_link
    control:
      joint_speed_limit: 1.0
      cartesian_speed_limit: 0.1

objects:
  - type: box
    name: table
    position: [0.5, 0.0, 0.0]
    size: [0.8, 0.8, 0.1]
```

Pre-built presets live in `environment/presets/` (UR5e, UR20, Meca500, face-robot, etc.).

## Architecture

| Module | Description |
|---|---|
| `core/config_schema.py` | Pydantic-based configuration models |
| `control/session.py` | Session orchestration and service factories |
| `services/ik/genesis_solver.py` | Genesis-native inverse kinematics |
| `services/planning/genesis_planner.py` | Genesis-native motion planning |
| `services/collision/genesis_world.py` | Collision validation via Genesis collider state |
| `interfaces/cli.py` | Command-line entry points (`run`, `proto_sim`, `init`, `demo`) |
| `core/transformations.py` | Quaternion and rotation utilities |

## API Usage

```python
from simforge_genesis.interfaces.cli import main

main([
    "proto_sim",
    "--config",
    "simforge_genesis/environment/presets/face_robot.yaml",
])
```

## Testing

```bash
pytest simforge_genesis/tests/

# With coverage
pytest --cov=simforge_genesis --cov-report=html
```

## Troubleshooting

- **Genesis viewer doesn't open**: Set `scene.show_viewer: false` in headless environments
- **macOS Tkinter issues**: Install wxPython extra or ensure Tkinter support
- **CUDA issues**: Verify NVIDIA drivers and PyTorch CUDA installation
- **Import errors**: Ensure all optional dependencies are installed for your use case
- **Performance issues**: Try `scene.backend: cpu` if GPU acceleration fails

## License

Proprietary
