# simforge_new

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

## Quick Start

### Prerequisites

- Python 3.10 or 3.11
- PyTorch (install based on your backend: CUDA/MPS/CPU)
- Genesis: `pip install genesis-world`

### Installation

```bash
# Clone the repository
git clone <repository-url>
cd simforge_new

# Install in development mode
pip install -e .

# Optional dependencies
pip install -e ".[planning]"  # For motion planning with OMPL
pip install -e ".[gui]"       # For wxPython GUI (alternative to Qt)
pip install -e ".[drake]"     # For advanced IK solving
pip install -e ".[dev]"       # For development tools
```

### Basic Usage

```bash
# Run with GUI using a configuration file
simforge_new run --config env_configs/ur5e_env.yaml

# Launch the prototype protocol simulation workspace
simforge_new proto_sim --config simforge_new/environment/presets/face_robot.yaml

# Create template configuration files
simforge_new init
```

## Configuration

simforge_new uses YAML configuration files to define simulation environments. Example:

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

## Architecture

- `simforge_new/core/config_schema.py`: Pydantic-based configuration models
- `simforge_new/control/session.py`: Session orchestration and service factories
- `simforge_new/services/ik/genesis_solver.py`: Genesis-native inverse kinematics
- `simforge_new/services/planning/genesis_planner.py`: Genesis-native motion planning
- `simforge_new/services/collision/genesis_world.py`: Collision validation via Genesis collider state
- `simforge_new/interfaces/cli.py`: Command-line entry points (including `proto_sim`)
- `simforge_new/core/transformations.py`: Quaternion and rotation utilities

## API Usage

```python
from simforge_new.interfaces.cli import main

# Launch proto_sim with a preset environment
main([
    "proto_sim",
    "--config",
    "simforge_new/environment/presets/face_robot.yaml",
])
```

## Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_config_reader.py

# Run with coverage
pytest --cov=simforge_new --cov-report=html
```

## Development

### Code Quality

```bash
# Format code
ruff format .

# Lint code
ruff check . --fix
```

### Virtual Environment

The codebase expects to run in a virtual environment. To debug or run code:

```bash
# Activate virtual environment
source .simforge_new/bin/activate  # or your venv path

# Run tests
pytest

# Run the application
simforge_new run --config env_configs/ur5e_env.yaml
```

## Troubleshooting

- **Genesis viewer doesn't open**: Set `scene.show_viewer: false` in headless environments
- **macOS Tkinter issues**: Install wxPython extra or ensure Tkinter support
- **CUDA issues**: Verify NVIDIA drivers and PyTorch CUDA installation
- **Import errors**: Ensure all optional dependencies are installed for your use case
- **Performance issues**: Try `scene.backend: cpu` if GPU acceleration fails

## License

Proprietary
