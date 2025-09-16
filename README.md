# Simforge

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
cd simforge

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
simforge run --config env_configs/ur5e_env.yaml

# Create template configuration files
simforge init
```

## Configuration

Simforge uses YAML configuration files to define simulation environments. Example:

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

- `simforge/config_reader.py`: Pydantic-based configuration models
- `simforge/movement_controller.py`: Main control logic and command processing
- `simforge/ik_drake.py`: Drake-powered inverse kinematics
- `simforge/path_planner.py`: OMPL-based motion planning
- `simforge/collision_checker.py`: FCL-based collision detection
- `simforge/control_gui.py`: wxPython control interface
- `simforge/genesis_renderer.py`: Genesis backend wrapper
- `simforge/logging_utils.py`: Colored logging utilities
- `simforge/transformations.py`: Mathematical transformation utilities

## API Usage

```python
from simforge.config_reader import SimforgeConfig
from simforge.movement_controller import MovementController

# Load configuration
config = SimforgeConfig.from_yaml("env_configs/ur5e_env.yaml")

# Create controller
controller = MovementController(config)

# Start simulation
controller.start()

# Control robot
controller.set_joint_targets("ur5e", [0, -90, 90, -90, -90, 0])
controller.move_cartesian("ur5e", (0.5, 0.2, 0.3), (0, 0, 0))

# Stop simulation
controller.stop()
```

## Testing

```bash
# Run all tests
pytest

# Run specific test module
pytest tests/test_config_reader.py

# Run with coverage
pytest --cov=simforge --cov-report=html
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
source .simforge/bin/activate  # or your venv path

# Run tests
pytest

# Run the application
simforge run --config env_configs/ur5e_env.yaml
```

## Troubleshooting

- **Genesis viewer doesn't open**: Set `scene.show_viewer: false` in headless environments
- **macOS Tkinter issues**: Install wxPython extra or ensure Tkinter support
- **CUDA issues**: Verify NVIDIA drivers and PyTorch CUDA installation
- **Import errors**: Ensure all optional dependencies are installed for your use case
- **Performance issues**: Try `scene.backend: cpu` if GPU acceleration fails

## License

Proprietary

