# How to Install wxPython for GUI Support

wxPython is now buildable on your system since we've installed the required GTK3 development libraries.

## Quick Install (takes 10-15 minutes)

```bash
# Activate the virtual environment
source .simforge/bin/activate

# Install wxPython (this will compile from source)
pip install wxpython
```

The build process will:
- Download wxPython source (~58MB)
- Compile wxWidgets library
- Build Python bindings
- Install the compiled package

**Note:** This is a one-time build. Once installed, it will be cached in your virtual environment.

## After Installation

Test the installation:
```bash
source .simforge/bin/activate
python -c "import wx; print('wxPython version:', wx.version())"
```

Check simforge status:
```bash
python -c "
from simforge_new.interfaces.gui import HAS_WX
print('wxPython GUI Available:', HAS_WX)
"
```

## Alternative: Pre-built Wheels

For faster installation, you can try finding pre-built wheels for Ubuntu 24.04 ARM64:
```bash
# Check wxPython extras for ARM builds
pip install --extra-index-url https://extras.wxpython.org/wxPython4/extras/linux/gtk3/ubuntu-24.04/ wxpython
```

Note: Pre-built wheels may not be available for all Ubuntu/ARM combinations.

## If You Don't Need GUI

The simforge package works perfectly without wxPython for:
- Programmatic control via Python API
- CLI-based operation
- Headless simulation
- Automated testing
- Batch processing

Only interactive GUI features require wxPython.
