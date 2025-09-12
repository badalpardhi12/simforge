import numpy as np
from simforge.ik_solver import solve_ik, HAS_PINOCCHIO
import pytest


@pytest.mark.skipif(not HAS_PINOCCHIO, reason="Pinocchio not available")
def test_solve_ik_identity():
    import pinocchio as pin
    import tempfile
    
    # Simple 1-link revolute robot URDF
    urdf_content = """<?xml version="1.0"?>
    <robot name="test_robot">
      <link name="base_link"/>
      <joint name="joint1" type="revolute">
        <parent link="base_link"/>
        <child link="link1"/>
        <origin xyz="0 0 0" rpy="0 0 0"/>
        <axis xyz="0 0 1"/>
        <limit lower="-3.14" upper="3.14" effort="1" velocity="1"/>
      </joint>
      <link name="link1"/>
    </robot>"""
    
    with tempfile.NamedTemporaryFile(mode='w', suffix='.urdf', delete=False) as f:
        f.write(urdf_content)
        urdf_path = f.name
    
    try:
        model = pin.buildModelFromUrdf(urdf_path)
        data = model.createData()
        
        q_init = np.zeros(model.nq)
        target_pos = (0.0, 0.0, 0.0)
        target_quat = (1.0, 0.0, 0.0, 0.0)  # Identity quaternion
        
        result = solve_ik(
            model, data, "link1", q_init, target_pos, target_quat, max_iters=10
        )
        
        assert result is not None
        assert len(result) == model.nq
        
    finally:
        import os
        os.unlink(urdf_path)
