import numpy as np
from simforge.path_planner import plan_joint_path


def test_plan_joint_path():
    q_start = np.array([0.0, 1.0], dtype=np.float32)
    q_goal = np.array([1.0, 3.0], dtype=np.float32)
    waypoints, times = plan_joint_path(q_start, q_goal, resolution=5)
    
    assert waypoints.shape == (5, 2)
    assert len(times) == 5
    # Check endpoints
    assert np.allclose(waypoints[0], q_start)
    assert np.allclose(waypoints[-1], q_goal)
    # Check time ordering
    assert times[0] == 0.0
    assert all(times[i] <= times[i+1] for i in range(len(times)-1))

