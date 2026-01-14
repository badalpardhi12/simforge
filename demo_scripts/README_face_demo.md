# Face Robot Demo

This demo script loads the `face_robot.yaml` configuration and moves the TX2-90XL robot through a sequence of 4 poses in each of the 8 face_object reference frames.

## Running the Demo

Make sure you're in the simforge environment:

```bash
cd /home/badal/simforge
. .simforge/bin/activate  # or however you activate your environment
```

Then run the demo:

```bash
python demo_scripts/face_robot_demo.py
```

## Movement Completion Detection

The demo script uses sophisticated completion detection that goes beyond simple timeouts:

### **Multi-Stage Completion Criteria**
1. **Trajectory Execution**: Waits for the planned trajectory to finish executing
2. **Pose Validation**: Ensures the robot has actually reached the target pose within tolerance
3. **Refinement**: If needed, performs additional IK refinement to achieve tighter tolerances

### **Completion States**
- ✅ **Active Trajectory**: Movement planning and execution in progress
- ✅ **Trajectory Complete**: Planned path finished, but pose validation pending
- ✅ **Pose Validated**: Robot confirmed to be at target pose within tolerance
- ✅ **Fully Complete**: Movement finished and validated

### **Error Handling**
- **Planning Failures**: Detected when no trajectory starts within 10 seconds
- **Execution Timeouts**: 60-second limit for complete movement execution
- **Validation Issues**: Handles cases where pose cannot be achieved within tolerance
- **Graceful Degradation**: Continues to next pose on failures

## Movement Sequence Details

- **Poses per Face Object**: 4 poses with Y=0.4m (as requested)
- **Frame Reference**: Each pose is relative to the respective face_object's local coordinate system
- **Orientation**: All poses use RPY (90°, 0°, 0°) - end effector pointing downward
- **Timing**: 30-second timeout per movement, 1-second pause between face objects

## Controls

- The Genesis viewer will open showing the robot and face objects
- Use Ctrl+C in the terminal to stop the demo
- The simulation will continue running after the sequence completes

## Configuration Details

- **Robot**: TX2-90XL at world origin
- **Face Objects**: 8 objects in a 1m radius circle at z=1.0m
- **Movement Frame**: Each movement is relative to the respective face_object's local frame
- **Timing**: 3 seconds between poses, 1 second between face objects