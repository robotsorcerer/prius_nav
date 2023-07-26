# Exploratorio
To experiment with the utilities here, it is necessary to run the `car_demo`.
To do so, from the catkin workspace root, execute:

```bash
# Source the ros environment
source /opt/ros/noetic/setup.bash

# Build all packages
colcon build

# Source ros packages and launch the car demo
source install/setup.bash
roslaunch src/prius_autonav/car_demo/launch/demo.launch
```

## Online Replay Buffer
After having launched the `car_demo` as outlined above,
the online replay buffer node can be launched with

```bash
rosrun exploratorio camera_replay.py
```

To test the replay buffer, one may instead launch

```bash
rosrun exploratorio test_camera_replay.py
```

This will launch the replay buffer and periodically display the
amount of data collected for each camera sensor. Additionally,
once enough data has been collected, it will sample batches of
video sequences from the replay buffer and log some (meta)data from
the samples.

## Scene Parsing
After having launched the `car_demo` as outlined above,
simply run the test script (again from the catkin workspace root):

```bash
rosrun exploratorio test_scene_parser.py
```

This script will periodically print out metadata of the sensor data collected
by the `SceneParser` class from the Prius's laser scanners and cameras.
