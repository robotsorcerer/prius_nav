# Scene Parsing
First, it is necessary to launch the `car_demo`. From the catkin workspace root:

```bash
source install/setup.bash
roslaunch src/prius_autonav/car_demo/launch/demo.launch
```

Next, simply run the test script (again from the catkin workspace root):

```bash
rosrun exploratorio test_scene_parser.py
```

This script will periodically print out metadata of the sensor data collected
by the `SceneParser` class from the Prius's laser scanners and cameras.
