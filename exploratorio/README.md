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

## SfM
This project performs unsupervised learning from video data to predict
depth maps, using the [SfM algorithm](https://arxiv.org/abs/1704.07813).

The `scripts/sfm_trainer.py` script provides a node that trains a SfM model
either from online data that is being collected, or from an offline dataset.
This script can be executed by

```bash
# Online training
rosrun exploratorio sfm_trainer \
    --buffer-size <capacity of replay buffer> \
    --num-steps <number of training updates>

# Offline training
rosrun exploratorio sfm_trainer \
    --offline \
    --data-path <path of replay buffer dataset pickle> \
    --sensor-path <path of sensor info pickle> \
    --num-steps <number of training steps>
```

In the case of offline training, the `--data-path` and `--sensor-path` arguments
default to `dataset.p` and `sensor-info.p`, which is where the online replay buffer
stores these objects by default.

To track experiment progress, execute the following from the catkin workspace root:

```bash
aim up
```

and navigate to the `localhost:43800` in your browser. Upon selecting a training run,
loss metrics are logged in the `Metrics` pane, while image predictions are logged in the
`Images` pane.

In order to experiment with the SfM algorithm, there is convenient python
script that loads up a node that builds a replay buffer and instantiates an
SfM model to interact with. Run

```bash
source src/prius_autonav/pydebug.bash
```

from the workspace root (only needs to be done once per shell). Then, execute

```bash
rosipython exploratorio test_sfm_integ.py
```

to get an iPython shell with the buffer and SfM model.

## Scene Parsing
After having launched the `car_demo` as outlined above,
simply run the test script (again from the catkin workspace root):

```bash
rosrun exploratorio test_scene_parser.py
```

This script will periodically print out metadata of the sensor data collected
by the `SceneParser` class from the Prius's laser scanners and cameras.
