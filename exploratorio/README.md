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

## Landmark Discovery
The `landmarker.py` script provides a ROS node that identifies landmarks
from the collection of camera observations. The node creates a ROS tf tree of
frames corresponding to the centroids of the landmarks. Landmark frames from a
given scene form a chain in the tf tree.

To launch the node, run

```bash
rosrun exploratorio landmarker.py
```

## Point Cloud Generation
First, you must download the pretrained models for image segmentation (Segment
Anything) and depth prediction (SfM and vision transformer). Download the file `models/sg_data.tar` from
the OneDrive, place it in the workspace root, and extract it.

Next, install the python dependencies listed in `exploratorio/scene_generation_requirements.txt`.

We are now ready to run the scene generation script. Be sure to source the ROS setup, and execute

```bash
rosrun exploratorio scene_generator.py
```

This will initialize some python objects and open up PDB. Enter the command `interact` to
enter a Python shell, where we will interact with the scene generation object. This object
is the variable `sg` that has been initialized.

The shell also has variables `img_test` and `img_multi`, containing image data from the car's
sensors. The `img_test` image is simply an image from the car's front sensor, whereas `img_multi`
is a dictionary from keys in `CAMERA_KEYS` to images corresponding to the relevant camera.
Additionally, the `data` is a dictionary mapping from `CAMERA_KEYS` to a sequence of images from
the corresponding camera.

You can use `sg.preview(img)` for a single image `img` to view side-by-side depictions of
the 'raw' image and the depth prediction from the model. The depth prediction will by default
'mask out' the pixels identified as background (and therefore only show depth for the predicted
obstacles). To show the depth prediction for the whole image, run `sg.preview(img, mask=False)`.

To simply generate depth estimates without visualizing them, run `sg.masked_depth_image(img)` for
an image `img`. This returns the depth prediction as a numpy array.

You can use `sg.scene(imgs)` called with
a dictionary of images (like `img_multi`) to generate the pointcloud estimated from the camera
observations, which will appear in an interactive visualizer. Press `q` within the visualizer
to quit. You can also use `sg.scene_anim(imgs)`
called with a dictionary of images (like `img_multi`) to generate an animation of the rendered
pointcloud, which will be saved to `scene.gif`.

To simply generate a pointcloud without visualizing it, run `sg.multi_masked_pointcloud(imgs)` for a
dictionary of images (like `img_multi`). This returns a list of pointclouds (one for each camera).
The points are positioned relative to the car's local frame.

### Depth Estimation and Scene Generation Jobs

There is also a number of predefined jobs for convenience. These are specified with the `--job`
flag to `scene_generator.py`. These are described briefly below.

#### Depth Estimation Analysis

```bash
rosrun exploratorio scene_generator.py --job analyze-depth \
    [--depth-maxlen N1] [--depth-start-frame N2] [--depth-num-pixels N3] [--depth-bootstrap-samples N4]
```

Evaluates the robustness of pixelwise depth estimates in the presence of observation noise.
This involves two experiments:

1. For a bunch of randomly chosen pixels, estimate the depth prediction at those
   pixels over some time horizon where the camera is held fixed. Plot the evolution
   of depth estimates for the pixel with the largest variance over the horizon.
2. Use bootstrapping to estimate the distribution over depth standard deviations over
   the horizon.

Here, the arguments `N1`, `N2`, `N3`, and `N4` are all integers:
- `N1` is the length of the time horizon over which to compute depth predictions;
- `N2` is the time index for the first frame to start computing depth predictions;
- `N3` is the number of random pixel coordinates at which the depth will be evaluated;
- `N4` is the number of bootstrap samples for estimating the posterior standard deviation.

#### Scene Rendering (Interactive)

```bash
rosrun exploratorio scene_generator.py --job scene \
    [--scene-mesh] [--no-landmarks]
```

Generates 3D scene from camera images, and displays the 3D scene in an interactive visualizer.

The `--scene-mesh` flag triggers mesh rendering. When `--no-landmarks` is specified, the landmark
frames will not be rendered.

#### Scene Animation

```bash
rosrun exploratorio scene_generator.py --job scene-anim \
    [--scene-frames N] [--scene-mesh] [--no-landmarks]
```

Generates 3D scene from camera images, and generates a gif animation of the scene.

The `--scene-mesh` flag triggers mesh rendering. When `--no-landmarks` is specified, the landmark
frames will not be rendered. The integer `N` specifies how many frames to generate in the gif.

#### Pointcloud Streaming

```bash
rosrun exploratorio scene_generator.py --job stream-scene
```

Infers landmark positions and pointclouds from camera images, and publishes the landmark pointclouds
to the `/landmark_pointcloud` topic.

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
