# MCAP Bag Processor

Offline MCAP bag processor for adding tf_static, camera_info, and filtering pointclouds. 
Used to add intrinsic and extrinsic calibration to our tractor dataset (v1), since we had no sufficient calibration during recording time.

## Features

- **tf_static generation**: Parse URDF and generate static transforms
- **camera_info generation**: Match camera intrinsics to image timestamps
- **Pointcloud filtering**: Alpha channel + voxel-based outlier filtering for ZED cameras
- **LiDAR odometry**: KISS-ICP scan-to-map registration from Ouster pointclouds
- **Batch processing**: Select an input directory and process all `.mcap` files at once
- **Persistent preferences**: Default URDF and calibration paths saved across sessions
- **Modern GUI**: Grid layout, dark log panel, accent-styled buttons, progress per bag

## Installation

```bash
cd ~/ros2_ws/src
pip install mcap mcap-ros2-support

# Optional: install kiss-icp for LiDAR odometry
pip install kiss-icp

# Build the package
cd ~/ros2_ws
colcon build --packages-select mcap_bag_processor
source install/setup.bash
```

## Usage

### GUI Mode

```bash
ros2 run mcap_bag_processor mcap_processor
```

Or run directly:

```bash
python3 -m mcap_bag_processor.main
```

### Workflow

1. **Input Directory**: Select a directory containing `.mcap` rosbags (all bags are processed)
2. **Output Directory**: Choose a separate directory for the processed bags
3. **URDF File**: Select the robot URDF for tf_static generation
4. **Calibration Directory**: Select directory with `*.intrinsics.yaml` files

### Preferences

Click **Preferences...** in the top-right corner to set default paths for URDF and calibration
directory. These are stored in `~/.config/mcap_bag_processor/preferences.json` and persist
across sessions. Last-used input/output directories are also remembered.

### Processing Options

- **Generate tf_static from URDF**: Extract all fixed joints as static transforms
- **Add map->odom->base_link TF**: Add identity transforms for localization chain
- **Generate camera_info for images**: Create CameraInfo messages matched to image timestamps
- **Filter ZED pointcloud**: Apply alpha filter (confidence) + voxel outlier removal
- **Generate LiDAR odometry (KISS-ICP)**: Compute scan-to-map odometry from `/ouster/points`

## Default Topics

The processor handles these topics by default:

### Passthrough (unchanged)
- `/camera/*/image_raw/compressed` - Compressed camera images
- `/novatel/oem7/*` - GNSS data
- `/ouster/imu`, `/ouster/points` - LiDAR data
- `/zed/zed_node/odom` - ZED odometry
- `/zed/zed_node/*/compressed` - ZED camera images

### Generated
- `/tf_static` - Static transforms from URDF
- `/camera/*/image_raw/camera_info` - Camera intrinsics (matched to image timestamps)
- `/zed/zed_node/point_cloud/cloud_registered/filtered` - Filtered ZED pointcloud
- `/kiss_icp/odom` - LiDAR odometry (nav_msgs/Odometry)

## Pointcloud Filtering

The ZED pointcloud filter applies two stages:

1. **Alpha filter**: Keeps only points with alpha=255 (full stereo confidence)
2. **Voxel outlier filter**: Removes isolated points (voxel size 0.2m, min 2 points)

This removes ZED stereo artifacts and noise while preserving valid geometry.

## File Structure

```
mcap_bag_processor/
├── mcap_bag_processor/
│   ├── __init__.py
│   ├── main.py              # GUI application
│   ├── processor.py         # Core MCAP processing
│   ├── tf_generator.py      # URDF parsing + tf_static
│   ├── camera_info.py       # Camera intrinsics loader
│   ├── pointcloud.py        # Pointcloud filtering
│   ├── lidar_odometry.py    # KISS-ICP LiDAR odometry
│   └── preferences.py       # Persistent user preferences
├── setup.py
├── setup.cfg
├── package.xml
└── README.md
```

## Dependencies

- `mcap` - MCAP file format library
- `mcap-ros2-support` - ROS2 message serialization
- `numpy` - Pointcloud array processing
- `pyyaml` - YAML parsing
- `tkinter` - GUI (included with Python)
- `kiss-icp` - LiDAR odometry (optional, required for LiDAR odom feature)

