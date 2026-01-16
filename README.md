# MCAP Bag Processor

Offline MCAP bag processor for adding tf_static, camera_info, and filtering pointclouds. 
Used to add intrinsic and extrinsic calibration to our tractor dataset (v1), since we had no sufficient calibration during recording time.

## Features

- **tf_static generation**: Parse URDF and generate static transforms
- **camera_info generation**: Match camera intrinsics to image timestamps
- **Pointcloud filtering**: Alpha channel + voxel-based outlier filtering for ZED cameras
- **Simple GUI**: File pickers, progress bar, and log output

## Installation

```bash
cd ~/ros2_ws/src
pip install mcap mcap-ros2-support

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

### Features

1. **Input MCAP**: Select your input rosbag (MCAP format)
2. **Output Directory**: Choose where to save the processed bag
3. **URDF File**: Select the robot URDF for tf_static generation
4. **Calibration Directory**: Select directory with `*.intrinsics.yaml` files

### Processing Options

- **Generate tf_static from URDF**: Extract all fixed joints as static transforms
- **Add map->odom->base_link TF**: Add identity transforms for localization chain
- **Generate camera_info for images**: Create CameraInfo messages matched to image timestamps
- **Filter ZED pointcloud**: Apply alpha filter (confidence) + voxel outlier removal

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
│   ├── main.py          # GUI application
│   ├── processor.py     # Core MCAP processing
│   ├── tf_generator.py  # URDF parsing + tf_static
│   ├── camera_info.py   # Camera intrinsics loader
│   └── pointcloud.py    # Pointcloud filtering
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

